
from collections import OrderedDict
from math import floor
import json

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from asreview import ASReviewData
from asreview.analysis import Analysis
from asreview.entry_points.base import BaseEntryPoint
from asreview.models import get_model
from asreview.balance_strategies import get_balance_model
from asreview.feature_extraction import get_feature_model
from asreview.state import open_state
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

from asreviewcontrib.pro.powertail import ExpTailNorm


def sample_proba(proba):
    sample_idx = []
    order = np.argsort(-proba)
    sorted_proba = proba[order]
    cum_proba = np.cumsum(sorted_proba)
    last_bound = order[0]
    for i in range(1, floor(cum_proba[-1])+1):
        sample_idx.append(last_bound)
        last_bound = np.searchsorted(cum_proba, i, side='right')
        last_bound = order[last_bound]
#         print(i, last_bound, proba[last_bound])
#     if np.random.random_sample() < cum_proba[-1]-floor(cum_proba[-1]):
#         sample_idx.append(last_bound)
    return sample_idx, cum_proba[-1]


def discrete_norm_dist(dist, train_percentage, bins):
    norm_cdf = dist.cdf(bins)
    norm_pdf = train_percentage*(norm_cdf[1:]-norm_cdf[:-1])
    norm_hist = norm_pdf/norm_pdf.sum()
    return norm_hist/(bins[1]-bins[0])


def percentage_found(norm_opt_cum_df, train_percentage, bins, mu, sigma):
    normalized_bins = (bins-mu)/sigma
    x_cum = norm_opt_cum_df[0]
    y_cum = norm_opt_cum_df[1]
    d_cum = y_cum[1]-y_cum[0]

    prob_found = 0
    for i_bin in range(len(train_percentage)):
        bin_start = normalized_bins[i_bin]
        bin_end = normalized_bins[i_bin+1]

        i_cum_start = np.searchsorted(x_cum, bin_start)
        i_cum_end = np.searchsorted(x_cum, bin_end)

        if i_cum_start == len(x_cum):
            continue

        y_start = y_cum[i_cum_start] - d_cum
        if i_cum_end == len(x_cum):
            y_end = 1
        else:
            y_end = y_cum[i_cum_end] - d_cum

        prob = y_end-y_start
        prob_found += prob*train_percentage[i_bin]

    return prob_found


def prob_all_found(min_df, df_pool, mu, sigma):
    df_max = np.max(df_pool)
    df_max_norm = (df_max-mu)/sigma

    x_min_df = min_df[0]
    y_min_df = min_df[1]

    i_min_df = np.searchsorted(x_min_df, df_max_norm)

    if i_min_df == 0:
        return 1.0
    else:
        return 1-y_min_df[i_min_df - 1]


def kl_divergence(train_dist, norm_dist):
    klb = 0
    for i in range(len(norm_dist)):
        if norm_dist[i] == 0 or train_dist[i] == 0:
            continue
        klb += train_dist[i]*np.log(train_dist[i]/norm_dist[i])
    return klb


def log_likelihood(train_dist, expected_dist):
    likelihood = 0
    for i_bin in range(len(expected_dist)):
        if train_dist[i_bin]:
            likelihood += train_dist[i_bin] * np.log(expected_dist[i_bin])
    return -likelihood


def estimate_inclusions(train_idx, pool_idx, X, y, optimization_fp, plot=False):
    model = get_model("nb")
    balance_model = get_balance_model("double")
    X_train, y_train = balance_model.sample(X, y, train_idx, {})

    with open(optimization_fp, "r") as f:
        opt_results = json.load(f)

    model.fit(X_train, y_train)
    proba = model.predict_proba(X)[:, 1]
    df_all_corrected = -np.log(1/proba-1)

    train_one_idx = np.where(y[train_idx] == 1)[0]
    train_zero_idx = np.where(y[train_idx] == 0)[0]
    correct_one_proba = []

    for _ in range(10):
        if len(train_one_idx) == 1:
            correct_one_proba.append(df_all_corrected[train_idx[0]])
            continue
        for rel_train_idx in train_one_idx:
            new_train_idx = np.delete(train_idx, rel_train_idx)
            X_train, y_train = balance_model.sample(X, y, new_train_idx, {})
            model.fit(X_train, y_train)
            correct_proba = model.predict_proba(X[train_idx[rel_train_idx]])[0, 1]
            correct_one_proba.append(correct_proba)

    correct_one_proba = np.array(correct_one_proba)

    df_one_corrected = -np.log(1/correct_one_proba-1)
    df_train_corrected = df_all_corrected[train_idx]
    df_train_zero = df_all_corrected[train_zero_idx]
    df_pool = df_all_corrected[pool_idx]

    df_all = np.concatenate((df_all_corrected, df_one_corrected))
    h_min = np.min(df_all)
    h_max = np.max(df_all)
    h_range = (h_min, h_max)
    n_bins = 40

    hist, bin_edges = np.histogram(df_one_corrected, bins=n_bins,
                                   range=h_range, density=True)
    hist_all, _ = np.histogram(df_all_corrected, bins=n_bins, range=h_range,
                               density=False)
    hist_pool, _ = np.histogram(df_pool, bins=n_bins, range=h_range,
                                density=False)
    hist_train_zero, _ = np.histogram(df_train_zero, bins=n_bins, range=h_range,
                                      density=False)
    hist_train_one, _ = np.histogram(df_one_corrected, bins=n_bins, range=h_range,
                                     density=False)
    hist_train, _ = np.histogram(df_train_corrected, bins=n_bins,
                                 range=h_range, density=False)

    perc_train = (hist_train_zero + hist_train_one/10 + 0.000001)/(
        hist_train_zero + hist_train_one/10 + hist_pool + 0.000001)
#     perc_train = hist_train/(hist_all+0.0001)

    def guess_func(x):
        dist = ExpTailNorm(*x, *opt_results["extra_param"])
        corrected_dist = discrete_norm_dist(dist, perc_train, bin_edges)
        return log_likelihood(hist, corrected_dist)

    mu_range = h_range
    est_sigma = np.sqrt(np.var(df_one_corrected))
    sigma_range = (0.7*est_sigma, 1.3*est_sigma)
    x0 = np.array((np.average(df_one_corrected), est_sigma))

    minim_result = minimize(fun=guess_func, x0=x0,
                            bounds=[mu_range, sigma_range])

#     print(minim_result)
    param = minim_result.x
    est_true_dist = ExpTailNorm(*param, *opt_results["extra_param"])

    est_found_dist = discrete_norm_dist(est_true_dist, perc_train, bin_edges)

    perc_found = percentage_found(opt_results["cum_df"], perc_train, bin_edges,
                                  param[0], param[1])

    p_all_found = prob_all_found(
        opt_results["min_df"], df_pool, param[0], param[1])
    if plot:
        plt.plot((bin_edges[1:]+bin_edges[:-1])/2, est_found_dist)
        plt.plot((bin_edges[1:]+bin_edges[:-1])/2, hist)
        plt.plot((bin_edges[1:]+bin_edges[:-1])/2, perc_train)
        plt.show()
    return np.sum(y[train_idx])/perc_found, p_all_found


class ErrorEntryPoint(BaseEntryPoint):
    description = "XXX."

    def __init__(self):
        super(ErrorEntryPoint, self).__init__()
        from asreviewcontrib.pro.__init__ import __version__
        from asreviewcontrib.pro.__init__ import __extension_name__

        self.version = __version__
        self.extension_name = __extension_name__

    def execute(self, argv):
        state_path = argv[0]
        data_path = argv[1]
        optimization_fp = argv[2]

        as_data = ASReviewData.from_file(data_path)
        model = get_model("nb")
        feature_model = get_feature_model("tfidf")

        X = feature_model.fit_transform(
            as_data.texts, as_data.headings, as_data.bodies, as_data.keywords
        )

        labels = as_data.labels
        all_n = []
        all_inclusions = []
        all_p_found = []
        with open_state(state_path) as state:
            n_queries = state.n_queries()
            for query_i in range(n_queries):
                try:
                    train_idx = state.get("train_idx", query_i=query_i)
                    pool_idx = state.get("pool_idx", query_i=query_i)
                    proba = state.get("proba", query_i=query_i)
                except KeyError:
                    continue
                n_inc, p_all = estimate_inclusions(
                    train_idx, pool_idx, X, labels,
                    optimization_fp)
                print(n_inc, np.sum(labels[train_idx]), np.sum(labels), p_all)
                all_n.append(n_inc)
                all_p_found.append(p_all)
                all_inclusions.append(np.sum(labels[train_idx]))
        plt.plot(all_n)
        plt.plot(all_inclusions)
        plt.show()

        plt.plot(all_p_found)
        plt.plot(np.array(all_inclusions)/np.sum(labels))
        plt.show()