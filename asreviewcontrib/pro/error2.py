
from collections import OrderedDict
from math import floor

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


def discrete_norm_dist(mu, sigma, train_percentage, bins):
    norm_cdf = stats.norm.cdf(bins, loc=mu, scale=sigma)
    norm_pdf = train_percentage*(norm_cdf[1:]-norm_cdf[:-1])
    norm_hist = norm_pdf/norm_pdf.sum()
    return norm_hist/(bins[1]-bins[0])


def percentage_found(mu, sigma, train_percentage, bins):
    norm_cdf = stats.norm.cdf(bins, loc=mu, scale=sigma)
    norm_dpdf = (norm_cdf[1:]-norm_cdf[:-1])
    norm_dpdf /= norm_dpdf.sum()
    return np.sum(train_percentage*norm_dpdf)


def kl_divergence(train_dist, norm_dist):
    klb = 0
    for i in range(len(norm_dist)):
        if norm_dist[i] == 0 or train_dist[i] == 0:
            continue
        klb += train_dist[i]*np.log(train_dist[i]/norm_dist[i])
    return klb



# def log_likelihood(values, distribution):
#     
#     for i in range(len(bins)-1):
#         

def estimate_inclusions(train_idx, pool_idx, X, y, proba, plot=False):
#     temp_labels = np.zeros(y.shape)
#     temp_labels[train_idx] = y[train_idx]
#     model.fit(X[train_idx], y[train_idx])
#     print(model._model.decision_function(X))
#     proba = model.predict_proba(X)[:, 1]

    model = get_model("nb")
    balance_model = get_balance_model("double")
    X_train, y_train = balance_model.sample(X, y, train_idx, {})

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
        mu = x[0]
        sigma = x[1]
        corrected_dist = discrete_norm_dist(mu, sigma, perc_train, bin_edges)
        return kl_divergence(hist, corrected_dist)

    mu_range = h_range
    sigma_range = (2*(h_max-h_min)/n_bins, h_max-h_min)
    x0 = np.array((np.average(df_one_corrected), np.var(df_one_corrected)))

    mu_best, sigma_best = minimize(fun=guess_func, x0=x0,
                                   bounds=[mu_range, sigma_range]).x
    corrected_dist = discrete_norm_dist(mu_best, sigma_best, perc_train, bin_edges)


#     for mu in np.linspace(2.0, 3.0, 50):
#         corrected_dist = discrete_norm_dist(mu, 0.4, perc_train, bin_edges)
#         klb = kl_divergence(hist, corrected_dist)
#         print(mu, klb)


#     print(hist, bin_edges)
#     print(correct_one_proba)
#     print(np.average(correct_one_proba))

#     print(discrete_norm_dist(3, 1, bin_edges).sum())

    perc_found = percentage_found(mu_best, sigma_best, perc_train, bin_edges)
    if plot:
        plt.plot((bin_edges[1:]+bin_edges[:-1])/2, corrected_dist)
        plt.plot((bin_edges[1:]+bin_edges[:-1])/2, hist)
        plt.plot((bin_edges[1:]+bin_edges[:-1])/2, perc_train)
        plt.show()
    return np.sum(y[train_idx])/perc_found
#     zero_idx = np.where(y == 0)[0]
#     one_idx = np.where(y == 1)[0]
# #     df_values = proba
# #     model.fit(X[train_idx], y[train_idx])
# #     df_values = model._model.decision_function(X).reshape((-1, 1))
#     df_values = -np.log(1/proba-1)
# #     df_values = model.predict_proba(X)[:, 1]
# #     print(model._model.classes_)
# #     print(model.predict_proba(X[one_idx]))
#     df_train_zero = df_values[train_idx[np.where(y[train_idx] == 0)[0]]]
#     print([np.average(df_train_zero), np.average(df_values[one_idx]),
#            np.average(df_values[zero_idx]), np.average(df_one_corrected)])
#     plt.hist([df_train_zero, df_values[one_idx], df_values[zero_idx], df_one_corrected], 30, histtype='bar', density=True)
# #     plt.hist(df_values[one_idx])
# #     plt.hist(df_values[zero_idx])
#     plt.show()
#     print(np.sum(y[pool_idx]))
#     return
#     
#     C = np.sum(y[train_idx])/(len(y)-np.sum(y[train_idx]))
#     log_model = LogisticRegression(penalty="l2", fit_intercept=True, C=C)
# #     print(df_values.reshape((-1, 1)))
# 
#     temp_ones = np.array([], dtype=int)
#     n_extra_ones = 0
#     for _ in range(500):
#         temp_labels = np.zeros(y.shape)
#         temp_labels[train_idx] = y[train_idx]
#         temp_labels[temp_ones] = 1
#         log_model.fit(df_values, temp_labels)
#         log_proba = log_model.predict_proba(df_values)[:, 1]
#         rel_sample_idx, cum_proba = sample_proba(log_proba[pool_idx])
#         temp_ones = pool_idx[rel_sample_idx]
#         if len(temp_ones) == n_extra_ones:
#             break
#         n_extra_ones = len(temp_ones)
#     print(len(pool_idx), cum_proba)
#     return len(temp_ones)
#     print(len(temp_ones), cum_proba)
#     print("--------------------")
#     print(-np.log(1/proba-1))
#     for _ in range(5):
#         new_temp_labels = np.zeros(y.shape)
#         new_temp_labels[train_idx] = y[train_idx]
# #         pool_order = pool_idx[np.argsort(-proba[pool_idx])]
#         n_choice = round(np.sum(proba) - np.sum(y[train_idx]))
# #         n_choice = round(np.sum(proba[pool_idx]))
#         p = proba[pool_idx]/np.sum(proba[pool_idx])
#         temp_ones = np.random.choice(pool_idx, int(n_choice), p=p,
#                                      replace=False)
#         new_temp_labels[temp_ones] = 1
#         model.fit(X, new_temp_labels)
#         proba = model.predict_proba(X)[:, 1]
#         print(np.sum(new_temp_labels), np.sum(proba))
#         print(temp_ones)
#         print(proba[pool_order])
#     print(-np.sort(-proba))
#     print(np.sum(proba))
#     print(np.sum(y), np.sum(y[train_idx]))
#     return [np.sum(y[pool_idx]), np.sum(proba[pool_idx])]


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

        as_data = ASReviewData.from_file(data_path)
        model = get_model("nb")
        feature_model = get_feature_model("tfidf")

        X = feature_model.fit_transform(
            as_data.texts, as_data.headings, as_data.bodies, as_data.keywords
        )

        labels = as_data.labels
        all_n = []
        all_inclusions = []
        with open_state(state_path) as state:
            n_queries = state.n_queries()
            for query_i in range(n_queries):
#             for query_i in [n_queries-1]:
                try:
                    train_idx = state.get("train_idx", query_i=query_i)
                    pool_idx = state.get("pool_idx", query_i=query_i)
                    proba = state.get("proba", query_i=query_i)
                except KeyError:
                    continue
                n_inc = estimate_inclusions(train_idx, pool_idx, X, labels, proba,
                                            plot=(query_i == n_queries-1))
                print(n_inc, np.sum(labels[train_idx]), np.sum(labels))
                all_n.append(n_inc)
                all_inclusions.append(np.sum(labels[train_idx]))
        plt.plot(all_n)
        plt.plot(all_inclusions)
        plt.show()
