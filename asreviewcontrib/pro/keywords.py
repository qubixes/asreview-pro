from argparse import ArgumentParser

from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize.minpack import curve_fit
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from asreview import ASReviewData
from asreview.feature_extraction import get_feature_model
from asreview.balance_strategies import get_balance_model
from asreview.models import get_model
from asreview.entry_points.base import BaseEntryPoint


class KeywordEntryPoint(BaseEntryPoint):
    def __init__(self):
        super(KeywordEntryPoint, self).__init__()
        from asreviewcontrib.pro.__init__ import __version__
        from asreviewcontrib.pro.__init__ import __extension_name__

        self.version = __version__
        self.extension_name = __extension_name__

    def execute(self, argv):
        parser = _parse_arguments()
        arg_dict = vars(parser.parse_args(argv))

        dataset = arg_dict["dataset"]
        growth = arg_dict["growth"]
        correlation = arg_dict["correlation"]
        n_sample = arg_dict["n_sample"]

        name = Path(dataset).stem
        as_data = ASReviewData.from_file(dataset)
        self.effect_words(as_data, compute_growth=growth, name=name,
                          correlation=correlation, n_sample=n_sample)

    def effect_words(self, as_data, compute_growth=False, correlation=None,
                     name="unknown", n_sample=None):
#         feature_model = get_feature_model("tfidf", split_ta=0, use_keywords=0)
        feature_model = TfidfVectorizer()
        balance_model = get_balance_model("double")
        model = get_model("nb")

        texts = as_data.texts
        labels = as_data.labels
        one_idx = np.where(labels == 1)[0]
        zero_idx = np.where(labels == 0)[0]
        X = feature_model.fit_transform(texts)
        y = labels
        train_idx = np.arange(len(y))
        X_train, y_train = balance_model.sample(X, y, train_idx, {})
        model.fit(X_train, y_train)
        tokenizer = feature_model.build_analyzer()
        count_model = CountVectorizer()
        counts = count_model.fit_transform(texts)

        if correlation is not None:
            cor_zero = compute_effect_correlation(
                correlation[0], correlation[1], one_idx, texts,
                tokenizer, X, model, feature_model)
            cor_one = compute_effect_correlation(
                correlation[0], correlation[1], zero_idx, texts,
                tokenizer, X, model, feature_model)
            print(f"{correlation[0]} vs {correlation[1]}: {cor_zero}/{cor_one}"
                  )
        elif compute_growth:
            compute_effect_growth(one_idx, zero_idx, texts, tokenizer, X,
                                  model, feature_model, counts, count_model,
                                  name=name, n_sample=n_sample)
        else:
            print_effect_words(one_idx, zero_idx, counts, count_model, X,
                               model, feature_model, n_sample=n_sample)


def compute_effect_growth(one_idx, zero_idx, texts, tokenizer, X, model,
                          feature_model, counts, count_model, name="unknown",
                          n_sample=200):
    all_alpha_pos = []
    all_alpha_neg = []
    all_percentage = [50, 66, 80, 90, 95]
    for perc in all_percentage:
        alpha_pos = effect_words_fast(
            one_idx, texts, tokenizer, X, model, feature_model,
            counts, count_model, percentile=perc, name=name)
        alpha_neg = effect_words_fast(
            zero_idx, texts, tokenizer, X, model, feature_model,
            counts, count_model, n_sample=n_sample, percentile=perc,
            positive_effect=False, name=name)
        all_alpha_pos.append(alpha_pos)
        all_alpha_neg.append(alpha_neg)

    plt.scatter(all_percentage, all_alpha_pos, label="inclusions")
    plt.scatter(all_percentage, all_alpha_neg, label="exclusions")
    plt.legend()
    plt.title(name)
    plt.show()
    return


def print_effect_words(one_idx, zero_idx, counts, count_model, X, model,
                       feature_model, n_sample=1000):
#     one_effect_words = compute_effect_words(
#         one_idx, texts, tokenizer, X, model, feature_model)
#     zero_effect_words = compute_effect_words(
#         zero_idx, texts, tokenizer, X, model, feature_model, n_sample=300)

#     one_tokens = list(one_effect_words)
#     one_count = [effect[0] for effect in one_effect_words.values()]
#     one_effect = [effect[1] for effect in one_effect_words.values()]
#     one_sorted_idx = np.argsort(one_effect)

#     zero_tokens = list(zero_effect_words)
#     zero_count = [effect[0] for effect in zero_effect_words.values()]
#     zero_effect = [effect[1] for effect in zero_effect_words.values()]
#     zero_sorted_idx = np.argsort(zero_effect)

    one_effect_results = effect_words_fast(
        one_idx, X, model, feature_model, counts, count_model)
    one_alpha = effect_alpha_fast(
        one_idx, X, model, feature_model, counts, count_model,
        positive_effect=True, percentile=80, plot=False,
        effect_results=one_effect_results)
    one_tokens, one_effects, one_counts = convert_results(
        one_effect_results, one_alpha, sort_high_low=True)

    zero_effect_results = effect_words_fast(
        zero_idx, X, model, feature_model, counts, count_model, n_sample=n_sample)
    zero_alpha = effect_alpha_fast(
        zero_idx, X, model, feature_model, counts, count_model,
        positive_effect=False, percentile=80, plot=False,
        effect_results=zero_effect_results)
    zero_tokens, zero_effects, zero_counts = convert_results(
        zero_effect_results, zero_alpha, sort_high_low=False)

    for idx in range(20):
        print(one_tokens[idx], one_effects[idx]/len(one_idx), one_counts[idx])
    print("-------------------------------------------")
    for idx in range(20):
        print(zero_tokens[idx], zero_effects[idx]/min(len(zero_idx), n_sample),
              zero_counts[idx])


def compute_effect_words(idx_list, texts, tokenizer, X, model, feature_model,
                         n_sample=None):
    if n_sample is not None and n_sample < len(idx_list):
        idx_list = np.random.choice(idx_list, n_sample, replace=False)

    effect_words = {}

    for inclusion in idx_list:
        abstract = texts[inclusion]

        tokens = np.array(tokenizer(abstract))
        token_set = set()
        df_old = _compute_df(model, X[inclusion])[0]

        token_list = []
        gain_list = []
        for token in tokens:
            if token in token_set:
                continue
            token_idx = np.where(tokens == token)[0]
            token_set.add(token)
            new_tokens = np.delete(tokens, token_idx)
            new_abstract = " ".join(new_tokens)
            new_X = feature_model.transform([new_abstract])
            df_new = _compute_df(model, new_X)[0]
#             print(df_new, df_old)
            token_list.append(token)
            gain_list.append((df_old-df_new)/len(token_idx))
            if token not in effect_words:
                effect_words[token] = [0, 0.0]
            effect_words[token][0] += 1
            effect_words[token][1] += (df_old-df_new)/len(token_idx)
    return effect_words


def effect_words_fast(idx_list, X, model, feature_model,
                      counts, count_model,
                      n_sample=None):
    if n_sample is not None and n_sample < len(idx_list):
        idx_list = np.random.choice(idx_list, n_sample, replace=False)

    X_sub = X[idx_list].tocsc()
    count_sub = counts[idx_list].tocsc()
    feature_names = feature_model.get_feature_names()
    count_names = count_model.get_feature_names()

    assert np.all(feature_names == count_names)

    df_old = _compute_df(model, X_sub)

    results = {}
    for i, token in enumerate(feature_names):
        token_idx = X_sub[:, i].indices
        if not len(token_idx):
            continue

        token_count = count_sub[token_idx, i].toarray().reshape(-1)
        X_new = X_sub[token_idx].copy()
        X_new[:, i] = 0
        df_new = _compute_df(model, X_new)
#         proba_new = model.predict_proba(X_new)[:, 1]
#         df_new = -np.log(1/proba_new - 1)
        results[token] = [df_old[token_idx]-df_new, token_count]

    return results


def convert_results(results, alpha=1.0, sort_high_low=True):
    tokens = []
    effects = []
    counts = []

    for token, res in results.items():
        df_vals = res[0]
        count = res[1]
#         print(token, df_vals, count, alpha)
        new_effect = np.sum(df_vals*count**-alpha)
        if np.isnan(new_effect):
            print(count, df_vals)
        tokens.append(token)
        effects.append(new_effect)
        counts.append(len(count))

    tokens = np.array(tokens)
    effects = np.array(effects)
    counts = np.array(counts)
    if sort_high_low:
        order = np.argsort(-effects)
    else:
        order = np.argsort(effects)

    return tokens[order], effects[order], counts[order]


def effect_alpha_fast(idx_list, X, model, feature_model,
                      counts, count_model, positive_effect=True,
                      percentile=70, plot=False, name="unknown",
                      n_sample=None, effect_results=None):
    if n_sample is not None and n_sample < len(idx_list):
        idx_list = np.random.choice(idx_list, n_sample, replace=False)

    if effect_results is None:
        effect_words_fast(idx_list, X, model, feature_model, counts,
                          count_model, n_sample)

    ddf_one_vals = []
    for ddf, token_count in effect_results.values():
        one_ddf = np.where(token_count == 1)[0]
        ddf_one_vals.append(np.mean(ddf[one_ddf]))

    if plot:
        plt.hist(ddf_one_vals, bins=100)
        plt.show()

    if positive_effect:
        df_cutoff = np.nanpercentile(ddf_one_vals, percentile)
    else:
        df_cutoff = np.nanpercentile(ddf_one_vals, 100-percentile)

#     print(percentile, df_cutoff)

    x_vals = []
    y_vals = []
    for ddf, token_count in effect_results.values():
        u = np.unique(token_count)
        if u[0] != 1 or len(u) <= 1:
            continue
        temp_dict = {}
        for i, count in enumerate(token_count):
            if count not in temp_dict:
                temp_dict[count] = []
            temp_dict[count].append(ddf[i])

        mean_one = np.mean(temp_dict[1])
        if positive_effect and mean_one < df_cutoff:
            continue
        elif not positive_effect and mean_one > df_cutoff:
            continue

        for n_count, res_list in temp_dict.items():
            if n_count == 1:
                continue

            mean_other = np.mean(res_list)
            ratio = mean_other/mean_one
            if ratio < 1e-4:
                continue
            x_vals.append(n_count)
            y_vals.append(np.log(mean_other/mean_one))

    def log_func(x, alpha):
        return alpha*np.log(x)
    par_opt, _ = curve_fit(log_func, x_vals, y_vals)
    if plot:
        plt.scatter(x_vals, y_vals, label="data")
        plt.plot(np.arange(1, 15), par_opt*np.log(np.arange(1, 15)),
                 label=f"{par_opt[0]:.2f} log(x)")
        plt.plot(np.arange(1, 15), np.log(np.arange(1, 15)),
                 label=f"log(x)")
        plt.legend
        plt.title(name)
        plt.show()
    return par_opt[0]


def compute_effect_correlation(word_a, word_b, idx_list, texts, tokenizer, X,
                               model, feature_model):
    feature_names = feature_model.get_feature_names()

    index_a = feature_names.index(word_a)
    index_b = feature_names.index(word_b)
    col_a = X[:, index_a].toarray()
    col_b = X[:, index_b].toarray()
    both_idx = np.where((col_a != 0) & (col_b != 0))[0]
    if len(both_idx) == 0:
        return 1

    df_base = _compute_df(model, X[both_idx])

    X_a_not_b = X[both_idx].copy()
    X_a_not_b[:, index_a] = 0
    df_a_not_b = _compute_df(model, X_a_not_b)

    X_b_not_a = X[both_idx].copy()
    X_b_not_a[:, index_b] = 0
    df_b_not_a = _compute_df(model, X_b_not_a)

    X_not_a_b = X[both_idx].copy()
    X_not_a_b[:, index_a] = 0
    X_not_a_b[:, index_b] = 0
    df_not_a_b = _compute_df(model, X_not_a_b)

    mean_proba_base = np.mean(df_base)

    delta_a_not_b = np.mean(df_a_not_b) - mean_proba_base
    delta_b_not_a = np.mean(df_b_not_a) - mean_proba_base
    delta_not_a_b = np.mean(df_not_a_b) - mean_proba_base
    return (delta_a_not_b + delta_b_not_a)/delta_not_a_b


def _compute_df(model, X):
    proba = model.predict_proba(X)[:, 1]
    return -np.log(1/proba-1)


def _parse_arguments():
    parser = ArgumentParser(prog="asreview keyword")

    parser.add_argument(
        'dataset',
        type=str,
        help="Dataset to analyze."
    )
    parser.add_argument(
        '--n_sample',
        type=int,
        default=500,
        help="Number of samples for the exclusions."
    )
    parser.add_argument(
        "--growth",
        action="store_true",
        help="Compute the growth of the effect size."
    )
    parser.add_argument(
        "--correlation",
        nargs=2,
        default=None,
        help="Compute the effect correlation between two words."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="nb",
        help="Model to analyze. Same models as in the base ASReview system are"
        " available. Note that the feature extraction is set to TF-IDF. Some "
        "models work better/worse with TF-IDF. Some may require too much memory"
        "."
    )
    return parser
