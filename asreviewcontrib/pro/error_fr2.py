from math import floor

import numpy as np

from asreview import ASReviewData
from asreview.analysis import Analysis
from asreview.entry_points.base import BaseEntryPoint
from asreview.models import get_model
from asreview.feature_extraction import get_feature_model
from asreview.state import open_state
from sklearn.linear_model import LogisticRegression


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
    return sample_idx, cum_proba[-1]


def estimate_inclusions(train_idx, pool_idx, X, y, model):
    model.fit(X[train_idx], y[train_idx])
    df_values = model._model.decision_function(X).reshape((-1, 1))
    C = np.sum(y[train_idx])/(len(y)-np.sum(y[train_idx]))
    log_model = LogisticRegression(penalty="l2", fit_intercept=True, C=C)

    temp_ones = np.array([], dtype=int)
    n_extra_ones = 0
    for _ in range(500):
        temp_labels = np.zeros(y.shape)
        temp_labels[train_idx] = y[train_idx]
        temp_labels[temp_ones] = 1
        log_model.fit(df_values, temp_labels)
        log_proba = log_model.predict_proba(df_values)[:, 1]
        rel_sample_idx, cum_proba = sample_proba(log_proba[pool_idx])
        temp_ones = pool_idx[rel_sample_idx]
        if len(temp_ones) == n_extra_ones:
            break
        n_extra_ones = len(temp_ones)
    print(len(pool_idx), cum_proba)
    return len(temp_ones)


class ErrorEntryPoint(BaseEntryPoint):
    description = "FastRead2 error computation [attempt]."

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
        model = get_model("logistic")
        feature_model = get_feature_model("tfidf")

        X = feature_model.fit_transform(
            as_data.texts, as_data.headings, as_data.bodies, as_data.keywords
        )

        labels = as_data.labels
        all_n = []
        with open_state(state_path) as state:
            n_queries = state.n_queries()
            for query_i in range(n_queries):
                try:
                    train_idx = state.get("train_idx", query_i=query_i)
                    pool_idx = state.get("pool_idx", query_i=query_i)
                except KeyError:
                    continue
                n_inc = estimate_inclusions(train_idx, pool_idx, X, labels, model)
                print(n_inc, np.sum(labels[pool_idx]))
                all_n.append(n_inc)
        print(all_n)
