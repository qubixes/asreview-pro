import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.sparsefuncs import inplace_column_scale

from asreview.feature_extraction.base import BaseFeatureExtraction


class BM25(BaseFeatureExtraction):
    """BM25 feature extraction."""
    name = "bm25"

    def __init__(self, *args, stop_words=0, ngram_max=1,
                 b=0.75, k1=1.5, min_df=1, **kwargs):
        super(BM25, self).__init__(*args, **kwargs)
        self.stop_words = int(stop_words)
        self.ngram_max = int(ngram_max)
        self.b = b
        self.k1 = k1
        self.min_df = int(min_df)
        if self.stop_words:
            stop_words = "english"
        else:
            stop_words = None

        self._model = CountVectorizer(
            stop_words=stop_words, ngram_range=(1, self.ngram_max),
            min_df=self.min_df)

    def fit(self, texts):
        self._model.fit(texts)

    def transform(self, texts):
        X = self._model.transform(texts)
        df = np.sum(X != 0, axis=0)
        idf = np.array(np.log((X.shape[1] - df + 0.5)/(df + 0.5))).reshape(-1)
        doc_len = np.array(np.sum(X, axis=1)).reshape(-1)
        avg_doc_len = np.average(doc_len)

        result = X.copy().astype(float)
        k1 = self.k1
        b = self.b
        for i in range(len(X.indptr)-1):
            denom = X.data[X.indptr[i]:X.indptr[i+1]] + k1 * (1-b+b*doc_len[i]/avg_doc_len)
            nom = X.data[X.indptr[i]:X.indptr[i+1]] * (k1+1)
            result.data[X.indptr[i]:X.indptr[i+1]] = nom/denom

        inplace_column_scale(result, idf)
        return result

    def full_hyper_space(self):
        from hyperopt import hp

        hyper_space, hyper_choices = super(BM25, self).full_hyper_space()
        hyper_space.update({
            "fex_ngram_max": hp.uniformint("fex_ngram_max", 1, 3),
            "fex_stop_words": hp.randint("fex_stop_words", 2),
            "fex_b": hp.uniform("fex_b", 0.1, 0.9),
            "fex_k1": hp.uniform("fex_k1", 0, 3),
            "fex_min_df": hp.quniform("fex_min_df", 0.5, 2.4999999, 1),
        })
        return hyper_space, hyper_choices
