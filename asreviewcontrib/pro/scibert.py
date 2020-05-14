#!/usr/bin/env python

from pathlib import Path
import numpy as np
from torch import tensor
from transformers import AutoTokenizer, AutoModel
from asreview.feature_extraction.base import BaseFeatureExtraction


class Scibert(BaseFeatureExtraction):
    """BERT feature extraction"""
    name = "scibert"

    def __init__(self, *args, **kwargs):
        super(Scibert, self).__init__(*args, **kwargs)
        self._model = AutoModel.from_pretrained(str(Path("scibert_scivocab_uncased")))
        self.tokenizer = AutoTokenizer.from_pretrained(str(Path("scibert_scivocab_uncased")))

    def fit(self, texts):
        pass

    def transform(self, texts):
        X = []
        for text in texts:
            input_ids = tensor([self.tokenizer.encode(text)[:512]])
            hidden_states, _ = self._model(input_ids)[-2:]
            X.append(np.average(hidden_states.detach().numpy(), axis=1))
        return np.array(X).reshape(-1, 768)
