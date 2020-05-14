import logging

import numpy as np

from asreview.init_sampling import sample_prior_knowledge
from asreview.review import BaseReview
from fractions import Fraction


class ReviewSubSimulate(BaseReview):
    name = "sub_simulate"

    def __init__(
            self, as_data, *args, n_prior_included=0, n_prior_excluded=0,
            prior_idx=None, seed=None, fraction_included=None,
            n_paper_sim=None,
            **kwargs):
        if as_data.labels is None:
            raise ValueError("Cannot simulate without labels!")
        r = np.random.RandomState(seed)

        # Remove unlabeled papers.
        labels = as_data.labels
        labeled_idx = np.where((labels == 0) | (labels == 1))[0]
        if len(labeled_idx) != len(labels):
            logging.warning("Simulating partial review, ignoring unlabeled"
                            f" papers (n={len(labels)-len(labeled_idx)}.")
            as_data = as_data.slice(labeled_idx)
            labels = as_data.labels

        zero_idx = np.where(labels == 0)[0]
        one_idx = np.where(labels == 1)[0]
        n_papers = len(as_data)

        if fraction_included is None:
            fraction_included = one_idx/n_papers

        n_zero_left = len(zero_idx) - n_prior_excluded
        n_one_left = len(one_idx) - n_prior_included
        if n_paper_sim is None or n_zero_left + n_one_left < n_paper_sim:
            if n_one_left/fraction_included > n_zero_left/(1-fraction_included):
                n_zero_sim = max(1, n_zero_left)
                n_one_sim = fraction_included/(1-fraction_included)*n_zero_left
                n_one_sim = max(1, n_one_sim)
                if n_one_sim + n_prior_included < 2:
                    n_one_sim = 2
            else:
                n_one_sim = max(1, n_one_left)
                n_zero_sim = (1-fraction_included)/fraction_included*n_one_left
                n_zero_sim = max(1, n_zero_sim)
                if n_zero_sim + n_prior_excluded < 2:
                    n_zero_sim = 2
        else:
            n_one_desired = round(fraction_included*n_paper_sim)
            n_zero_desired = n_paper_sim - n_one_desired
            if n_one_desired <= n_one_left and n_zero_desired <= n_zero_left:
                n_zero_sim = n_zero_desired
                n_one_sim = n_one_desired
            elif n_one_desired < n_one_left:
                n_one_sim = n_one_desired
                n_zero_sim = max(1, (1-fraction_included)/fraction_included*n_one_desired)
            else:
                n_zero_sim = n_zero_desired
                n_one_sim = max(1, fraction_included/(1-fraction_included)*n_zero_desired)
            if n_zero_sim + n_prior_excluded < 2:
                n_zero_sim = 2
            if n_one_sim + n_prior_included < 2:
                n_one_sim = 2

        if (n_one_sim > n_one_left or n_one_sim < 2 or
                n_zero_sim > n_zero_left or n_zero_sim < 2):
            raise ValueError("Cannot find valid configuration."
                             f" {n_one_sim} {n_one_left}, {n_zero_sim}"
                             f" {n_zero_left}, {fraction_included}.")

        slice_zero_idx = r.choice(zero_idx, n_zero_sim+n_prior_excluded, replace=False)
        slice_one_idx = r.choice(one_idx, n_one_sim+n_prior_included, replace=False)

        as_data = as_data.slice(np.concatenate((one_idx, zero_idx)))

        if n_prior_included + n_prior_excluded > 0:
            start_idx = sample_prior_knowledge(
                labels, n_prior_included, n_prior_excluded)
        else:
            start_idx = np.array([])

        super(ReviewSubSimulate, self).__init__(
            as_data, *args, start_idx=start_idx, **kwargs)

    def _get_labels(self, ind):
        """Get the labels directly from memory.

        Arguments
        ---------
        ind: list, np.array
            A list with indices

        Returns
        -------
        list, np.array
            The corresponding true labels for each indice.
        """

        return self.y[ind, ]
