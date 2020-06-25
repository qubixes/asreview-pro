import logging


from asreview.config import DEFAULT_BALANCE_STRATEGY
from asreview.config import DEFAULT_FEATURE_EXTRACTION
from asreview.config import DEFAULT_MODEL
from asreview.config import DEFAULT_N_INSTANCES
from asreview.config import DEFAULT_N_PRIOR_EXCLUDED
from asreview.config import DEFAULT_N_PRIOR_INCLUDED
from asreview.config import DEFAULT_QUERY_STRATEGY
from asreview.state.utils import open_state
from asreview.settings import ASReviewSettings
from asreview.models.utils import get_model
from asreview.query_strategies.utils import get_query_model
from asreview.balance_strategies.utils import get_balance_model
from asreview.feature_extraction.utils import get_feature_model
from asreview.review.factory import create_as_data
from asreview.utils import get_random_state

from asreviewcontrib.pro.oracle_mode import ReviewOracle


def get_reviewer(dataset,
                 mode="simulate",
                 model=DEFAULT_MODEL,
                 query_strategy=DEFAULT_QUERY_STRATEGY,
                 balance_strategy=DEFAULT_BALANCE_STRATEGY,
                 feature_extraction=DEFAULT_FEATURE_EXTRACTION,
                 n_instances=DEFAULT_N_INSTANCES,
                 n_papers=None,
                 n_queries=None,
                 embedding_fp=None,
                 prior_idx=None,
                 n_prior_included=DEFAULT_N_PRIOR_INCLUDED,
                 n_prior_excluded=DEFAULT_N_PRIOR_EXCLUDED,
                 config_file=None,
                 state_file=None,
                 model_param=None,
                 query_param=None,
                 balance_param=None,
                 feature_param=None,
                 seed=None,
                 abstract_only=False,
                 included_dataset=[],
                 excluded_dataset=[],
                 prior_dataset=[],
                 new=False,
                 **kwargs
                 ):
    """Get a review object from arguments.

    See __main__.py for a description of the arguments.
    """
    as_data = create_as_data(dataset, included_dataset, excluded_dataset,
                             prior_dataset, new=new)

    if len(as_data) == 0:
        raise ValueError("Supply at least one dataset"
                         " with at least one record.")

    cli_settings = ASReviewSettings(
        model=model, n_instances=n_instances, n_queries=n_queries,
        n_papers=n_papers, n_prior_included=n_prior_included,
        n_prior_excluded=n_prior_excluded, query_strategy=query_strategy,
        balance_strategy=balance_strategy,
        feature_extraction=feature_extraction,
        mode=mode, data_fp=None,
        abstract_only=abstract_only)
    cli_settings.from_file(config_file)

    if state_file is not None:
        with open_state(state_file) as state:
            if state.is_empty():
                state.settings = cli_settings
            settings = state.settings
    else:
        settings = cli_settings

    if n_queries is not None:
        settings.n_queries = n_queries
    if n_papers is not None:
        settings.n_papers = n_papers

    if model_param is not None:
        settings.model_param = model_param
    if query_param is not None:
        settings.query_param = query_param
    if balance_param is not None:
        settings.balance_param = balance_param
    if feature_param is not None:
        settings.feature_param = feature_param

    logging.debug(settings)

    random_state = get_random_state(seed)
    train_model = get_model(settings.model, **settings.model_param,
                            random_state=random_state)
    query_model = get_query_model(settings.query_strategy,
                                  random_state=random_state,
                                  **settings.query_param)
    balance_model = get_balance_model(settings.balance_strategy,
                                      random_state=random_state,
                                      **settings.balance_param)
    feature_model = get_feature_model(settings.feature_extraction,
                                      random_state=random_state,
                                      **settings.feature_param)

    if train_model.name.startswith("lstm-"):
        texts = as_data.texts
        train_model.embedding_matrix = feature_model.get_embedding_matrix(
            texts, embedding_fp)

    # Initialize the review class.
    reviewer = ReviewOracle(
        as_data,
        model=train_model,
        query_model=query_model,
        balance_model=balance_model,
        feature_model=feature_model,
        n_papers=settings.n_papers,
        n_instances=settings.n_instances,
        n_queries=settings.n_queries,
        state_file=state_file,
        **kwargs)

    return reviewer


def review(*args, mode="simulate", model=DEFAULT_MODEL, **kwargs):
    """Perform a review from arguments. Compatible with the CLI interface"""
    reviewer = get_reviewer(*args, mode=mode, model=model, **kwargs)

    # Start the review process.
    reviewer.review()
