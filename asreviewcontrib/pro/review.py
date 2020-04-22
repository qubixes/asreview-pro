from os.path import splitext
import os

import questionary

from asreview.config import LOGGER_EXTENSIONS
from asreview.datasets import DatasetManager, BaseVersionedDataSet

from asreviewcontrib.pro.factory import review


def get_num_datasets(dataset, **kwargs):
    num = len(dataset)
    num += len(kwargs["included_dataset"])
    num += len(kwargs["prior_dataset"])
    num += len(kwargs["excluded_dataset"])
    return num


def review_oracle(dataset, *args, state_file=None, **kwargs):
    """CLI to the interactive mode."""

    if get_num_datasets(dataset, **kwargs) == 0:
        dataset = [dataset_menu()]
        if dataset[0] is None:
            return
    if state_file is None:
        try:
            state_file = state_file_menu()
        except KeyboardInterrupt:
            return
    try:
        review(dataset, *args, mode='oracle', state_file=state_file, **kwargs)
    except KeyboardInterrupt:
        print('\nClosing down the automated systematic review.')


def state_file_menu():
    while True:
        state_file = questionary.text(
            'Please provide a file to store '
            'the results of your review:',
            validate=lambda val: splitext(val)[1] in LOGGER_EXTENSIONS,
        ).ask()
        if state_file is None:
            raise KeyboardInterrupt()
        if len(state_file) == 0:
            force_continue = questionary.confirm(
                'Are you sure you want to continue without saving?',
                default=False
            ).ask()
            if force_continue:
                return None
        else:
            if not os.path.isfile(state_file):
                return state_file
            action = questionary.select(
                f'File {state_file} exists, what do you want'
                ' to do?',
                default='Exit',
                choices=[
                    f'Continue review from {state_file}',
                    f'Delete review in {state_file} and start a new'
                    ' review',
                    f'Choose another file name.',
                    questionary.Separator(),
                    f'Exit'
                ]
            ).ask()
            if action == "Exit" or action is None:
                raise KeyboardInterrupt()
            if action.startswith("Continue"):
                return state_file
            if action.startswith("Choose another"):
                continue
            if action.startswith("Delete"):
                delete = questionary.confirm(
                    f'Are you sure you want to delete '
                    f'{state_file}?',
                    default=False,
                ).ask()
                if delete:
                    os.remove(state_file)
                    return state_file
                else:
                    continue


def dataset_menu():
    dataset = None
    while dataset is None:
        action = questionary.select(
            "You have not supplied a dataset, what do you want to do?",
            choices=[
                "Select installed dataset",
                "Give exact filename",
                questionary.Separator(),
                "Exit",
            ]
        ).ask()
        if action == "Exit":
            return None
        if action.startswith("Give"):
            dataset = get_dataset_path()
        if action.startswith("Select"):
            dataset = get_internal_dataset()
    return dataset


def get_dataset_path():
    return questionary.text(
        "Please provide the dataset path:",
    ).ask()


def get_internal_dataset():
    dm = DatasetManager()
    data_list = dm.list(latest_only=False)
    flat_data_list = []
    for group_id, group_list in data_list.items():
        for dataset in group_list:
            data_id = f"{group_id}:{dataset.dataset_id}"
            title = dataset.title
            if isinstance(dataset, BaseVersionedDataSet):
                title += f" ({len(dataset)} versions)"
            flat_data_list.append((data_id, title, dataset))

    choice_dict = {x[1]: (x[0], x[2]) for x in flat_data_list}

    return_id = None
    while return_id is None:
        title = questionary.select(
            "Select dataset:",
            choices=list(choice_dict) + [questionary.Separator(), "Back"],
        ).ask()
        if title == "Back":
            return None
        dataset_id, dataset = choice_dict[title]
        if isinstance(dataset, BaseVersionedDataSet):
            return_id = select_versioned_dataset(dataset_id, dm)
        else:
            return_id = select_dataset(dataset_id, dm)

    return return_id


def select_versioned_dataset(dataset_id, dm):
    group_id = dataset_id.split(":")[0]
    dataset = dm.find(dataset_id)
    titles = [d.title for d in reversed(dataset.datasets)]
    data_ids = [d.dataset_id for d in reversed(dataset.datasets)]
    return_id = None
    while return_id is None:
        title = questionary.select(
            "Select version:",
            choices=titles + [questionary.Separator(), "Back"],
        ).ask()
        if title == "Back":
            return None
        vers_data_id = group_id + ":" + data_ids[titles.index(title)]
        return_id = select_dataset(vers_data_id, dm)
    return return_id


def select_dataset(dataset_id, dm):
    dataset = dm.find(dataset_id)
    print(dataset)
    right_dataset = questionary.confirm(
        "Is this the correct dataset?",
        default=True,
    ).ask()
    if right_dataset:
        return dataset_id
    else:
        return None
