from os.path import splitext
import os

import questionary

from asreview.config import LOGGER_EXTENSIONS

from asreviewcontrib.pro.factory import review


def review_oracle(dataset, *args, state_file=None, **kwargs):
    """CLI to the interactive mode."""
    if state_file is None:
        while True:
            state_file = questionary.text(
                'Please provide a file to store '
                'the results of your review:',
                validate=lambda val: splitext(val)[1] in LOGGER_EXTENSIONS,
            ).ask()
            if state_file is None:
                return
            if len(state_file) == 0:
                force_continue = questionary.confirm(
                    'Are you sure you want to continue without saving?',
                    default=False
                ).ask()
                if force_continue:
                    state_file = None
                    break
            else:
                if os.path.isfile(state_file):
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
                        return
                    if action.startswith("Continue"):
                        break
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
                            break
                        else:
                            continue

                break
    try:
        review(dataset, *args, mode='oracle', state_file=state_file, **kwargs)
    except KeyboardInterrupt:
        print('\nClosing down the automated systematic review.')
