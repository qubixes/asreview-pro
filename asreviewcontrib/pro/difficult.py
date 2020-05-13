from argparse import ArgumentParser

from asreview import ASReviewData
from asreview.analysis import Analysis
from asreview.entry_points.base import BaseEntryPoint


class DifficultEntryPoint(BaseEntryPoint):
    description = "Show the most difficult records."

    def __init__(self):
        super(DifficultEntryPoint, self).__init__()
        from asreviewcontrib.pro.__init__ import __version__
        from asreviewcontrib.pro.__init__ import __extension_name__

        self.version = __version__
        self.extension_name = __extension_name__

    def execute(self, argv):
        parser = _parse_arguments()
        arg_dict = vars(parser.parse_args(argv))

        state_path = arg_dict["state_path"]
        data_path = arg_dict["data_path"]
        top_n = arg_dict["top"]
        as_data = ASReviewData.from_file(data_path)

        order, ttd = self.find_order(state_path)
        for key in order[:top_n]:
            print(f"{ttd[key]:.2f} %")
            as_data.print_record(key)

    def find_order(self, state_path):
        analysis = Analysis.from_path(state_path)
        ttd = analysis.avg_time_to_discovery(result_format="percentage")
        analysis.close()
        order = sorted(ttd, key=lambda x: -ttd[x])
        return order, ttd


def _parse_arguments():
    parser = ArgumentParser(prog="asreview difficult")

    parser.add_argument(
        'state_path',
        type=str,
        help="Path to state/log file to analyze."
    )
    parser.add_argument(
        'data_path',
        type=str,
        help="Path to data file corresponding to the state file."
    )
    parser.add_argument(
        "-n", "--top",
        type=int,
        default=3,
        help="Determines how many entries are shown."
    )

    return parser
