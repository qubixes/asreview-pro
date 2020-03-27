import logging

from asreview.entry_points.base import BaseEntryPoint
from asreview.entry_points.simulate import _base_parser

from asreviewcontrib.pro.config import ASCII_MSG_ORACLE
from asreviewcontrib.pro.review import review_oracle


class OracleEntryPoint(BaseEntryPoint):
    description = "Perform a systematic review on the command line."

    def __init__(self):
        super(OracleEntryPoint, self).__init__()
        from asreviewcontrib.pro.__init__ import __version__
        from asreviewcontrib.pro.__init__ import __extension_name__

        self.version = __version__
        self.extension_name = __extension_name__

    def execute(self, argv):
        parser = _base_parser()
        parser.add_argument(
            "--new",
            default=False,
            action="store_true",
            help="Start review from scratch."
        )
        args = parser.parse_args(argv)

        args_dict = vars(args)
        path = args_dict.pop("dataset")

        verbose = args_dict.get("verbose", 0)
        if verbose == 0:
            logging.getLogger().setLevel(logging.WARNING)
        elif verbose == 1:
            logging.getLogger().setLevel(logging.INFO)
        elif verbose >= 2:
            logging.getLogger().setLevel(logging.DEBUG)

        print(ASCII_MSG_ORACLE)
        review_oracle(path, **args_dict)
