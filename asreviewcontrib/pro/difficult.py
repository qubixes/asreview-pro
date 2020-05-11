
from collections import OrderedDict

from asreview import ASReviewData
from asreview.analysis import Analysis
from asreview.entry_points.base import BaseEntryPoint


class DifficultEntryPoint(BaseEntryPoint):
    description = "XXX."

    def __init__(self):
        super(DifficultEntryPoint, self).__init__()
        from asreviewcontrib.pro.__init__ import __version__
        from asreviewcontrib.pro.__init__ import __extension_name__

        self.version = __version__
        self.extension_name = __extension_name__

    def execute(self, argv):
        state_path = argv[0]
        data_path = argv[1]
        try:
            top_n = argv[2]
        except IndexError:
            top_n = 3

        as_data = ASReviewData.from_file(data_path)
        analysis = Analysis.from_path(state_path)
        ttd = analysis.avg_time_to_discovery(result_format="percentage")
        order = sorted(ttd, key=lambda x: -ttd[x])
#         ttd = OrderedDict((key, ttd[key]) for key in order)
        for key in order[:top_n]:
            print(f"{ttd[key]:.2f} %")
            as_data.print_record(key)
#             print(key, ttd[key])
#         print(ttd)
#         print(sorted(ttd, key=lambda x: -ttd[x]))
        analysis.close()