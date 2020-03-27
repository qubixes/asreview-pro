from asreview.config import GITHUB_PAGE, EMAIL_ADDRESS


ASCII_MSG_ORACLE = """
---------------------------------------------------------------------------------
|                                                                                |
|  Welcome to the ASReview Automated Systematic Review PRO edition software.     |
|  In this mode the computer will assist you in creating your systematic review. |
|  After giving it a few papers that are either included or excluded,            |
|  it will compute a model and show progressively more relevant papers.          |
|                                                                                |
|  GitHub page:        {0: <58}|
|  Questions/remarks:  {1: <58}|
|                                                                                |
---------------------------------------------------------------------------------
""".format(GITHUB_PAGE, EMAIL_ADDRESS)

DESCRIPTION_ORACLE = """
Automated Systematic Review (ASReview) PRO edition.

The oracle modus is used to perform a systematic review with
interaction by the reviewer (the ‘oracle’ in literature on active
learning). The software presents papers to the reviewer, whereafter
the reviewer classifies them."""
