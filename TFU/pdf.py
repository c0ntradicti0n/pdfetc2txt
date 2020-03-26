import logging
from typing import Callable


class Pdf:
    def __init__(self):
        pass

    def verify(self, serious=False, test_document=False):
        attributes = {attr: getattr(self,attr)
                      for attr in dir(self)
                      if not isinstance(getattr(self,attr), Callable)
                      and not attr.startswith("__")}
        for attr, value in attributes.items():
            if not value:
                logging.error(f"{attr} is not set")
                if serious:
                    assert value

        if test_document:
            score = 0
            for page, cols in self.pages_to_column_to_text.items():
                if len(cols) == self.columns:
                    score += 100
                for col_number, col_text in cols.items():
                    if "text" in col_text and "column" in col_text:
                        score += 1
                    if str(col_number) in col_text:
                        score += 1
                    if not str(col_number + 1) in col_text:
                        score += 1
                    if not str(col_number - 1) in col_text:
                        score += 1
            return score






    #title = ""
    columns = 0
    #columns_per_page = []
    text = ""
    indexed_words = {}
    #footnotes = []
    #bibliography = []