import os
from TFU.trueformatupmarker import TrueFormatUpmarker
from helpers.color_logger import *
import bs4
from bs4 import NavigableString

class TrueFormatUpmarkerHTML (TrueFormatUpmarker):
    def collect_all_divs(self, soup):
        self.text_divs = []
        self.cuts = []
        self.collect_all_divs_recursive(soup.body)
        return self.text_divs

    def collect_all_divs_recursive(self, soup):
        for tag in soup.find_all():
            if any(tag.name.startswith(possible_tag_beginning) for possible_tag_beginning in
                ["div", "p", "a", "span"]):
                if isinstance(tag, bs4.element.Tag) and all(isinstance(child, NavigableString) for child in tag.children):
                    self.text_divs.append(tag)
                    if self.is_cut(tag):
                       self.text_divs.append(self.CUT)

    def is_cut(self, tag):
        return tag.name == "p" or (tag.name == "p")


    def generate_css_tagging_document(self, html_before_path, html_after_path, debug_folder):
        with open(html_before_path, "r") as f:
            html = f.read()

        soup = bs4.BeautifulSoup(html, features="lxml")

        self.manipulate_document(soup, sorting = [], clusters_dict = {})

        with open(html_after_path, 'w') as f:
            f.write(str(soup))


import unittest

class TestPaperReader(unittest.TestCase):
    tfu = TrueFormatUpmarkerHTML(debug=True, parameterize=False)

    def test_working_and_file_existence(self):
        docs = [
            {
                'html_path_before': 'scraped_difference_between/Difference between Coronavirus and SARS.html',
                'html_path_after': 'scraped_difference_between/Difference between Coronavirus and SARS_html_test.html',
            }
        ]

        for kwargs in docs:
            self.tfu.convert_and_index(**kwargs)
            assert self.tfu.indexed_words
            assert os.path.exists(kwargs['html_path_after'])

if __name__ == '__main__':
    unittest.main()