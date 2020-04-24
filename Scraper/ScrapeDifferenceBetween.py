import logging
import os
import urllib
from urllib.request import Request

from bs4 import BeautifulSoup
from layouteagle import config
from layouteagle.LayoutReader.trueformatpdf2htmlEX import TrueFormatUpmarkerPDF2HTMLEX
from layouteagle.helpers.cache_tools import file_persistent_cached_generator
from layouteagle.pathant.Converter import converter


@converter("differencebetween.net", "html")
class DifferncenceBetweenScraper(TrueFormatUpmarkerPDF2HTMLEX):
    def __init__(self, debug=True, *args, n=15, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.debug = debug

    @file_persistent_cached_generator(config.cache + 'pdf2html.json', )
    def __call__(self, labeled_paths, *args, **kwargs):
        for doc_id, (doc_path, meta) in enumerate(labeled_paths):
            logging.info("downloading front page of differencebetween")

            req = Request(
                'https://differencebetween.com',
                headers={'User-Agent': 'Mozilla/5.0'})

            f = urllib.request.urlopen(req)
            page = f.read()

            soup = BeautifulSoup(page, 'html.parser')
            anchors = list(soup.find_all('a', attrs={'rel': 'bookmark'}))[:n]
            for anchor in anchors:
                text_ground_path = anchor.get_text()
                html_path = config.appcorpuscook_diff_document_dir + text_ground_path + ".html"

                if not os.path.exists(html_path):
                    logging.info(f"downloading page for '{text_ground_path}'")
                    diffpagereq = Request(
                        anchor['href'],
                        headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(diffpagereq) as response:
                        html = response.read().decode("utf-8")
                        with open(html_path, 'w') as hf:
                            hf.write(html)
                        yield html_path


