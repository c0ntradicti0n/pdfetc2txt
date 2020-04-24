import logging
import os
import urllib
from urllib.request import Request

import requests
from bs4 import BeautifulSoup
from layouteagle import config
from layouteagle.LayoutReader.trueformatpdf2htmlEX import TrueFormatUpmarkerPDF2HTMLEX
from layouteagle.helpers.cache_tools import file_persistent_cached_generator
from layouteagle.pathant.Converter import converter


@converter("text.json", "nlp.css")
class AnnotateJson(TrueFormatUpmarkerPDF2HTMLEX):
    def __init__(self, debug=True, *args, n=15, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.debug = debug

    @file_persistent_cached_generator(config.cache + '.json')
    def __call__(self, labeled_paths, *args, **kwargs):
        for doc_id, (text_json_path, meta) in enumerate(labeled_paths):

            requests.post(
                    url=f"http://localhost:{config.app_port}/annotate_certain_json_in_doc_folder",
                    json= {
                        'filename': text_json_path,
                        'meta': meta
                    },
                )

            yield doc_id + self.path_spec._to


