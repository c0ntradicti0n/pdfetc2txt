import json
import logging
import os
import urllib
from urllib.request import Request

import dariah
import requests
from bs4 import BeautifulSoup
from layouteagle import config
from layouteagle.LayoutReader.trueformatpdf2htmlEX import TrueFormatUpmarkerPDF2HTMLEX
from layouteagle.helpers.cache_tools import file_persistent_cached_generator
from layouteagle.pathant.Converter import converter
from layouteagle.pathant.PathSpec import PathSpec

from topic_modelling import rake


@converter("topic_model", "topic_prediction")
class PredictTopics(PathSpec):
    def __init__(self, debug=True, *args, n=15, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.debug = debug
        self.state_path = self.temporary + self.path_spec._to

    @file_persistent_cached_generator(config.cache + 'topics.json')
    def __call__(self, txt_paths, *args, **kwargs):
        txt_paths, metas = list(zip(txt_paths))

        for path in txt_paths:
            with open(self.directory + "/" + path + ".txt", 'r+', errors='ignore') as f:
                text += f.read() + " "
                text = self.clean_text(text)
                poss_headwords = rake.apply(text)


            if poss_headwords:
                result = poss_headwords[0][0]

                yield result, metas



