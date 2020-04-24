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


@converter("text.txt", "topic_model")
class MakeTopics(PathSpec):
    def __init__(self, debug=True, *args, n=15, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.debug = debug
        self.state_path = self.temporary + self.path_spec._to

    @file_persistent_cached_generator(config.cache + 'topics.json')
    def __call__(self, txt_paths, *args, **kwargs):
        txt_paths, metas = list(zip(txt_paths))
        for txt_path  in txt_paths:
            os.system(f'ln {txt_path} {config.appcorpuscook_diff_txt_dir}')


        self.lda_model, self.vis = dariah.topics(directory=config.appcorpuscook_txt_dir,
                                                 stopwords=100,
                                                 num_topics=8,
                                                 num_iterations=50)

        print(self.lda_model.topics.iloc[:10, :5])

        self.doc2topic = self.lda_model.topic_document.idxmax(axis=0).to_dict()
        self.topic2doc = {}
        for doc, topic in self.doc2topic.items():
            if topic in self.topic2doc:
                self.topic2doc[topic].append(doc)
            else:
                self.topic2doc[topic] = [doc]

        self.headword2doc = {}
        for topic, doc_paths in self.topic2doc.items():
            self.headword2doc[self.create_headwords(doc_paths)] = doc_paths

        with open(self.state_path, "w") as f:
            state = {
                'state': str(self.docs_paths),
                'result': self.headword2doc
            }
            f.write(json.dumps(state))

        yield self.state_path, metas


