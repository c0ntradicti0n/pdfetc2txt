import logging

import requests
from layouteagle import config
from layouteagle.helpers.cache_tools import file_persistent_cached_generator
from layouteagle.pathant.Converter import converter
from layouteagle.pathant.PathSpec import PathSpec


@converter("text.json", "nlp.css")
class AnnotateJson(PathSpec):
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


