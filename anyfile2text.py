import logging
import os
import time
import urllib
from collections import namedtuple
from statistics import mean
from urllib.request import urlopen
import bs4
import regex as re
from tika import parser
from scipy.stats import ks_2samp

import config
from TFU.trueformathtml import TrueFormatUpmarkerHTML
from TFU.trueformatpdf2htmlEX import TrueFormatUpmarkerPdf2HTMLEX
from helpers.str_tools import remove_ugly_chars

class PaperReader:
    """ multimedial extractor. it reads text from papers in pdfs, urls, html and other things.

        Formatting of text makes processing harder, text is cluttered up with remarks of the punlisher on every page,
        page and line numbers and other stuff, that must be ignored with the processing, especially, when joining the
        texts of different pages, where sentences continue.

        detecting text by comparing to the letter distribution of normal prose to parts of the text extracted.
    """

    def __init__(self, _threshold=0.001, _length_limit=20000):
        with open(config.wordlist, 'r') as f:
            self.wordlist = [w for w in list(f.readlines()) if len(w) >= 4]
        self.tfu_pdf = TrueFormatUpmarkerPdf2HTMLEX()
        self.tfu_html = TrueFormatUpmarkerHTML()

        self.length_limit = _length_limit
        self.threshold = _threshold
        self.normal_data = list(
            'used are variants of the predicate calculus. He  even says, Lately '
            'those who think  they ought to be so regarded seem to  be winning. '
            'Under these circumstances, it does seem odd for McDermott to devote '
            'much space to  complaining about the logical basis  of a book whose '
            'very title proclaims  it is about logical foundations. In any  '
            'case, given such a title, it wouldnt  seem necessary that readers '
            'should  be warned that the foundations being  explored are not '
            'In competition with this diversity  is the idea of a unified model '
            'of inference. The desire for such a model is  strong among those '
            'who study  declarative representations, and  Genesereth and Nilsson '
            'are no exception. As are most of their colleagues,  they are drawn '
            'to the model of inference as the derivation of conclusions  that '
            'are entailed by a set of beliefs.  They wander from this idea in a '
            'few  places but not for long. It is not hard  to see why: Deduction '
            'is one of the  fews kinds of inference for which we  have an '
            'interesting general theory. '.lower()
        )

    def just_extract_text_from_html(self, adress):
        logging.info(f"extracting text from {adress}")
        try:
            with urlopen(adress).read().decode('utf-8') as fdoc:
                soup = bs4.BeautifulSoup(fdoc, parent="lxml")
                return self.get_only_real_words(soup.get_text(), self.wordlist)
        except ValueError:
            with open(adress, "r") as fdoc:
                soup = bs4.BeautifulSoup(fdoc, features='lxml')
                return self.get_only_real_words(soup.get_text(), self.wordlist)

    def parse_file_format(self, adress):
        if adress.endswith('pdf'):
            paths = self.pdfpath2htmlpaths(adress)

            if config.parse_pdf2htmlEX:
                os.system(f"pdf2htmlEX  "
                          f"--optimize-text 1 "
                          f"--fit-width {config.reader_width}  "
                          f"\"{adress}\" \"{paths.html_before_indexing}\"")
            tfu = self.tfu_pdf
        elif adress.endswith('html'):
            tfu = self.tfu_html
            paths = self.htmlpath2htmlpaths(adress)
            logging.warning("trying with html...")
        else:
            logging.error(f"File '{adress}' could not be processed")
            return  None

        tfu.convert_and_index(paths.html_before_indexing, paths.html_after_indexing)
        tfu.save_doc_json(paths.json_path)
        os.system(f"cp \"{paths.html_after_indexing}\" \"{paths.apache_path}\"")
        self.text = " ".join(list(tfu.indexed_words.values()))


        # needed for topic modelling
        with open(paths.txt_path, "w") as f:
            f.write(self.text)
        logging.debug(paths)
        self.paths = paths
        time.sleep(2)

        logging.info(f"extracted text: {self.text[100:]}")
        return None

    def load_url(self, adress):
        response = urllib.request.urlopen(adress)
        data = response.read()  # a `bytes` object
        self.text = parser.from_buffer(data)

    def analyse(self):
        """
            Extracts prose text from  the loaded texts, that may contain line numbers somewhere, adresses, journal links etc.
        :return str:  prose text
        """
        logging.info("transferring text to CorpusCook...")

        paragraphs = self.text.split('\n\n')
        print("mean length of splitted lines", (mean([len(p) for p in paragraphs])))

        # If TIKA resolved '\n'
        if (mean([len(p) for p in paragraphs])) > 80:
            paragraphs = [re.sub(r"- *\n", '', p) for p in paragraphs]
            paragraphs = [p.replace('\n', " ") for p in paragraphs]
            paragraphs = [p.replace(';', " ") for p in paragraphs]
            joiner = " "
        else:
            # If TIKA did not
            joiner = " "

        processed_text = joiner.join([p
                                      for p in paragraphs
                                      if
                                      p and
                                      ks_2samp(self.normal_data, list(p)).pvalue > self.threshold
                                      ]
                                     )

        return processed_text.strip()[:self.length_limit]

    DocPaths = namedtuple("DocPaths", ["html_before_indexing",
                                       "html_after_indexing",
                                       "apache_path",
                                       "json_path",
                                       "txt_path"])

    def pdfpath2htmlpaths(self, adress):
        # file_extension = os.path.splitext(adress)[1] keep it, but unused
        # path = os.path.dirname(adress)
        filename = os.path.basename(adress)
        html_before_indexing = config.appcorpuscook_docs_document_dir + filename + ".html"
        filename = remove_ugly_chars(filename)
        html_after_indexing = config.appcorpuscook_docs_document_dir + filename + ".pdf2htmlEX.html"
        json_path = config.appcorpuscook_docs_json_dir + filename + ".json"
        txt_path = config.appcorpuscook_docs_txt_dir + filename + ".txt"
        apache_path = config.apache_dir_document + filename + ".html"

        return self.DocPaths(
            html_before_indexing,
            html_after_indexing,
            apache_path,
            json_path,
            txt_path)

    def get_only_real_words(self, text, wordlist):
        return text #" ".join([word for word in text.split() if word in wordlist])

    def htmlpath2htmlpaths(self, adress):
        filename = os.path.basename(adress)
        html_before_indexing = config.appcorpuscook_diff_document_dir + filename
        filename = remove_ugly_chars(filename)
        html_after_indexing = config.appcorpuscook_diff_html_dir + filename + ".pdf2htmlEX.html"
        json_path = config.appcorpuscook_diff_json_dir + filename + ".json"
        txt_path = config.appcorpuscook_docs_txt_dir + filename + ".txt"
        apache_path = config.apache_dir_document + filename + ".html"

        return self.DocPaths(
            html_before_indexing,
            html_after_indexing,
            apache_path,
            json_path,
            txt_path)

