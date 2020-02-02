import logging
import urllib
from statistics import mean

import regex as re
from tika import parser
from scipy.stats import ks_2samp

class paper_reader:
    """ multimedial extractor. it reads text from papers in pdfs, urls, html and other things.

        Formatting of text makes processing harder, text is cluttered up with remarks of the punlisher on every page,
        page and line numbers and other stuff, that must be ignored with the processing, especially, when joining the
        texts of different pages, where sentences continue.

        detecting text by comparing to the letter distribution of normal prose to parts of the text extracted.
    """
    def __init__(self, _threshold = 0.001, _length_limit = 20000):
        #with open('hamlet.txt', 'r+') as f:
        #    self.normal_data = list(f.read())

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

    def load_text(self, adress):
        logging.info("tika reading text...")
        self.rawText = parser.from_file(adress)

    def load_url(self, adress):
        response = urllib.request.urlopen(adress)
        data = response.read()  # a `bytes` object
        self.rawText = parser.from_buffer(data)

    def analyse(self):
        """ Extracts prose text from  the loaded texts, that may contain line numbers somewhere, adresses, journal links etc.
        :return str:  prose text
        """
        logging.info("transferring text to nlp thing...")

        text = self.rawText['content']
        paragraphs = text.split('\n\n')
        print ("mean length of splitted lines", (mean([len(p) for p in paragraphs])))

        # If TIKA resolved '\n'
        if (mean([len(p) for p in paragraphs])) > 80:
            paragraphs = [re.sub(r"- *\n", '', p) for p in paragraphs]
            paragraphs = [p.replace('\n', " ") for p in paragraphs]
            paragraphs = [p.replace(';', " ") for p in paragraphs]
            joiner = " "
        else:
            # If TIKA did not resolve '\n'
            joiner = " "

        processed_text = joiner.join([p
              for p in paragraphs
                   if
                        p and
                        ks_2samp(self.normal_data, list(p)).pvalue   >   self.threshold
                                      ]
                                     )

        return processed_text.strip() [:self.length_limit]
