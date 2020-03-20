import itertools
import json
import logging
import numpy
import tinycss
from bs4 import Tag
from more_itertools import flatten
from regex import regex
from tinycss.css21 import RuleSet
from zipp import OrderedDict
from pprint import pformat

import config


class TrueFormatUpmarker(object):
    def __init__(self, min_bottom=None, max_bottom=None, debug=False, parameterize=False):
        self.cuts = []
        self.text_divs = []
        self.debug = debug
        self.parameterize = parameterize
        din_rel = numpy.sqrt(2)
        if not min_bottom:
            self.min_bottom = 0
        else:
            self.min_bottom = min_bottom
        if not max_bottom:
            self.max_bottom = config.reader_width * din_rel  # * (1- config.page_margin_top)
        else:
            self.max_bottom = max_bottom

        self.WRAP_TAG = config.INDEX_WRAP_TAG_NAME

        self.CUT = "CUT"
        delimiters = r" ", r"\n"
        self.splitter = '|'.join(map(regex.escape, delimiters))

    def convert_and_index(self, html_path_before="", html_path_after=""):
        logging.warning(f"working on {html_path_before}")
        self.generate_css_tagging_document(html_path_before, html_path_after, "/debug_output")

    def get_indexed_words(self):
        return self.indexed_words

    def save_doc_json(self, json_path):
        doc_dict = {
            "text": " ".join(list(self.indexed_words.values())),
            "indexed_words": self.indexed_words,
            "cuts": self.cuts}
        with open(json_path, "w", encoding="utf8") as f:
            f.write(json.dumps(doc_dict))

    def add_text_coverage_markup(self, soup):
        z_style = "\nz {background: rgba(0, 0, 0, 1) !important;   font-weight: bold;} "
        soup.head.append(soup.new_tag('style', type='text/css'))
        soup.head.style.append(z_style)

    def manipulate_document(self,
                            soup,
                            **kwargs
                            ):
        self.indexed_words = {}  # reset container for words
        self.count_i = itertools.count()  # counter for next indices for new html-tags
        character_splitter = lambda divcontent: TrueFormatUpmarker.tokenize_differential_signs(divcontent)
        text_divs = self.collect_all_divs(soup)
        self.index_words(soup=soup,
                         **kwargs,
                         text_divs=text_divs,
                         )
        if self.debug:
            self.add_text_coverage_markup(soup)

    def get_css(self, soup):
        css_parts = soup.select('style[type="text/css"]')
        big_css = max(css_parts, key=lambda x: len(x.string))
        style_rules = tinycss.make_parser().parse_stylesheet(big_css.string, encoding="utf8").rules
        style_dict = OrderedDict(self.css_rule2entry(rule) for rule in style_rules)
        if None in style_dict:
            del style_dict[None]
        return style_dict

    def css_rule2entry(self, rule):
        if isinstance(rule, RuleSet):
            decla = rule.declarations
            ident = [sel for sel in rule.selector if sel.type == 'IDENT'][0].value
            if not isinstance(ident, str):
                logging.info("multi value css found, ignoring")
                return None, None
            return ident, decla
        return None, None

    keep_delims = r""",;.:'()[]{}&!?`/"""

    def tokenize_differential_signs(poss_token):
        try:
            words = poss_token.split(" ")
        except:
            raise
        applying_delims = [d for d in TrueFormatUpmarker.keep_delims if d in poss_token]
        for d in applying_delims:
            intermediate_list = []
            for line in words:
                splitted = line.split(d)
                # flattenning double list is done because two things splitted on tokens need
                # to be splittted into three: token, delimiter, token
                intermediate_list.extend(flatten([[e, d] if i < len(splitted) - 1 else [e]
                                                  for i, e in enumerate(line.split(d)) if e]))
            words = intermediate_list
        words = [word.strip() for word in words if word.strip()]
        return words

    def make_new_tag(self, soup, word, debug_percent, **kwargs):
        self.id = self.count_i.__next__()
        if self.debug:
            kwargs.update({"style": f"color:hsl({int(debug_percent * 360)}, 100%, 50%);"})
        tag = soup.new_tag(self.WRAP_TAG, id=f"{self.WRAP_TAG}{self.id}",
                           **kwargs)

        tag.append(word)
        return (self.id, word), tag


    def index_words(self,
                    soup,  # for generating new tags
                    sorting,
                    text_divs,
                    clusters_dict,
                    ):
        """
            splitter is a function that gives back a list of 2-tuples, that mean the starting index,
            where to replace and list of tokens
        """
        space = soup.new_tag("span", {'class': '_'})
        space.append(" ")
        if sorting:
            text_divs = [text_divs[i] for i in sorting]

        for div_index, text_div in enumerate(text_divs):
            if div_index == self.CUT:
                # If here was observed a possible cut, append it to slice the text here possibly.
                self.cuts.append(self.id)

            if clusters_dict and div_index not in clusters_dict:
                # this happens for the page containers
                continue
            if clusters_dict and not self.take_outliers and clusters_dict[div_index] == -1:
                # excluding outliers
                continue
            if clusters_dict:
                debug_percent = (abs(clusters_dict[div_index] + 2) / (10))
            else:
                debug_percent = 0.5

            if hasattr(text_div, "contents"):
                spaces = [] #[tag for tag in text_div.contents if isinstance(tag, Tag) and tag.get_text() == " "]

            # As side effect the elements of the html soup are wrapped in a new bs4 tag-element

            # a itself ending tag <bla/> cant have a text and throws an error
            try:
                text_to_split = text_div.get_text()
            except AttributeError:
                logging.debug("yet changed tag")
                continue
            words = TrueFormatUpmarker.tokenize_differential_signs(text_to_split)

            if not words or not any(words):
                logging.debug("tag without text")
                continue

            logging.debug(f"putting words into tag again {pformat(words[:100], compact=True)}")
            text_div.clear()
            ids2words_tagwords = [self.make_new_tag(
                soup,
                word,
                debug_percent=debug_percent)
                for word in words]
            if ids2words_tagwords:
                css_ids2words, tagged_words = list(
                            zip(*ids2words_tagwords))

            for i, tagged_word in enumerate(tagged_words[::-1]):
                text_div.contents.insert(0, tagged_word)
                text_div.contents.insert(0, spaces[i] if i < len(spaces) else space)

            self.indexed_words.update(dict(css_ids2words))

    def collect_all_divs(self, soup):
        raise NotImplementedError

