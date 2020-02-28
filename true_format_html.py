import json
from collections import OrderedDict
from pprint import pprint
import itertools
from statistics import stdev
import regex
from numpy import mean
from helpers.color_logger import *
import bs4
import tinycss
from bs4 import Tag
from tinycss.css21 import RuleSet

NORMAL_HEIGHT = 100
INDEX_WRAP_TAG_NAME = 'z'

class TrueFormatUpmarker:
    def __init__(self):
        self.index_wrap_tag_name = INDEX_WRAP_TAG_NAME
        delimiters = r" ", r"\n"
        self.splitter = '|'.join(map(regex.escape, delimiters))

    def factory(self):
        pass

    def get_pdf2htmlEX_header(tag):
        return tag.attrs['class'][3]

    def convert_and_index (self, html_path_before, html_path_after):
        indexed_words = self.load(html_path_before, html_path_after)
        return indexed_words

    def transform (self, pdf_path):
        if pdf_path.endswith(".pdf"):
            html_path = pdf_path[:-4] + ".html"
            return html_path
        else:
            raise ValueError(f"{pdf_path} must be a pdffile!")

    def load(self, html_before_path, html_after_path):
        with open (html_before_path, 'r', encoding='utf8') as f:
            self.soup = bs4.BeautifulSoup(f.read(), features='html.parser')

        self.css_dict = self.get_css(self.soup)
        self.pages_list = list(self.fmr_pages(self.soup))
        self.columns_per_page = list(self.fmr_leftright(self.pages_list, css_dict=self.css_dict))
        self.reading_sequence_text = list(self.fmr_upbottum(self.columns_per_page, css_dict=self.css_dict))
        self.indexed_words = {}
        self.count_i = itertools.count()

        character_splitter = lambda divcontent: TrueFormatUpmarker.tokenize_differential_signs(divcontent)
        self.index_words(self.reading_sequence_text, splitter=character_splitter, eat_up=False)

        with open(html_after_path, "w", encoding='utf8') as file:
            file.write(str(self.soup).replace("<coolwanglu@gmail.com>", "coolwanglu@gmail.com"))
        return None

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
            ident = [sel for sel in rule.selector if sel.type=='IDENT'][0].value
            if not isinstance(ident, str):
                logging.info ("multi value csss found, ignoring")
                return None, None
            return ident, decla
        return None, None

    def fmr_pages(self, soup):
        return soup.select("div[data-page-no]")

    def fmr_leftright(self, pages, css_dict):
        # dividing all tags based on mid of page
        widths_declarations = {sel: self.get_declaration_value(decla, key="left") for sel, decla in css_dict.items() if regex.match('x.+', sel)}
        mid_width = self.get_declaration_value(css_dict['w0'], key='width')*0.4
        left_sels, right_sels = [], []
        for width_sel, decla in widths_declarations.items():
            (left_sels, right_sels)[decla > mid_width].append(width_sel)

        # selecting the tags on the pages
        for page in pages:
            text_divs =  page.select("div.t")
            text_divs =  self.fmr_height (text_divs, css_dict)
            widths2page = { }
            for text_div in text_divs:
                for attr in text_div.attrs['class']:
                    if attr.startswith('x'):
                        if attr in widths2page:
                            widths2page[attr].append(text_div)
                        else:
                            widths2page[attr] = [text_div]

            left, right = [], []
            for width, divs in widths2page.items():
                print ((len(left_sels)>len(right_sels)*0.6))
                if width in left_sels or (len(left_sels)>len(right_sels)*0.8): #(len(left_sels)>len(right_sels)*0.7):
                    left.extend(divs)
                elif width in right_sels:
                    right.extend(divs)
                else:
                    logging.info(f"could not put {div} into right nor left")
            yield [left, right]

    def fmr_upbottum(self, leftright_pages, css_dict):
        for left, right in leftright_pages:
            print ([self.get_css_decla_for_tag(div, css_dict, css_class='y', key='bottom') for div in left])
            yield from (
                sorted(left,  key=lambda div: -self.get_css_decla_for_tag(div, css_dict, css_class='y', key='bottom')) +
                sorted(right, key=lambda div: -self.get_css_decla_for_tag(div, css_dict, css_class='y', key='bottom')) )

    def get_declaration_value(self, declaration, key):
        try:
            return [decla.value[0].value for decla in declaration if decla.name==key][0]
        except:
            raise ValueError(f"{key} not in {str(declaration)}")

    def seq_split(lst, cond):
        sublst = []
        start = 0
        for i, item in enumerate(lst):
            if not cond(item):
                sublst.append(item)
            else:
                yield start, sublst
                sublst = []
                start = i+1
        if sublst:
            yield start, sublst


    keep_delims = r""",;.:'()[]{}&!?`/"""
    def tokenize_differential_signs(poss_token):
        try:
            list_of_words =  poss_token.split(" ")
        except:
            raise
        applying_delims = [d for d in TrueFormatUpmarker.keep_delims if d in poss_token]
        for d in applying_delims:
            intermediate_list = []
            for line in list_of_words:
                splitted = line.split(d)
                intermediate_list.extend([e + d if i < len(splitted) - 1 else e for i, e in enumerate(line.split(d)) if e])
            list_of_words = intermediate_list
        return list_of_words

    def make_new_tag(self, word):
        id = self.count_i.__next__()
        tag = self.soup.new_tag(self.index_wrap_tag_name, id=f'{INDEX_WRAP_TAG_NAME}{id}')
        tag.append(word)
        return (id, word), tag

    def index_words(self, text_divs, splitter=None, eat_up=True):
        """
            splitter is a function that gives back a list of 2-tuples, that mean the starting index,
            where to replace and list of tokens

        """

        space = self.soup.new_tag("span", {'class':'_'})
        space.append(" ")

        i = 0
        for text_div in text_divs:
            spaces = [tag for tag in text_div.contents if isinstance(tag, Tag) and tag.get_text()==" "]
            words = TrueFormatUpmarker.tokenize_differential_signs(text_div.get_text())
            if not words or not any(words):
                logging.info("no words were contained, empty text div")
                continue
            text_div.clear()
            css_notes, tagged_words = list(zip(*[self.make_new_tag(word) for word in words if word]))
            for i, tagged_word in enumerate(tagged_words[::-1]):
                try:
                    text_div.contents.insert(0, tagged_word)
                    text_div.contents.insert(0, spaces[i] if i< len(spaces) else space)
                except:
                    raise

            self.indexed_words.update(dict(css_notes))


    def get_css_decla_for_tag(self, div, css_dict, css_class, key ):
        if isinstance(div, Tag):
            try:
                print (self.get_declaration_value(css_dict[[attr for attr in div.attrs['class'] if attr.startswith(css_class)][0]], key=key))
                return self.get_declaration_value(css_dict[[attr for attr in div.attrs['class'] if attr.startswith(css_class)][0]], key=key)
            except:
                logging.warning(f"{key} not found for {div} un css_dict, returning 0")
                return 0
        else:
            return 0

    def fmr_height(self, text_divs, css_dict):
        heights_declarations = {sel: self.get_declaration_value(decla, key="height")
                                for sel, decla in css_dict.items() if regex.match('h.+', sel)}
        heights_declarations = {k:v for k,v in heights_declarations.items() if v < NORMAL_HEIGHT}
        sigma = stdev(list(heights_declarations.values())) *1.3
        mid_height = mean(list(heights_declarations.values()))
        text_divs_up_to_height = [text_div for text_div in text_divs
                                  if self.get_css_decla_for_tag(text_div, css_dict, 'h', 'height') <= mid_height + sigma
                                  and self.get_css_decla_for_tag(text_div, css_dict, 'h', 'height') >= mid_height - sigma]
        return text_divs_up_to_height

    def get_indexed_words(self):
        return self.indexed_words

    def save_doc_json(self, json_path):
        doc_dict = {
            "text": " ".join (list(self.indexed_words.values())),
            "indexed_words" : self.indexed_words}
        with open(json_path, "w", encoding="utf8") as f:
            f.write(json.dumps(doc_dict))

if __name__ == '__main__':
    tfu = TrueFormatUpmarker()
    #pprint(tfu.convert_and_index(html_path='/home/stefan/cow/pdfetc2txt/docs/0013.html'))
    #tfu.save_doc_json(json_path='/home/stefan/cow/pdfetc2txt/docs/0013.json')

    #tfu.convert_and_index(html_path='/home/stefan/cow/pdfetc2txt/docs/what is string theory.html')
    #pprint(tfu.get_indexed_words())

    tfu.convert_and_index(html_path_before='/home/stefan/cow/pdfetc2txt/docs/Laia Font-Ribera - Short-Term Changes in Respiratory Biomarkers after Swimmingin a Chlorinated Pool.pdf.pdf2htmlEX.html',
                          html_path_after='/home/stefan/cow/pdfetc2txt/docs/Laia Font-Ribera - Short-Term Changes in Respiratory Biomarkers after Swimmingin a Chlorinated Pool.pdf.pdf2htmlEX.af.html')
    pprint(tfu.get_indexed_words())
