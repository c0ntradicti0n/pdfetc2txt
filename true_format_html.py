import json
from collections import OrderedDict
from pprint import pprint
import itertools
from statistics import stdev

import numpy
import pandas
import regex
from more_itertools.recipes import flatten
from numpy import mean
from scipy.stats import ks_2samp
from sklearn.utils import Memory
import matplotlib.pyplot as plt
import seaborn as sns

from helpers.color_logger import *
import bs4
import tinycss
from bs4 import Tag
from tinycss.css21 import RuleSet
import hdbscan

from helpers.list_tools import reverse_dict_of_lists


def normalized(a, axis=-1, order=2):
    l2 = numpy.atleast_1d(numpy.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / numpy.expand_dims(l2, axis)

NORMAL_HEIGHT = 100
INDEX_WRAP_TAG_NAME = 'z'

algorithms = ['best','generic','prims_kdtree','prims_balltree','boruvka_kdtree','boruvka_balltree']
metrics = {'braycurtis': hdbscan.dist_metrics.BrayCurtisDistance,
 'canberra': hdbscan.dist_metrics.CanberraDistance,
 'chebyshev': hdbscan.dist_metrics.ChebyshevDistance,
 'cityblock': hdbscan.dist_metrics.ManhattanDistance,
 'dice': hdbscan.dist_metrics.DiceDistance,
 'euclidean': hdbscan.dist_metrics.EuclideanDistance,
 'hamming': hdbscan.dist_metrics.HammingDistance,
 'haversine': hdbscan.dist_metrics.HaversineDistance,
 'infinity': hdbscan.dist_metrics.ChebyshevDistance,
 'jaccard': hdbscan.dist_metrics.JaccardDistance,
 'kulsinski': hdbscan.dist_metrics.KulsinskiDistance,
 'l1': hdbscan.dist_metrics.ManhattanDistance,
 'l2': hdbscan.dist_metrics.EuclideanDistance,
 'mahalanobis': hdbscan.dist_metrics.MahalanobisDistance,
 'manhattan': hdbscan.dist_metrics.ManhattanDistance,
 'matching': hdbscan.dist_metrics.MatchingDistance,
 'minkowski': hdbscan.dist_metrics.MinkowskiDistance,
 'p': hdbscan.dist_metrics.MinkowskiDistance,
 'pyfunc': hdbscan.dist_metrics.PyFuncDistance,
 'rogerstanimoto': hdbscan.dist_metrics.RogersTanimotoDistance,
 'russellrao': hdbscan.dist_metrics.RussellRaoDistance,
 'seuclidean': hdbscan.dist_metrics.SEuclideanDistance,
 'sokalmichener': hdbscan.dist_metrics.SokalMichenerDistance,
 'sokalsneath': hdbscan.dist_metrics.SokalSneathDistance,
 'wminkowski': hdbscan.dist_metrics.WMinkowskiDistance}

NORMAL_TEXT = list(
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
    'interesting general theory. tively. The genotyping frequency was 91%  of occupational asthma and trichloramine'
 'sensitization have been described in pool life-'.lower()
)
class TrueFormatUpmarker:
    def __init__(self):
        self.index_wrap_tag_name = INDEX_WRAP_TAG_NAME
        delimiters = r" ", r"\n"
        self.splitter = '|'.join(map(regex.escape, delimiters))


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
        for algorithm in ['boruvka_balltree', 'boruvka_kdtree', 'generic', 'prims']:
            for metric in ['hamming', 'infinity', 'p', 'chebyshev', 'cityblock', 'braycurtis', 'euclidean']:
                try:
                    with open (html_before_path, 'r', encoding='utf8') as f:
                        self.soup = bs4.BeautifulSoup(f.read(), features='html.parser')

                    self.css_dict = self.get_css(self.soup)
                    self.pages_list = list(self.fmr_pages(self.soup))
                    #self.clusters_dict = self.fmr_hdbscan(self.pages_list, self.css_dict, metric=metric, algorithm=algorithm)
                    self.columns_per_page = list(self.fmr_leftright(self.pages_list, css_dict=self.css_dict))
                    self.reading_sequence_text = list(self.fmr_upbottum(self.columns_per_page, css_dict=self.css_dict))
                    self.indexed_words = {}
                    self.count_i = itertools.count()

                    character_splitter = lambda divcontent: TrueFormatUpmarker.tokenize_differential_signs(divcontent)
                    self.index_words(self.reading_sequence_text, splitter=character_splitter, eat_up=False, clusters_dict=self.clusters_dict)

                    with open(html_after_path+metric+algorithm+".html", "w", encoding='utf8') as file:
                        file.write(str(self.soup).replace("<coolwanglu@gmail.com>", "coolwanglu@gmail.com"))
                except Exception as e:
                    logging.info(f"plotting failes for {algorithm} {metric}")
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

    def fmr_hdbscan(self, pages, css_dict, debug_pic_name="debug_pics/output.png", **kwargs):
        """
        hdbscan on font height, x position and y position to recognize all groups of textboxes in different parts of
        the layout as footnotes, textcolumns, headers etc.
        """
        page2divs = [page.select('div[class*=x]') for page in pages]
        all_divs = list(flatten(page2divs))
        hs_xs_ys = [list(self.getxyh(tag, css_dict)) for tag in all_divs]
        text_prob = [[
                      2+page_number,
                      len(div.text),
                      0]#ks_2samp(NORMAL_TEXT, list(div.text.lower())).pvalue *100 if div.text else 0]
                      for page_number, divs in enumerate(page2divs) for div in divs]
        assert(len(hs_xs_ys)==len(text_prob))

        data = numpy.array([list(hxy) + tp for hxy, tp in zip(hs_xs_ys, text_prob)])

        """for metric in metrics.keys():
            for algorithm in algorithms:
                try:
                    clusterer = hdbscan.HDBSCAN( metric=metric, algorithm=algorithm,
                       min_cluster_size=40, alpha=0.95).   fit(data)
                    threshold = pandas.Series(clusterer.outlier_scores_).quantile(0.8)
                    outliers = numpy.where(clusterer.outlier_scores_ > threshold)[0]

                    color_palette = sns.color_palette('deep', 8)
                    cluster_colors = [color_palette[x] if x >= 0
                                      else (0.5, 0.5, 0.5)
                                      for x in clusterer.labels_]
                    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                             zip(cluster_colors, clusterer.probabilities_)]
                    coords = data.T[1:3]
                    plt.scatter(*list(coords), c=cluster_member_colors, linewidth=0)
                    plt.scatter(*list(data[outliers].T[1:3]), s=50, linewidth=0, c='red')
                    plt.savefig(debug_pic_name+"."+algorithm+"_"+metric+"_"+".png", bbox_inches='tight')
                except Exception as e:
                    logging.info(f"Plotting failed {str(e)}" )
        # best choice was
        """
        clusterer = hdbscan.HDBSCAN(**kwargs,
                                    min_cluster_size=40, alpha=0.95).fit(normalized(data))
        threshold = pandas.Series(clusterer.outlier_scores_).quantile(0.8)
        outliers = numpy.where(clusterer.outlier_scores_ > threshold)[0]


        # to collect all non text-stuff, look for one or more columns, that end at the same height
        # make groups of indices
        cluster2indexlist = {
            k: [t[0] for t in  g] for k,g in
                itertools.groupby(list(sorted (enumerate(clusterer.labels_),
                                       key=lambda t:t[1])),
                                  key= lambda t:t[1])
            }
        # left lowest corners
        cluster2minbottom_minleft = {
            k: (min(hs_xs_ys [i][0] for i in g),
                min(hs_xs_ys[i][1] for i in g),
                min(hs_xs_ys[i][2] for i in g),
                len(g))

                for k, g in cluster2indexlist.items()

        }
        clusters_dict = {all_divs[k]: (abs(i[0]+2)/(len(cluster2indexlist)+2))  if k not in outliers else 1
             for k, i in reverse_dict_of_lists(cluster2indexlist).items()}
        return clusters_dict



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

    def make_new_tag(self, word, debug_percent):
        p = 1 if debug_percent == 1 else 40
        id = self.count_i.__next__()
        tag = self.soup.new_tag(self.index_wrap_tag_name, id=f'{INDEX_WRAP_TAG_NAME}{id}', style=f"color:hsl({int(debug_percent*100)}, {p}%, {p}%);" )
        tag.append(word)
        return (id, word), tag

    def index_words(self, text_divs, splitter=None, eat_up=True, clusters_dict ={}):
        """
            splitter is a function that gives back a list of 2-tuples, that mean the starting index,
            where to replace and list of tokens

        """

        space = self.soup.new_tag("span", {'class':'_'})
        space.append(" ")

        i = 0
        for text_div in text_divs:
            debug_percent = clusters_dict[text_div]
            spaces = [tag for tag in text_div.contents if isinstance(tag, Tag) and tag.get_text()==" "]
            words = TrueFormatUpmarker.tokenize_differential_signs(text_div.get_text())
            if not words or not any(words):
                logging.info("no words were contained, empty text div")
                continue
            text_div.clear()
            css_notes, tagged_words = list(zip(*[self.make_new_tag(word, debug_percent=debug_percent) for word in words if word]))
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

    def getxyh(self, tag, css_dict):
        ht,xt,yt = sorted (attr for attr in tag.attrs['class']
                           if attr.startswith('x') or
                              attr.startswith('y') or
                              attr.startswith('h'))
        hxys = [self.get_declaration_value(css_dict[ht], 'height'),
                self.get_declaration_value(css_dict[xt], 'left'),
                self.get_declaration_value(css_dict[yt], 'bottom')]

        hxys = hxys+ [numpy.sqrt(((hxys[1]-300)**2/(hxys[1]-300)**2+(hxys[2]-600)**2))]
        return hxys

if __name__ == '__main__':
    tfu = TrueFormatUpmarker()
    #pprint(tfu.convert_and_index(html_path='/home/stefan/cow/pdfetc2txt/docs/0013.html'))
    #tfu.save_doc_json(json_path='/home/stefan/cow/pdfetc2txt/docs/0013.json')

    #tfu.convert_and_index(html_path='/home/stefan/cow/pdfetc2txt/docs/what is string theory.html')
    #pprint(tfu.get_indexed_words())

    tfu.convert_and_index(html_path_before='/home/stefan/cow/pdfetc2txt/docs/Laia Font-Ribera - Short-Term Changes in Respiratory Biomarkers after Swimmingin a Chlorinated Pool.pdf.pdf2htmlEX.html',
                          html_path_after='/home/stefan/cow/pdfetc2txt/docs/Laia Font-Ribera - Short-Term Changes in Respiratory Biomarkers after Swimmingin a Chlorinated Pool.pdf.pdf2htmlEX.af.html')
    pprint(tfu.get_indexed_words())
