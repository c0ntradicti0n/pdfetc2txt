import io
import json
from collections import OrderedDict, Counter, defaultdict
from enum import Enum
from pprint import pprint
import itertools
from statistics import stdev
import numpy
import pandas
import cv2
import regex
from more_itertools.recipes import flatten
from numpy import mean
from PIL import Image
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import ks_2samp
from sklearn.utils import Memory
import matplotlib.pyplot as plt
import seaborn as sns

import config
from HoughBundler import HoughBundler
from helpers.color_logger import *
import bs4
import tinycss
from bs4 import Tag
from tinycss.css21 import RuleSet
import hdbscan

from helpers.str_tools import insert_at_index
from helpers.list_tools import reverse_dict_of_lists


class Page_Features:
    page_number = 6
    text_len = 7
    text_distribution = 8
    dist_ = 5
    height = 2
    font_size = 0
    line_height = 1
    left = 3
    bottom = 4
    density = 9
    label_ = 10
    index_ = 11

def normalized(a, axis=-1, order=2):
    l2 = numpy.atleast_1d(numpy.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / numpy.expand_dims(l2, axis)


NORMAL_HEIGHT = 100
INDEX_WRAP_TAG_NAME = 'z'

algorithms = ['best', 'generic', 'prims_kdtree', 'prims_balltree', 'boruvka_kdtree', 'boruvka_balltree']
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
    'should  be warned that the foundations being  explored are not 100'
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
    def __init__(self, min_bottom=None, max_bottom=None):
        din_rel = numpy.sqrt(2)
        if not min_bottom:
            self.min_bottom = 0 #config.reader_width * din_rel * config.page_margin_bottom
        else:
            self.min_bottom = min_bottom
        if not max_bottom:
            self.max_bottom =  config.reader_width * din_rel #* (1- config.page_margin_top)
        else:
            self.max_bottom = max_bottom

        self.index_wrap_tag_name = INDEX_WRAP_TAG_NAME
        delimiters = r" ", r"\n"
        self.splitter = '|'.join(map(regex.escape, delimiters))

    def get_pdf2htmlEX_header(tag):
        return tag.attrs['class'][3]

    def convert_and_index(self, html_path_before, html_path_after):
        indexed_words = self.load(html_path_before, html_path_after, "/debug_output")
        return indexed_words

    def transform(self, pdf_path):
        if pdf_path.endswith(".pdf"):
            html_path = pdf_path[:-4] + ".html"
            return html_path
        else:
            raise ValueError(f"{pdf_path} must be a pdffile!")

    def load(self, html_before_path, html_after_path, debug_folder):
        for algorithm in ['boruvka_balltree']:#, 'boruvka_kdtree', 'generic', 'prims']:
            for metric in ['hamming']:#['l1', 'l2', 'hamming', 'infinity', 'p', 'chebyshev', 'cityblock', 'braycurtis', 'euclidean']:
                file_name = insert_at_index(html_after_path, html_after_path.rfind("/"),
                                              debug_folder)  + metric + algorithm

                try:
                    with open(html_before_path, 'r', encoding='utf8') as f:
                        self.soup = bs4.BeautifulSoup(f.read(), features='html.parser')

                    self.css_dict = self.get_css(self.soup)
                    self.pages_list = list(self.fmr_pages(self.soup))
                    self.divs_in_cols_lefrig_botup_sorted, self.clusters_dict = self.fmr_hdbscan(
                          [Page_Features.left,
                           Page_Features.line_height],
                          self.pages_list,
                          self.css_dict,
                          metric=metric,
                          algorithm=algorithm,
                          debug_pic_name=file_name+"_horizontal_",
                          cluster_selection_epsilon = 0.2,
                          allow_single_cluster = False,
                          min_cluster_size = 100,
                          min_samples = 50,
                          alpha = 0.9999
                    )

                    self.indexed_words         = {} # container for words
                    self.count_i               = itertools.count() # indices for tags
                    character_splitter = lambda divcontent: TrueFormatUpmarker.tokenize_differential_signs(divcontent)
                    self.index_words(self.divs_in_cols_lefrig_botup_sorted,
                                     splitter=character_splitter,
                                     eat_up=False,
                                     clusters_dict=self.clusters_dict)

                    self.add_text_coverage(self.soup)
                    with open(file_name + ".html", "w",
                              encoding='utf8') as file:
                        file.write(str(self.soup).replace("<coolwanglu@gmail.com>", "coolwanglu@gmail.com"))
                except Exception as e:
                    raise #
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

    houghbundler = HoughBundler(min_distance_to_merge=80, min_angle_to_merge=30)

    def css_rule2entry(self, rule):
        if isinstance(rule, RuleSet):
            decla = rule.declarations
            ident = [sel for sel in rule.selector if sel.type == 'IDENT'][0].value
            if not isinstance(ident, str):
                logging.info("multi value css found, ignoring")
                return None, None
            return ident, decla
        return None, None

    def fmr_hdbscan(self,
                    features_to_use,
                    pages,
                    css_dict,
                    debug_pic_name="debug_pics/output.png",
                    debug = True,
                    **kwargs):
        """
        hdbscan on font height, x position and y position to recognize all groups of textboxes in different parts of
        the layout as footnotes, textcolumns, headers etc.
        """

        # Collect divs (such, that they have an x... attribute, that is generated by pdf2htmlEX
        page2divs = [page.select('div[class*=x]') for page in pages]
        all_divs = list(flatten(page2divs))

        # Generate features
        hs_xs_ys = [list(self.getxyh(tag, css_dict)) for tag in all_divs]
        text_prob = [[
            2 + page_number,
            len(div.text),
            0]  # ks_2samp(NORMAL_TEXT, list(div.text.lower())).pvalue *100 if div.text else 0]
            for page_number, divs in enumerate(page2divs) for div in divs]
        assert (len(hs_xs_ys) == len(text_prob))
        data = [list(hxy) + tp for hxy, tp in zip(hs_xs_ys, text_prob)]
        data = [pf
                if pf[Page_Features.bottom] > self.min_bottom
                and pf[Page_Features.bottom] < self.max_bottom
                else [0, 0, 0, 0, 0, 0, 0, 0, 0]
                for pf in data
                ]
        data = numpy.array(data)
        coords = data.T[[Page_Features.left, Page_Features.bottom]]
        densities_at_points, density_field = self.point_density_frequence(points2D=coords.T)
        data = numpy.column_stack((data, densities_at_points))

        # Clustering
        clusterer = hdbscan.HDBSCAN(**kwargs)
        clusterer.fit(normalized(data[:, features_to_use]))
        threshold = pandas.Series(clusterer.outlier_scores_).quantile(0.8)
        outliers = numpy.where(clusterer.outlier_scores_ > threshold)[0]

        # Determine number of clusters
        number_columns = self.number_of_columns(density2D=density_field)
        logging.info(f"detected {number_columns} columns")
        what_clusters = set(clusterer.labels_)
        cluster_counts = Counter([l for l in clusterer.labels_ if l>0])
        relevant_clusters = sorted(cluster_counts, key=cluster_counts.get)[-number_columns:]

        if debug:
            with open(debug_pic_name +".txt", "w") as f:
                f.write(f"Which clusters are there? {str(what_clusters)}")
                f.write(f"number of relevant columns {number_columns}")
                f.write(f"content items in columns {str(cluster_counts)}")

            self.debug_pic(clusterer, coords, debug_pic_name, outliers)

        # Ordering pages and columns in dicts
        # page_groups[page_number] = [...]
        #                             left to right sorting of page clusters (not all divs
        #                             together!)

        # collecting divs per page (and append clustering label and index to the features)
        groups_of_pages = defaultdict(list)
        for feature_line in zip(*list(data.T), clusterer.labels_, range(len(clusterer.labels_))):
            groups_of_pages[feature_line[Page_Features.page_number]].append(
                feature_line)

        # sort these groups of clusters on pages
        page_cluster_lr_groups = defaultdict(list)
        for page, page_group in groups_of_pages.items():
            # group the clusters per page
            # itertools would break up the clusters, when the clusters
            # would not be sorted on the labels
            # result is {pages:{clusters{sorted by left to right}}}
            # TODO right to left and up to down!

            page_group = sorted(page_group,
                                key=lambda r: r[Page_Features.label_])
            groups_of_clusters = itertools.groupby(page_group,
                                                   key=lambda s: s[Page_Features.label_])
            page_cluster_up_bottom_groups = \
                [ (cluster, sorted(cluster_group,
                         key=lambda r: r[Page_Features.bottom]))
                    for cluster, cluster_group in groups_of_clusters
                    if cluster in relevant_clusters
                ]
            page_cluster_lr_groups[page] = \
                sorted(page_cluster_up_bottom_groups, key=lambda r:
                       min(r[1], key=lambda s: s[Page_Features.left]))

        # clusters_dict = {all_divs[index]: (abs(cluster_label[0] + 2) / (len(cluster2indexlist) + 2)) if index not in outliers else 1
        #                 for index, cluster_label in reverse_dict_of_lists(cluster2indexlist).items()
        #                 if cluster_label[0] != -1}


        clusters_dict = {all_divs[features[Page_Features.index_]] : cluster_label
                          for page, clusters in page_cluster_lr_groups.items()
                          for cluster_label, cluster in clusters
                          for features in cluster}


        div_reading_sequence = [all_divs[features[Page_Features.index_]]
                          for page, clusters in page_cluster_lr_groups.items()
                          for cluster_label, cluster in clusters
                          for features in cluster]

        return div_reading_sequence, clusters_dict

    def debug_pic(self, clusterer, coords, debug_pic_name, outliers):
        color_palette = sns.color_palette('deep', 20)
        cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in clusterer.labels_]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                 zip(cluster_colors, clusterer.probabilities_)]
        plt.scatter(*list(coords), c=cluster_member_colors, linewidth=0)
        #plt.scatter(*list(coords[:,outliers].T), linewidth=0, c='red')
        plt.savefig(debug_pic_name + ".png", bbox_inches='tight')

    def fmr_pages(self, soup):
        return soup.select("div[data-page-no]")

    def fmr_leftright(self, pages, css_dict, clusters_dict):

        # selecting the tags on the pages
        column_clusters = {}
        for page in pages:
            text_divs = page.select("div.t")
            rl_text_divs = sorted(text_divs,
                                  key=lambda div:
                                  self.get_css_decla_for_tag(div, css_dict, 'h', 'height'))
            for div in rl_text_divs:
                if div in clusters_dict:
                    cluster = clusters_dict[div]
                    if cluster not in column_clusters:
                        column_clusters[cluster] = []
                    column_clusters[cluster].append(div)

        return column_clusters

    def fmr_upbottum(self, leftright_pages, css_dict):
        for left, right in leftright_pages:
            #print([self.get_css_decla_for_tag(div, css_dict, css_class='y', key='bottom') for div in left])
            yield from (
                    sorted(left,
                           key=lambda div: -self.get_css_decla_for_tag(div, css_dict, css_class='y', key='bottom')) +
                    sorted(right,
                           key=lambda div: -self.get_css_decla_for_tag(div, css_dict, css_class='y', key='bottom')))

    def get_declaration_value(self, declaration, key):
        try:
            return [decla.value[0].value for decla in declaration if decla.name == key][0]
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
                start = i + 1
        if sublst:
            yield start, sublst

    keep_delims = r""",;.:'()[]{}&!?`/"""

    def tokenize_differential_signs(poss_token):
        try:
            list_of_words = poss_token.split(" ")
        except:
            raise
        applying_delims = [d for d in TrueFormatUpmarker.keep_delims if d in poss_token]
        for d in applying_delims:
            intermediate_list = []
            for line in list_of_words:
                splitted = line.split(d)
                intermediate_list.extend([e + d if i < len(splitted) - 1 else e
                                          for i, e in enumerate(line.split(d)) if e])
            list_of_words = intermediate_list
        return list_of_words

    def make_new_tag(self, word, debug_percent):
        p = 1 if debug_percent == 1 else 99
        id = self.count_i.__next__()
        tag = self.soup.new_tag(self.index_wrap_tag_name, id=f'{INDEX_WRAP_TAG_NAME}{id}',
                                style=f"color:hsl({int(debug_percent * 360)}, 100%, 50%);")
        tag.append(word)
        return (id, word), tag

    def index_words(self, text_divs, splitter=None, eat_up=True, clusters_dict={}):
        """
            splitter is a function that gives back a list of 2-tuples, that mean the starting index,
            where to replace and list of tokens
        """
        space = self.soup.new_tag("span", {'class': '_'})
        space.append(" ")

        for text_div in text_divs:
            if text_div not in clusters_dict or clusters_dict[text_div] == 1:
                logging.info("excluding non-main-text" + str(text_div.contents)[:36])
                continue
            debug_percent = clusters_dict[text_div]

            spaces = [tag for tag in text_div.contents if isinstance(tag, Tag) and tag.get_text() == " "]
            words = TrueFormatUpmarker.tokenize_differential_signs(text_div.get_text())
            if not words or not any(words):
                logging.info("found an empty text div, excluing")
                continue
            text_div.clear()
            css_notes, tagged_words = list(
                zip(*[self.make_new_tag(word, debug_percent=debug_percent) for word in words if word]))
            for i, tagged_word in enumerate(tagged_words[::-1]):
                try:
                    text_div.contents.insert(0, tagged_word)
                    text_div.contents.insert(0, spaces[i] if i < len(spaces) else space)
                except:
                    raise

            self.indexed_words.update(dict(css_notes))

    def get_css_decla_for_tag(self, div, css_dict, css_class, key):
        if isinstance(div, Tag):
            try:
                #print(self.get_declaration_value(
                #    css_dict[[attr for attr in div.attrs['class'] if attr.startswith(css_class)][0]], key=key))
                return self.get_declaration_value(
                    css_dict[[attr for attr in div.attrs['class'] if attr.startswith(css_class)][0]], key=key)
            except:
                logging.warning(f"{key} not found for {div} un css_dict, returning 0")
                return 0
        else:
            return 0

    def fmr_height(self, text_divs, css_dict):
        heights_declarations = {sel: self.get_declaration_value(decla, key="height")
                                for sel, decla in css_dict.items() if regex.match('h.+', sel)}
        heights_declarations = {k: v for k, v in heights_declarations.items() if v < NORMAL_HEIGHT}
        sigma = stdev(list(heights_declarations.values())) * 1.3
        mid_height = mean(list(heights_declarations.values()))
        text_divs_up_to_height = [text_div for text_div in text_divs
                                  if self.get_css_decla_for_tag(text_div, css_dict, 'h', 'height') <= mid_height + sigma
                                  and self.get_css_decla_for_tag(text_div, css_dict, 'h',
                                                                 'height') >= mid_height - sigma]
        return text_divs_up_to_height

    def get_indexed_words(self):
        return self.indexed_words

    def save_doc_json(self, json_path):
        doc_dict = {
            "text": " ".join(list(self.indexed_words.values())),
            "indexed_words": self.indexed_words}
        with open(json_path, "w", encoding="utf8") as f:
            f.write(json.dumps(doc_dict))

    point_before = (0,0)
    def getxyh(self, tag, css_dict):
        try:
            fft, fst, ht, xt, yt = sorted(attr for attr in tag.attrs['class']
                                if  attr.startswith('x') or
                                    attr.startswith('y') or
                                    attr.startswith('h') or
                                    attr.startswith('ff') or
                                    attr.startswith('fs') )
        except ValueError:
            logging.info(f"tag with missing attributes (containing '{tag.contents}'")
            return [0] * 6

        resolution = 50
        hxys = [
                self.get_declaration_value(css_dict[fft], 'line-height'),
                self.get_declaration_value(css_dict[fst], 'font-size'),

                self.get_declaration_value(css_dict[ht], 'height'),
                self.get_declaration_value(css_dict[xt], 'left'),
                self.get_declaration_value(css_dict[yt], 'bottom')
                ]

        dist = numpy.sqrt(
            (hxys[3]-self.point_before[0])**2 \
               + (hxys[4]-self.point_before[1])**2 )
        self.point_before = (hxys[3], hxys[4])
        dist  = int(dist/resolution) * resolution
        hxys = hxys + [dist]
        hxys [3] = int(hxys[3]/resolution) * resolution
        return hxys

    def add_text_coverage(self, soup):
        z_style = "\nz {background: rgba(0, 0, 0, 1) !important;   font-weight: bold;} "
        soup.head.append(soup.new_tag('style', type='text/css'))
        soup.head.style.append(z_style)

    def point_density_frequence(self, points2D, debug=False):
        # Extract x and y
        x = points2D[:, 0]
        y = points2D[:, 1]  # Define the borders
        deltaX = (max(x) - min(x)) / 10
        deltaY = (max(y) - min(y)) / 10
        xmin = min(x) - deltaX
        xmax = max(x) + deltaX
        ymin = min(y) - deltaY
        ymax = max(y) + deltaY
        print(xmin, xmax, ymin, ymax)  # Create meshgrid
        xx, yy = numpy.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = numpy.vstack([xx.ravel(), yy.ravel()])
        values = numpy.vstack([x, y])
        kernel = stats.gaussian_kde(values)
        f = numpy.reshape(kernel(positions).T, xx.shape)

        if debug:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.gca()
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
            ax.imshow(numpy.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
            cset = ax.contour(xx, yy, f, colors='k')
            ax.clabel(cset, inline=1, fontsize=10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.title('2D Gaussian Kernel density estimation')

        return kernel.evaluate(points2D.T) * 10000000, f

    def number_of_columns(self, density2D):
        peaks, _ = find_peaks(density2D[:, 30], distance=1)
        return len(peaks)


if __name__ == '__main__':
    tfu = TrueFormatUpmarker()
    # pprint(tfu.convert_and_index(html_path='/home/stefan/cow/pdfetc2txt/docs/0013.html'))
    # tfu.save_doc_json(json_path='/home/stefan/cow/pdfetc2txt/docs/0013.json')

    # tfu.convert_and_index(html_path='/home/stefan/cow/pdfetc2txt/docs/what is string theory.html')
    # pprint(tfu.get_indexed_words())

    docs = [{
        'html_path_before': '/home/stefan/cow/pdfetc2txt/docs/Laia Font-Ribera - Short-Term Changes in Respiratory Biomarkers after Swimmingin a Chlorinated Pool.pdf.html',
        'html_path_after' : '/home/stefan/cow/pdfetc2txt/docs/Laia Font-Ribera - Short-Term Changes in Respiratory Biomarkers after Swimmingin a Chlorinated Pool.pdf.pdf2htmlEX.af.html'
        },
        {
            'html_path_before': '/home/stefan/cow/pdfetc2txt/docs/F. Ning - Toward automatic phenotyping of developing embryos from videos.pdf.html',
            'html_path_after': '/home/stefan/cow/pdfetc2txt/docs/F. Ning - Toward automatic phenotyping of developing embryos from videos.pdf.pdf2htmlEX.af.html'
        },
        {
            'html_path_before': '/home/stefan/cow/pdfetc2txt/docs/HumKno.pdf.html',
            'html_path_after': '/home/stefan/cow/pdfetc2txt/docs/HumKno.pdf.pdf2htmlEX.af.html'
        }

    ]

    for kwargs in docs:
        tfu.convert_and_index(**kwargs)
        pprint(tfu.get_indexed_words())
