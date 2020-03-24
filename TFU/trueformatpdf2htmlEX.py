import json
import os
from collections import OrderedDict, Counter, defaultdict
import itertools
from statistics import stdev

import more_itertools
import numpy
import pandas
import regex
import scipy
from fastkde import fastKDE
from more_itertools.recipes import flatten
from numpy import mean
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import pprint

from scipy.stats import norm

import config
from Exceptions.ConversionException import EmptyPageConversionError
from TFU.trueformatupmarker import TrueFormatUpmarker
from helpers.color_logger import *
import bs4
import tinycss
from bs4 import Tag
from tinycss.css21 import RuleSet
import hdbscan

from helpers.programming import overrides
from helpers.str_tools import insert_at_index


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

class TrueFormatUpmarkerPdf2HTMLEX (TrueFormatUpmarker):
    def generate_css_tagging_document(self, html_before_path, html_after_path, debug_folder):
        """
        This manipulates an html-file from the result of pdf2htmlEX, that inserts word for word tags with css ids
        to apply markup to these words only with a css-file, without changing this document again.

        This has to restore the semantics of the layout, as the reading sequence of left right, top to bottom,
        column for column, page for page. Other layout things should disappear from the text sequence.
        """
        with open(html_before_path, 'r', encoding='utf8') as f:
            soup = bs4.BeautifulSoup(f.read(), features='lxml')

        # create data and features for clustering
        css_dict = self.get_css(soup)
        all_divs, coords, data, density_field = self.extract_features(soup=soup, css_dict=css_dict)

        if not self.parameterize:
            # e - 0.50, a - 0.69, cs - 188, s - 38
            self.generate_css_tagging_document_(
                all_divs=all_divs,
                coords=coords,
                data=data,
                density_field=density_field,
                html_before_path=html_before_path,
                parametrerized_file_in_folder=html_after_path,
                algorithm='boruvka_balltree',
                metric='hamming',
                epsilon=0.5,
                alpha=0.69,
                cluster_size=188,
                samples=38
            )
        else:
            range_cluster_selection_epsilon = self.create_eval_range(0.455, sigma=0.1, resolution=0.33)
            range_min_cluster_size = self.create_eval_range(210, sigma=0.3, resolution=0.33)
            range_min_samples = self.create_eval_range(36, sigma=0.3, resolution=0.33)
            range_alpha = self.create_eval_range(0.99, sigma=0.3, resolution=0.33)

            for epsilon in range_cluster_selection_epsilon:
                for cluster_size in range_min_cluster_size:
                    for samples in range_min_samples:
                        for alpha in range_alpha:
                            for algorithm in ['boruvka_balltree']:  # , 'boruvka_kdtree', 'generic', 'prims']:
                                for metric in [
                                    'hamming']:  # , 'l1', 'l2', 'hamming', 'infinity', 'p', 'chebyshev', 'cityblock', 'braycurtis', 'euclidean']:

                                    file_in_folder = insert_at_index(html_after_path,
                                                                     html_after_path.rfind("/"),
                                                                     debug_folder)
                                    parametrerized_file_in_folder = insert_at_index(
                                        file_in_folder,
                                        file_in_folder.rfind("/") + 1,
                                        f"e-{epsilon:.2f},a-{alpha:.2f},"
                                        f"cs-{int(cluster_size)},s-{int(samples)},"
                                        f"m-{metric},a-{algorithm}")

                                    # Following function is so overbusted with parameters to test all features and to precompute some data for experimenting
                                    self.generate_css_tagging_document_(# input, output destination to write file in the same form
                                                                        html_before_path=html_before_path,
                                                                        parametrerized_file_in_folder=parametrerized_file_in_folder,

                                                                        # precomputed data as input
                                                                        all_divs=all_divs,
                                                                        data=data,
                                                                        coords=coords,
                                                                        density_field=density_field,

                                                                        # clustering parameters
                                                                        algorithm=algorithm,
                                                                        alpha=alpha,
                                                                        cluster_size=cluster_size,
                                                                        epsilon=epsilon,
                                                                        metric=metric,
                                                                        samples = samples
                                                                        )

    def generate_css_tagging_document_(self, algorithm, all_divs, alpha, cluster_size, coords, data, density_field,
                                       epsilon, html_before_path, metric, parametrerized_file_in_folder, samples):
        sort_indices_divs_in_cols_lefrig_botup, indices_clusters_dict = self.fmr_hdbscan(
            [
                Page_Features.left,
                Page_Features.density,
                Page_Features.line_height
            ],
            all_divs, coords, data, density_field,
            metric=metric,
            algorithm=algorithm,
            cluster_selection_method="leaf",
            debug_pic_name=parametrerized_file_in_folder,
            cluster_selection_epsilon=float(epsilon),  # ,
            allow_single_cluster=False,
            min_cluster_size=int(cluster_size),  # 100,
            min_samples=int(samples),  # 45,
            alpha=alpha  # 0.99
        )

        with open(html_before_path, 'r', encoding='utf8') as f:
            soup = bs4.BeautifulSoup(f.read(), features='lxml')
        self.manipulate_document(soup=soup,
                                 sorting=sort_indices_divs_in_cols_lefrig_botup,
                                 clusters_dict=indices_clusters_dict,
                                 )

        # change '<' and '>' mail adress of pdf2htmlEX-author, because js thinks, it's a tag
        with open(parametrerized_file_in_folder, "w",
                  encoding='utf8') as file:
            file.write(str(soup).replace("<coolwanglu@gmail.com>", "coolwanglu@gmail.com"))


    def fmr_pages(self, soup):
        return soup.select("div[data-page-no]")


    def fmr_hdbscan(self,
                    features_to_use,
                    all_divs, coords, data, density_field,
                    debug_pic_name="/debug_pics/output.png",
                    debug=False,
                    **kwargs):
        """
        hdbscan on font height, x position and y position to recognize all groups of textboxes in different parts of
        the layout as footnotes, text columns, headers etc.
        """

        # Clustering
        clusterer = hdbscan.HDBSCAN(**kwargs)
        clusterer.fit(data[:, features_to_use])
        threshold = pandas.Series(clusterer.outlier_scores_).quantile(0.85)
        outliers = numpy.where(clusterer.outlier_scores_ > threshold)[0]

        # Determine number of clusters
        number_columns = self.number_of_columns(density2D=density_field)
        self.number_columns = number_columns
        logging.info(f"Detection of {number_columns} columns")
        what_clusters = set(clusterer.labels_)
        if number_columns == len(what_clusters):
            self.take_outliers = True
        elif number_columns  < len(what_clusters) * 0.33:
            logging.warning("found unrealistic number of clusters, so I just take all")
            self.take_outliers = True
            clusterer.labels_ = [0]* len(clusterer.labels_)
        else:
            self.take_outliers = False

        cluster_counts = Counter([l for l in clusterer.labels_ if self.take_outliers or l > -1])
        relevant_clusters = sorted(cluster_counts, key=cluster_counts.get)[-number_columns:]

        logging.info(f"Which clusters are there? {what_clusters}")
        logging.info(f"Number of relevant columns {number_columns}")
        logging.info(f"These are {relevant_clusters}")
        logging.info(f"How many content items in columns {cluster_counts}")
        logging.info(f"Unsing outliers also for column, {str(self.take_outliers).lower()}")


        if debug:
            logging.info(f"sorting and detecting textboxes with \n{pprint.pformat(kwargs)}")
            self.debug_pic(clusterer, coords, debug_pic_name, outliers)

            if len(what_clusters) < number_columns:
                logging.error("Too few columns found")
                raise ArithmeticError

        # Ordering pages and columns and divs
        # page_groups[pages]        = {columns: divs]
        #                             left to right sorting of columns and
        #                             top-down of textboxes
        #                             (not all divs together!)

        # collecting divs on the page (and append clustering label and index to the features)
        groups_of_pages = defaultdict(list)
        for feature_line in zip(*list(data.T), clusterer.labels_, range(len(clusterer.labels_))):
            groups_of_pages[feature_line[Page_Features.page_number]].append(
                feature_line)

        # sorting groups of clusters within pages
        page_cluster_lr_groups = defaultdict(list)
        for page, page_group in groups_of_pages.items():
            # itertools would break up the clusters, when the clusters are unsorted
            # TODO right to left and up to down!

            page_group = sorted(page_group,
                                key=lambda r: r[Page_Features.label_])
            groups_of_clusters = itertools.groupby(page_group,
                                                   key=lambda s: s[Page_Features.label_])
            page_cluster_up_bottom_groups = \
                [(cluster, sorted(cluster_group,
                                  key=lambda r: r[Page_Features.bottom]))
                 for cluster, cluster_group in groups_of_clusters
                 if cluster in relevant_clusters or self.take_outliers
                 ]
            page_cluster_lr_groups[page] = \
                sorted(page_cluster_up_bottom_groups, key=lambda r:
                min(r[1], key=lambda s: s[Page_Features.left]))

        clusters_dict = {features[Page_Features.index_]: cluster_label
                         for page, clusters in page_cluster_lr_groups.items()
                         for cluster_label, cluster in clusters
                         for features in cluster}

        div_reading_sequence = [features[Page_Features.index_]
                                for page, clusters in page_cluster_lr_groups.items()
                                for cluster_label, cluster in clusters
                                for features in cluster]

        return div_reading_sequence, clusters_dict

    def extract_features(self, soup, css_dict):
        # Collect divs (that they have an x... attribute, that is generated by pdf2htmlEX)
        pages_list = list(self.fmr_pages(soup))
        page2divs = self.collect_pages_dict(pages_list)
        all_divs = self.collect_all_divs(soup)

        # Generate positional features
        hs_xs_ys = [list(self.getxyh(tag, css_dict)) for tag in all_divs]
        text_prob = [[
            2 + page_number,
            len(div.text),
            0]  # ks_2samp(NORMAL_TEXT, list(div.text.lower())).pvalue *100 if div.text else 0]
            for page_number, divs in enumerate(page2divs) for div in divs]
        assert (len(hs_xs_ys) == len(text_prob))
        data = [list(hxy) + tp for hxy, tp in zip(hs_xs_ys, text_prob)]
        # Filter features based on margins
        data = [pf
                if pf[Page_Features.bottom] > self.min_bottom
                   and pf[Page_Features.bottom] < self.max_bottom
                else tuple([0] * 9)
                for pf in data
                ]
        # Density feature from positional features
        data = numpy.array(data)
        try:
            coords = data.T[[Page_Features.left, Page_Features.bottom]]
            page_coords = data.T[[Page_Features.page_number, Page_Features.left, Page_Features.bottom]].T
            page_2_coords  = numpy.split(page_coords[:, 1:], numpy.cumsum(numpy.unique(page_coords[:, 0], return_counts=True)[1])[:-1])
        except IndexError:
            raise EmptyPageConversionError
        densities_at_points, density_field, margin_mask = self.point_density_frequence_per_page(page_2_coords, debug=True)

        data = numpy.column_stack((data, densities_at_points))
        #return all_divs[margin_mask], coords[margin_mask], data[margin_mask], density_field
        return all_divs, coords, data, density_field

    def collect_pages_dict(self, pages):
        page2divs = [page.select('div[class*=x]') for page in pages]
        return page2divs

    def get_pdf2htmlEX_header(tag):
        return tag.attrs['class'][3]

    def debug_pic(self, clusterer, coords, debug_pic_name, outliers):
        color_palette = sns.color_palette('deep', 20)
        cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in clusterer.labels_]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                 zip(cluster_colors, clusterer.probabilities_)]
        plt.scatter(*list(coords), c=cluster_member_colors, linewidth=0)
        # plt.scatter(*list(coords[:,outliers].T), linewidth=0, c='red')
        plt.savefig(debug_pic_name + ".png", bbox_inches='tight')

    def get_declaration_value(self, declaration, key):
        try:
            return [decla.value[0].value for decla in declaration if decla.name == key][0]
        except:
            logging.error(f"{key} not in {str(declaration)}")
            return 0

    def seq_split(lst, cond):
        sublst = []
        start = 0
        for i, item in enumerate(lst):
            if not cond(item  # logging.info("excluding non-main-text" + str(text_div.contents)[:36])
                        ):
                sublst.append(item)
            else:
                yield start, sublst
                sublst = []
                start = i + 1
        if sublst:
            yield start, sublst

    def get_css_decla_for_tag(self, div, css_dict, css_class, key):
        if isinstance(div, Tag):
            try:
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


    point_before = (0, 0)

    def getxyh(self, tag, css_dict):
        try:
            fft, fst, ht, xt, yt = sorted(attr for attr in tag.attrs['class']
                                          if attr.startswith('x') or
                                          attr.startswith('y') or
                                          attr.startswith('h') or
                                          attr.startswith('ff') or
                                          attr.startswith('fs'))
        except ValueError:
            logging.info(f"Tag with missing attributes (containing '{tag.contents}'")
            return [0] * 6

        resolution = 1
        hxys = [
            self.get_declaration_value(css_dict[fft], 'line-height'),
            self.get_declaration_value(css_dict[fst], 'font-size'),

            self.get_declaration_value(css_dict[ht], 'height'),
            self.get_declaration_value(css_dict[xt], 'left'),
            self.get_declaration_value(css_dict[yt], 'bottom')
        ]

        dist = numpy.sqrt(
            (hxys[3] - self.point_before[0]) ** 2 \
            + (hxys[4] - self.point_before[1]) ** 2)
        self.point_before = (hxys[3], hxys[4])
        dist = int(dist / resolution) * resolution
        hxys = hxys + [dist]
        hxys[3] = int(hxys[3] / resolution) * resolution
        return hxys

    def point_density_frequence_per_page (self, pages_2_points, **kwargs):
        coords_densities, fields, margin_masks = list(zip(*[self.point_density_frequence(points2D=points2D, **kwargs) for points2D in pages_2_points]))
        return numpy.hstack(coords_densities), sum(f for i, f in enumerate(fields) if margin_masks[i])/len(margin_masks), numpy.hstack(margin_masks)

    edges = numpy.array(
        [[0, 0], [0, config.reader_height], [config.reader_width, 0], [config.reader_width, config.reader_height]])

    def normalized(self, a, axis=-1, order=-2):
        l2 =numpy.atleast_1d(numpy.linalg.norm(a, order, axis))
        l2[l2==0] = 1
        return a/numpy.expand_dims(l2, axis)

    def point_density_frequence(self, points2D, debug=True):

        edges_and_points = numpy.vstack((points2D, self.edges))
        edges_and_points = self.normalized(edges_and_points, axis=0, order=100)
        kde = fastKDE.fastKDE(edges_and_points.T, beVerbose=True)
        f = kde.pdf
        f = f * 1 / sum(sum(f)) * 10000  # normalize f, that it's 1 in sum
        indices = f.shape * edges_and_points[:-4]
        indices = indices.astype(int)
        pdfs = f[indices]
        # project

        #if debug:
        #    plt.contour(v1, v2, f)
        #    plt.show()

        margin_mask = self.hv_border(points2D.T)

        return pdfs, f, margin_mask

    def number_of_columns(self, density2D):
        peaks_at_height_steps = []
        for height in range(
                int(config.page_array_model * 0.1),
                int(config.page_array_model * 0.9),
                int(config.page_array_model * 0.05)):
            peaks, _ = find_peaks(density2D[:, height], distance=2, prominence=0.1)
            peaks_at_height_steps.append(peaks)
        lens = [len(peaks) for peaks in peaks_at_height_steps]
        counts = Counter(lens)
        return max(counts, key=counts.get)

    def create_eval_range(self, number, sigma=0.5, resolution=0.2):
        start = number * (1 - sigma)
        end = number * (1 + sigma)
        step = (end - start) * resolution
        return numpy.arange(start, end, step)

    def collect_all_divs(self, soup):
        return soup.select('div[class*=x]')

    def hv_border(self, points2d):
        max_coord = points2d[1, :].max()
        xrow = numpy.array([0] + points2d[1, :].tolist() + [max_coord + 10])

        dx = list(zip(*list((a, b-a) for (a,b) in more_itertools.pairwise(sorted(xrow)))))
        fun = scipy.interpolate.interp1d(*dx)
        minx = points2d[1, :].min()
        maxx = points2d[1, :].max()
        n_values = complex(config.page_array_model)
        try:
            y = fun(numpy.mgrid[minx : maxx : n_values])
            h_peaks, _ = find_peaks(y, distance=25, prominence=0.0000009)
            plt.hist(y, bins=len(y), normed=True)
        except ValueError:
            logging.error("out of interpolation range")

        if len(h_peaks) == 1:
            return numpy.full_like(points2d, True)

        if len(h_peaks) > 1:
            logging.info("found some page with header or footnote?")

import unittest


class TestPaperReader(unittest.TestCase):
    tfu_pdf = TrueFormatUpmarkerPdf2HTMLEX(debug=True, parameterize=False)

    def test_columns_and_file_existence(self):
        docs = [
            {'html_path_before': 'Ludwig Wittgenstein - Tractatus-Logico-Philosophicus.pdf.html',
             'html_path_after': 'Ludwig Wittgenstein - Tractatus-Logico-Philosophicus.pdf.html.pdf2htmlEX.test.html',
             'cols': 1
             },
            {
                'html_path_before': 'Wei Quian - Translating Embeddings for Knowledge Graph Completion with Relation Attention Mechanism.pdf.html',
                'html_path_after': 'Wei Quian - Translating Embeddings for Knowledge Graph Completion with Relation Attention Mechanism.test.html',
                'cols': 2
            },
            {
                'html_path_before': 'Sonja Vermeulen - Climate Change and Food Systems.pdf.html',
                'html_path_after': 'Sonja Vermeulen - Climate Change and Food Systems.pdf.html.pdf2htmlEX.test.html',
                'cols': 2
            },

            {
                'html_path_before': 'Filipe Mesquita - KnowledgeNet: A Benchmark Dataset for Knowledge Base Population.pdf.html',
                'html_path_after': 'Filipe Mesquita - KnowledgeNet: A Benchmark Dataset for Knowledge Base Population.pdf.pdf2htmlEX.test.html',
                'cols': 2
            },
            {
                'html_path_before': 'Laia Font-Ribera - Short-Term Changes in Respiratory Biomarkers after Swimmingin a Chlorinated Pool.pdf.html',
                'html_path_after': 'Laia Font-Ribera - Short-Term Changes in Respiratory Biomarkers after Swimmingin a Chlorinated Pool.pdf.pdf2htmlEX.test.html',
                'cols': 3
            },
            {
                'html_path_before': 'F. Ning - Toward automatic phenotyping of developing embryos from videos.pdf.html',
                'html_path_after': 'F. Ning - Toward automatic phenotyping of developing embryos from videos.pdf.pdf2htmlEX.test.html',
                'cols': 2
            },
            {
                'html_path_before': 'HumKno.pdf.html',
                'html_path_after': 'HumKno.pdf.pdf2htmlEX.test.html',
                'cols': 1
            }
        ]

        for kwargs in docs:
            columns = kwargs['cols']
            del kwargs['cols']
            kwargs['html_path_before'] = config.appcorpuscook_docs_document_dir + kwargs['html_path_before']
            kwargs['html_path_after'] = config.appcorpuscook_docs_document_dir + kwargs['html_path_after']
            self.tfu_pdf.convert_and_index(**kwargs)
            assert self.tfu_pdf.number_columns == columns
            assert self.tfu_pdf.indexed_words
            assert os.path.exists(kwargs['html_path_after'])

if __name__ == '__main__':
    unittest.main()