import os
import pathlib
import bs4

from collections import Counter, defaultdict, namedtuple
import more_itertools
import itertools
from typing import List, Dict, Tuple

import numpy
import pandas
import scipy
import hdbscan

from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import seaborn as sns
import pprint
import regex

import config
from Exceptions.ConversionException import EmptyPageConversionError
from TFU.pdf import Pdf
from TFU.trial_tools import range_parameter
from TFU.trueformatupmarker import TrueFormatUpmarker
from helpers.color_logger import *
from helpers.list_tools import threewise, third_fractal


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

class TrueFormatUpmarkerPdf2HTMLEX (TrueFormatUpmarker):
    hdbscan_kwargs = {
        'algorithm': 'boruvka_balltree',
        'metric': 'hamming',
        'cluster_selection_epsilon': 0.5,
        'cluster_selection_method': 'leaf',
        'alpha': 0.95,
        'min_cluster_size': 0.7,
        'min_samples': 0.4
    }

    def generate_css_tagging_document(self, html_read_from="", html_write_to="", parameterizing=False):
        """
        This manipulates an html-file from the result of pdf2htmlEX, that inserts word for word tags with css ids
        to apply markup to these words only with a css-file, without changing this document again.

        This has to restore the semantics of the layout, as the reading sequence of left right, top to bottom,
        column for column, page for page. Other layout things should disappear from the text sequence.
        """

        with open(html_read_from, 'r', encoding='utf8') as f:
            soup = bs4.BeautifulSoup(f.read(), features='lxml')

        # create data and features for clustering
        css_dict = self.get_css(soup)
        features = self.extract_features(soup=soup, css_dict=css_dict)

        hdbscan_kwargs = {
            'algorithm': 'boruvka_balltree',
            'metric': 'hamming',
            'cluster_selection_epsilon': 0.5,
            'cluster_selection_method': 'leaf',
            'alpha': 0.95,
            'min_cluster_size': int((len(features.divs) * 0.7 / self.number_columns)),
            'min_samples': int((len(features.divs) * TrueFormatUpmarkerPdf2HTMLEX.hdbscan_kwargs["min_samples"] / self.number_columns))
        }

        self.pdf_obj.columns = self.number_columns


        self.HDBSCAN_cluster_divs(
                features=features,
                features_to_use=["x", "font-size"],
                debug_path=html_write_to,
                hdbscan_kwargs=hdbscan_kwargs
                )

        self.manipulate_document(soup=soup,
                                 features=features)

        # sanitizing
        # change '<' and '>' mail adress of pdf2htmlEX-author, because js thinks, it's a tag
        with open(html_write_to, "w",
                  encoding='utf8') as file:
            file.write(str(soup).replace("<coolwanglu@gmail.com>", "coolwanglu@gmail.com"))

        self.pdf_obj.text = " ".join(self.indexed_words.values())
        self.pdf_obj.indexed_words = self.indexed_words



    def get_page_tags(self, soup):
        return soup.select("div[data-page-no]")

    SortedClusteredDivs = namedtuple("SortedClusteredDivs", ["reading_sequence", "index_to_clusters"])
    def HDBSCAN_cluster_divs(self,
                             features_to_use: List[str],
                             features: pandas.DataFrame,
                             debug_path: str ="/debug_pics/output.png",
                             debug: bool = True,
                             hdbscan_kwargs: Dict = {}) -> SortedClusteredDivs:
        """
        hdbscan on font height, x position and y position to recognize all groups of textboxes in different parts of
        the layout as footnotes, text columns, headers etc.
        """

        # Clustering
        clusterer = hdbscan.HDBSCAN(**hdbscan_kwargs)
        clusterer.fit(features[features_to_use])
        threshold = pandas.Series(clusterer.outlier_scores_).quantile(0.85)
        outliers = numpy.where(clusterer.outlier_scores_ > threshold)[0]

        # Determine number of clusters

        logging.debug(f"detected {self.number_columns} columns")
        what_clusters = set(clusterer.labels_)

        if self.number_columns == len(what_clusters):
            self.take_outliers = True
        elif self.number_columns  < len(what_clusters) * 0.33:
            logging.error("found unrealistic number of clusters, so I just take all")
            self.take_outliers = True
            clusterer.labels_ = [0]* len(clusterer.labels_)
        else:
            self.take_outliers = False

        cluster_counts = Counter([l for l in clusterer.labels_ if self.take_outliers or l > -1])
        relevant_clusters = sorted(cluster_counts, key=cluster_counts.get)[-self.number_columns:]

        logging.debug(f"which clusters are there? {what_clusters}")
        logging.debug(f"number of relevant columns {self.number_columns}")
        logging.debug(f"these are {relevant_clusters}")
        logging.debug(f"how many content items in columns {cluster_counts}")
        logging.debug(f"using outliers also for column? {str(self.take_outliers).lower()}")

        if len(relevant_clusters)==len(cluster_counts):
            logging.warning("too many outliers, take all")
            self.take_outliers = True

        """if debug:
            logging.info(f"sorting and detecting textboxes with \n{pprint.pformat(hdbscan_kwargs)}")
            self.debug_pic(clusterer, numpy.column_stack((features.x, features.y)), debug_path, outliers)

            if len(what_clusters) < self.number_columns:
                logging.error("Too few columns found")
                logging.warning("#### Take all debugging ####")
                self.take_outliers = True
                clusterer.labels_ = [0] * len(clusterer.labels_)"""

        features["cluster"] = clusterer.labels_

        # sorting groups of clusters within pages
        page_cluster_lr_groups = defaultdict(list)
        for page, page_group in features.groupby(by="page_number"):
            # itertools would break up the clusters, when the clusters are unsorted
            # TODO right to left and up to down!

            # Assuming, also column labels have been sorted yet from left to right
            groups_of_clusters = page_group.groupby(by="cluster")

            page_cluster_up_bottom_groups = \
                [(cluster, cluster_group.sort_values(by="y"))
                 for cluster, cluster_group in groups_of_clusters
                 ]

            page_cluster_lr_groups[page] = \
                sorted(page_cluster_up_bottom_groups, key=lambda c: c[1].x.min())

        reading_i = itertools.count()  # counter for next indices for new html-tags

        features["reading_sequence"] = list(more_itertools.flatten([cluster_content["index"].tolist()
                                for page_number, page_content in page_cluster_lr_groups.items()
                                for cluster, cluster_content in page_content
                                ]))

        if not self.take_outliers:
            features["relevant"] = features.cluster.isin(relevant_clusters)
        else:
            features["relevant"] = True


        self.pdf_obj.pages_to_column_to_text = \
            features.groupby(by="cluster").apply(lambda x:
                x.groupby(by="page_number").apply(lambda x:
                      " ".join([div.text for div in x.divs]))).to_dict()
        return features

    FeatureStuff = namedtuple("FeatureStuff", ["divs", "coords", "data", "density_field", "labels"])
    def extract_features(self, soup, css_dict) -> FeatureStuff:
        features = pandas.DataFrame()

        # Collect divs (that they have an x... attribute, that is generated by pdf2htmlEX)
        page_tags = list(self.get_page_tags(soup))
        page_to_divs = self.collect_pages_dict(page_tags)

        i = itertools.count()
        features["index"], features["page_number"], features["divs"] = zip(*[(next(i), pn, div) for pn, divs in page_to_divs for div in divs])
        features["len"] = features.divs.apply(lambda div:len(div.text))

        # Generate positional features
        self.set_css_attributes(features, css_dict)
        self.point_density_frequence_per_page(features, debug=True)
        return features[features.relevance_mask]

    def collect_pages_dict(self, pages):
        page_to_divs = [(page_number, page.select('div[class*=x]')) for page_number, page in enumerate(pages)]
        return page_to_divs

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

    point_before = (0, 0)

    assinging_features = ['line-height', 'font-size', 'height', 'left',  'bottom']
    def get_tag_attribute_names(self, tag):
        return sorted(attr for attr in tag.attrs['class']
                                          if attr.startswith('x') or
                                          attr.startswith('y') or
                                          attr.startswith('h') or
                                          attr.startswith('ff') or
                                          attr.startswith('fs'))

    def set_css_attributes(self, features, css_dict):
        features["tag_attributes"]  = features.divs.apply(self.get_tag_attribute_names)
        for index, attribute in enumerate(self.assinging_features):
            features[attribute] = features.tag_attributes.apply(lambda x: self.get_declaration_value(css_dict[x[index]], attribute))
        features.rename(columns={"bottom":"y", "left":"x"}, inplace=True)

    def point_density_frequence_per_page (self, features, **kwargs):
        # map computation to page
        page_groups = features.groupby(by="page_number").apply(
            lambda page_group: self.analyse_point_density_frequence(
                        page_group,
                        **kwargs)
             ).tolist()
        other_feature_kinds_stacked = self.FeatureKinds(*list(zip(*page_groups)))

        self.number_columns = self.most_common_value(other_feature_kinds_stacked.number_of_columns)

        features["coarse_grained_pdf"] = numpy.hstack(other_feature_kinds_stacked.coarse_grained_pdfs)
        coarse_grained_field = sum(f for i, f in enumerate(other_feature_kinds_stacked.coarse_grained_field))
        features["relevance_mask"] = numpy.hstack(other_feature_kinds_stacked.mask)

        return coarse_grained_field

    edges = numpy.array(
        [[0, 0], [0, config.reader_height], [config.reader_width, 0], [config.reader_width, config.reader_height]])

    def normalized(self, a):
        return a/[a_line.max()-a_line.min() for a_line in a.T]

    FeatureKinds = namedtuple("FeatureKinds", ["coarse_grained_pdfs", "coarse_grained_field", "mask", "number_of_columns"])

    def analyse_point_density_frequence(self, page_features, debug=True, axe_len_X=100, axe_len_Y=100) -> FeatureKinds:
        points2d = numpy.column_stack((page_features.x, page_features.y))
        edges_and_points = numpy.vstack((points2d, self.edges))
        edges_and_points = self.normalized(edges_and_points)

        indices = (edges_and_points[:-4] * [axe_len_X, axe_len_Y]).astype(int)

        dotted = numpy.zeros((100,100))
        dotted [indices[:,0], indices[:,1]] = 1

        coarse_grained_field =  gaussian_filter(dotted, sigma=3)
        coarse_grained_pdfs = coarse_grained_field [indices[:,0], indices[:,1]]

        fine_grained_field =  gaussian_filter(dotted, sigma=0.5)
        fine_grained_pdfs = fine_grained_field [indices[:,0], indices[:,1]]

        number_of_columns = self.number_of_columns(density2D=fine_grained_field.T)

        mask = self.header_footer_mask(
            fine_grained_field,
            fine_grained_pdfs,
            edges_and_points[:-4],
            number_of_columns,
            page_features.divs)
        return self.FeatureKinds(coarse_grained_pdfs, coarse_grained_field, mask, number_of_columns)

    def most_common_value(self, values, constraint=None):
        if constraint:
            test_values = [v for v in values if constraint(v)]
        else:
            test_values =  values
        counts = Counter(test_values)
        return max(counts, key=counts.get)

    def create_eval_range(self, number, sigma=0.5, resolution=0.2):
        start = number * (1 - sigma)
        end = number * (1 + sigma)
        step = (end - start) * resolution
        return numpy.arange(start, end, step)

    def collect_all_divs(self, soup):
        return soup.select('div[class*=x]')

    def number_of_columns(self, density2D):
        peaks_at_height_steps = []
        for height in range(
                int(config.page_array_model * 0.1),
                int(config.page_array_model * 0.9),
                int(config.page_array_model * 0.05)):
            peaks, _ = find_peaks(density2D[height], distance=15, prominence=0.0001)
            peaks_at_height_steps.append(peaks)
        lens = [len(peaks) for peaks in peaks_at_height_steps]
        return self.most_common_value(lens)

    def header_footer_mask(self, field, pdfs, points, number_of_culumns, divs):
        mask = numpy.full_like(pdfs, False).astype(bool)

        if  len(pdfs) > 10:
            indexed_points = list(enumerate(points))
            indices = numpy.array(range(len(points)))
            # raw column sorting
            x_sorted_points = [(int(indexed_point[1][0] / (0.7 / number_of_culumns)), indexed_point) for indexed_point in indexed_points]
            # raw top down sorting
            xy_sorted_points = sorted(x_sorted_points, key=lambda x:x[0] - x[1][1][1])

            y_sorted_indexed_points = [(len(points)+1,0)] + \
                                      [(column_index_point_tuple[1][0], column_index_point_tuple[1][1][1])
                                          for column_index_point_tuple
                                          in xy_sorted_points] + \
                                      [(len(points)+2,1)]

            indexed_distances = [(i2, (numpy.abs(b - a), numpy.abs(c - b)))
                                for (i1, a), (i2, b), (i3, c)
                                in threewise(y_sorted_indexed_points)]
            dY = list(id[1] for id in indexed_distances)
            dI = numpy.array(list(id[0] for id in indexed_distances))

            # If there are text boxes on the same height, the distance will be very small due to rounding,
            # replace them with the value for the textbox in the same line
            threshold = 0.00001
            d0y1 = 0
            d0y2 = 0
            for i, (dy1, dy2) in enumerate(dY):
                if dy1 < threshold:
                    dY[i] = (d0y1, dy2)
                else:
                    d0y1 = dy1
                if dy1 < threshold:
                    dY[i] = (dy1, d0y2)
                else:
                    d0y2 = dy2

            norm_distance = numpy.median(list(more_itertools.flatten(dY)))
            distance_std = norm_distance * 0.1
            logging.debug(f"median {norm_distance} std {distance_std}")

            valid_points_at = numpy.logical_and(dY > norm_distance - distance_std, dY < norm_distance + distance_std).any(axis=1)
            good = indices[dI[valid_points_at]]
            mask[good] = True
        return mask

import unittest


class Updater(object):
    params = {}
    def __init__(self, params : Dict[
                                 str, Tuple[float, float, float]]):
        for param_name, range_tuple in params.items():
            setattr(self, param_name, range_parameter(range_tuple))
            self.params[param_name] = range_tuple

    def next(self, str):
        value = getattr(self, str)
        return next(value)

    def update(self, choices):
        best = max(choices)
        for param, tuple in self.params.items():
            self.params[param] = third_fractal(choices[best][param], tuple)

    def trials(self):
        for param in self.params:
            yield (param, self.next(param))

    def options(self):
        for change in itertools.chain(self.trials()):
            yield change

class TestPaperReader(unittest.TestCase):
    tfu_pdf = TrueFormatUpmarkerPdf2HTMLEX(debug=True, parameterize=False)

    def test_a_train(self):
        updating = {"min_samples": (0.1, 0.5, 0.9)}
        updater = Updater(updating)
        files = list(pathlib.Path('test/data').glob('*.html'))
        for path in files:

            train = True


            i = 0


            while train:
                choices = {}
                i += 1
                if i>10:
                    logging.warning(f"reached {self.tfu_pdf.hdbscan_kwargs}")
                    break

                for option in updater.options():

                    logging.info(f"updating {dict([option])}")
                    self.tfu_pdf.hdbscan_kwargs.update(dict([option]))

                    path = str(path)
                    kwargs = {}
                    kwargs['html_read_from'] = path
                    kwargs['html_write_to'] = path + ".computed.htm"
                    columns = int(regex.search(r"\d", path).group(0))

                    pdf_obj = self.tfu_pdf.convert_and_index(**kwargs)

                    score = pdf_obj.verify(serious=True, test_document=True)
                    logging.info(f"PDF extraction score: {score}")

                    choices[score] = option

            updater.update(choices)

    def test_layout_files(self):
        updater = Updater({"min_samples": (0., 110., 220.)})
        files = list(pathlib.Path('test/data').glob('*.html'))
        for path in files:
            path = str(path)
            kwargs = {}
            kwargs['html_read_from'] = path
            kwargs['html_write_to']  = path + ".computed.htm"
            columns = int(regex.search(r"\d", path).group(0))

            pdf_obj = self.tfu_pdf.convert_and_index(**kwargs)

            score = pdf_obj.verify(serious=True, test_document=True)
            logging.info(f"PDF extraction score: {score}")

            assert pdf_obj.columns == columns
            assert os.path.exists(kwargs['html_write_to'])



    def xtest_columns_and_file_existence(self):
        docs = [
            {
                'html_path_before': 'Laia Font-Ribera - Short-Term Changes in Respiratory Biomarkers after Swimmingin a Chlorinated Pool.pdf.html',
                'html_path_after': 'Laia Font-Ribera - Short-Term Changes in Respiratory Biomarkers after Swimmingin a Chlorinated Pool.pdf.pdf2htmlEX.test.html',
                'cols': 3
            },
            {
                'html_path_before': 'Sonja Vermeulen - Climate Change and Food Systems.pdf.html',
                'html_path_after': 'Sonja Vermeulen - Climate Change and Food Systems.pdf.html.pdf2htmlEX.test.html',
                'cols': 2
            },
            {
                'html_path_before': 'Wei Quian - Translating Embeddings for Knowledge Graph Completion with Relation Attention Mechanism.pdf.html',
                'html_path_after': 'Wei Quian - Translating Embeddings for Knowledge Graph Completion with Relation Attention Mechanism.test.html',
                'cols': 2
            },
            {'html_path_before': 'Ludwig Wittgenstein - Tractatus-Logico-Philosophicus.pdf.html',
             'html_path_after': 'Ludwig Wittgenstein - Tractatus-Logico-Philosophicus.pdf.html.pdf2htmlEX.test.html',
             'cols': 1
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
            logging.error(kwargs)
            columns = kwargs['cols']
            del kwargs['cols']
            kwargs['html_path_before'] = config.appcorpuscook_docs_document_dir + kwargs['html_path_before']
            kwargs['html_path_after'] = config.appcorpuscook_docs_document_dir + kwargs['html_path_after']
            self.tfu_pdf.convert_and_index(**kwargs)
            print (self.tfu_pdf.number_columns, columns)
            assert self.tfu_pdf.number_columns == columns
            assert self.tfu_pdf.indexed_words
            assert os.path.exists(kwargs['html_path_after'])

if __name__ == '__main__':
    unittest.main()