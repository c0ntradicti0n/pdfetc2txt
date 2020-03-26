import os
import pathlib
import bs4

from collections import Counter, defaultdict, namedtuple
import more_itertools
import itertools
from typing import List, Dict

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
from TFU.trueformatupmarker import TrueFormatUpmarker
from helpers.color_logger import *
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
    def generate_css_tagging_document(self, html_before_path, html_after_path):
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
        features = self.extract_features(soup=soup, css_dict=css_dict)

        hdbscan_kwargs = {
            'algorithm' : 'boruvka_balltree',
            'metric' : 'hamming',
            'cluster_selection_epsilon' :  0.5,
            'cluster_selection_method' : 'leaf',
            'alpha' :  0.95,
            'min_cluster_size' :  200,
            'min_samples' :  50
        }

        reading_sequence_sorted_and_indexed_divs, indices_to_clusters = \
            self.HDBSCAN_cluster_divs(
                features=features,
                features_to_use=[
                    Page_Features.left,
                    Page_Features.density,
                    Page_Features.font_size,
                ],
                debug_path=html_after_path,
                hdbscan_kwargs=hdbscan_kwargs
        )

        self.manipulate_document(soup=soup,
                                 divs=reading_sequence_sorted_and_indexed_divs,
                                 clusters_dict=indices_to_clusters,
                                 )

        # sanitizing
        # change '<' and '>' mail adress of pdf2htmlEX-author, because js thinks, it's a tag
        with open(html_after_path, "w",
                  encoding='utf8') as file:
            file.write(str(soup).replace("<coolwanglu@gmail.com>", "coolwanglu@gmail.com"))

    def fmr_pages(self, soup):
        return soup.select("div[data-page-no]")

    SortedClusteredDivs = namedtuple("SortedClusteredDivs", ["reading_sequence", "index_to_clusters"])
    def HDBSCAN_cluster_divs(self,
                             features_to_use:List[int],
                             features,
                             debug_path: str ="/debug_pics/output.png",
                             debug: bool = True,
                             hdbscan_kwargs: Dict = {}) -> SortedClusteredDivs:
        """
        hdbscan on font height, x position and y position to recognize all groups of textboxes in different parts of
        the layout as footnotes, text columns, headers etc.
        """

        # Clustering
        clusterer = hdbscan.HDBSCAN(**hdbscan_kwargs)
        clusterer.fit(features.data[:, features_to_use])
        threshold = pandas.Series(clusterer.outlier_scores_).quantile(0.85)
        outliers = numpy.where(clusterer.outlier_scores_ > threshold)[0]

        # Determine number of clusters
        number_columns = self.number_of_columns(density2D=features.density_field.T)
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

        logging.warning("#### Take all debugging ####")
        self.take_outliers = True
        clusterer.labels_ = [0] * len(clusterer.labels_)

        cluster_counts = Counter([l for l in clusterer.labels_ if self.take_outliers or l > -1])
        relevant_clusters = sorted(cluster_counts, key=cluster_counts.get)[-number_columns:]

        logging.info(f"which clusters are there? {what_clusters}")
        logging.info(f"number of relevant columns {number_columns}")
        logging.info(f"these are {relevant_clusters}")
        logging.info(f"how many content items in columns {cluster_counts}")
        logging.info(f"using outliers also for column? {str(self.take_outliers).lower()}")

        if debug:
            logging.info(f"sorting and detecting textboxes with \n{pprint.pformat(hdbscan_kwargs)}")
            self.debug_pic(clusterer, features.coords, debug_path, outliers)

            if len(what_clusters) < number_columns:
                logging.error("Too few columns found")
                logging.warning("#### Take all debugging ####")
                self.take_outliers = True
                clusterer.labels_ = [0] * len(clusterer.labels_)

        # collecting divs on the page (and append clustering label and index to the features)
        groups_of_pages = defaultdict(list)
        for feature_line in zip(*list(features.data.T), clusterer.labels_, range(len(clusterer.labels_))):
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

        index_reading_sequence = [features[Page_Features.index_]
                                for page, clusters in page_cluster_lr_groups.items()
                                for cluster_label, cluster in clusters
                                for features in cluster]
        div_reading_sequence = [features.divs[i] for i in index_reading_sequence]

        return TrueFormatUpmarkerPdf2HTMLEX.SortedClusteredDivs(div_reading_sequence, clusters_dict)

    FeatureStuff = namedtuple("FeatureStuff", ["divs", "coords", "data", "density_field"])
    def extract_features(self, soup, css_dict) -> FeatureStuff:
        # Collect divs (that they have an x... attribute, that is generated by pdf2htmlEX)
        pages_list = list(self.fmr_pages(soup))
        page_to_divs = self.collect_pages_dict(pages_list)
        all_divs = self.collect_all_divs(soup)

        # Generate positional features
        hs_xs_ys = [list(self.getxyh(tag, css_dict)) for tag in all_divs]
        text_prob = [[
            2 + page_number,
            len(div.text),
            0]  # ks_2samp(NORMAL_TEXT, list(div.text.lower())).pvalue *100 if div.text else 0]
            for page_number, divs in enumerate(page_to_divs) for div in divs]
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
            page_to_coords = numpy.split(
                page_coords[:, 1:],
                numpy.cumsum(
                    numpy.unique(page_coords[:, 0],
                                 return_counts=True
                                 )[1]
                )[:-1]
            )
        except IndexError:
            raise EmptyPageConversionError

        densities_at_points, density_field, mask = self.point_density_frequence_per_page(page_to_coords, page_to_divs, debug=True)
        data = numpy.column_stack((data, densities_at_points))

        relevant_divs = [div for div, to_use in zip(all_divs, mask) if to_use]
        relevant_coords = coords[:, mask]
        relevant_data = data[mask]

        return TrueFormatUpmarkerPdf2HTMLEX.FeatureStuff(relevant_divs, relevant_coords, relevant_data, density_field)

    def collect_pages_dict(self, pages):
        page_to_divs = [page.select('div[class*=x]') for page in pages]
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
            #logging.info(f"Tag with missing attributes (containing '{tag.contents}'")
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

    def point_density_frequence_per_page (self, pages_to_points, pages_to_divs, **kwargs):
        # map computation to page
        coords_densities, fields, masks = \
            list(
                zip(*[
                    self.point_density_frequence(
                        points2d=points2d,
                        divs=pages_to_divs[page_number],
                        **kwargs)
                    for page_number, points2d in enumerate(pages_to_points)
                    ]
                    )
            )
        # filter, reduce
        return numpy.hstack(coords_densities), sum(f for i, f in enumerate(fields)), numpy.hstack(masks)

    edges = numpy.array(
        [[0, 0], [0, config.reader_height], [config.reader_width, 0], [config.reader_width, config.reader_height]])

    def normalized(self, a):
        return a/[a_line.max()-a_line.min() for a_line in a.T]

    def point_density_frequence(self, points2d, divs, debug=True, axe_len_X=100, axe_len_Y=100):
        edges_and_points = numpy.vstack((points2d, self.edges))
        edges_and_points = self.normalized(edges_and_points)

        indices = (edges_and_points[:-4] * [axe_len_X, axe_len_Y]).astype(int)

        dotted = numpy.zeros((100,100))
        dotted [indices[:,0], indices[:,1]] = 1

        coarse_grained_field =  gaussian_filter(dotted, sigma=3)
        coarse_grained_pdfs = coarse_grained_field [indices[:,0], indices[:,1]]

        fine_grained_field =  gaussian_filter(dotted, sigma=0.5)
        fine_grained_pdfs = fine_grained_field [indices[:,0], indices[:,1]]

        mask = self.header_footer_mask(fine_grained_field, fine_grained_pdfs, edges_and_points[:-4], divs)
        return coarse_grained_pdfs, coarse_grained_field, mask

    def number_of_columns(self, density2D):
        peaks_at_height_steps = []
        for height in range(
                int(config.page_array_model * 0.1),
                int(config.page_array_model * 0.9),
                int(config.page_array_model * 0.05)):
            peaks, _ = find_peaks(density2D[height], distance=20, prominence=0.05)
            peaks_at_height_steps.append(peaks)
        lens = [len(peaks) for peaks in peaks_at_height_steps]
        return self.most_common_value(lens)

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

    def header_footer_mask(self, field, pdfs, points, divs):
        mask = numpy.full_like(pdfs, True).astype(bool)

        if  len(pdfs) > 10:
            index_to_Y = sorted(enumerate(points[:,1]), key=lambda x:x[1])

            index_to_yD = {i1: numpy.abs(b - a) for (i1, a), (i2, b) in more_itertools.pairwise(index_to_Y) }
            dY = list(index_to_yD.values())
            indices = numpy.array(range(len(index_to_Y)))
            d0y = 0
            for i, dy in enumerate(dY):
                if dy < 0.001:
                    dY[i] = d0y
                else:
                    d0y = dy

            norm_distance = numpy.median(dY)
            distance_std = numpy.std([d for d in dY if abs(d) < norm_distance*2])

            invalid_points_at = numpy.logical_or(dY < norm_distance - distance_std,
                                               dY > norm_distance + distance_std)

            invalid_points_at = numpy.hstack((invalid_points_at,[False]))
            invalid_points_shift = numpy.hstack(([False], invalid_points_at[:-1]))

            bad_indices = indices[numpy.logical_or(invalid_points_at, invalid_points_shift)]
            mask[bad_indices] = False
            logging.debug(f"Header/Footer in: " +'\n'.join([str(divs[i]) for i in bad_indices]))
        return mask

import unittest

class TestPaperReader(unittest.TestCase):
    tfu_pdf = TrueFormatUpmarkerPdf2HTMLEX(debug=True, parameterize=False)

    def test_layout_files(self):
        files = list(pathlib.Path('test/data').glob('*.html'))
        for path in files:
            path = str(path)
            kwargs = {}
            kwargs['html_path_before'] = path
            kwargs['html_path_after']  = path + ".computed.htm"
            columns = int(regex.search(r"\d", path).group(0))
            self.tfu_pdf.convert_and_index(**kwargs)
            print(self.tfu_pdf.number_columns, columns)
            assert self.tfu_pdf.number_columns == columns
            assert self.tfu_pdf.indexed_words
            assert os.path.exists(kwargs['html_path_after'])


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