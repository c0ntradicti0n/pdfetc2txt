import os
import pathlib
import bs4

from collections import Counter, defaultdict, namedtuple
import more_itertools
import itertools
from typing import List, Dict

import numpy
import pandas

from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import seaborn as sns
import regex

import config
from AutoUpdate.autoupdate import Updater
from TFU.trueformatupmarker import TrueFormatUpmarker
from helpers.color_logger import *
from helpers.list_tools import threewise
logging.getLogger().setLevel(logging.WARNING)

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920x1080")
chrome_options.binary_location = '/usr/bin/google-chrome'


class TrueFormatUpmarkerPdf2HTMLEX (TrueFormatUpmarker):
    browser = webdriver.Chrome(executable_path=os.path.abspath("venv/bin/chromedriver"),   chrome_options=chrome_options)

    def generate_css_tagging_document(self, html_read_from="", html_write_to="", parameterizing=False):
        """
        This manipulates an html-file from the result of pdf2htmlEX, that inserts word for word tags with css ids
        to apply markup to these words only with a css-file, without changing this document again.

        This has to restore the semantics of the layout, as the reading sequence of left right, top to bottom,
        column for column, page for page. Other layout things should disappear from the text sequence.
        """

        with open(html_read_from, 'r', encoding='utf8') as f:
            soup = bs4.BeautifulSoup(f.read(), features='lxml')


        web_view = self.browser.get("file:///home/stefan/cow/pdf2etc/"+ html_read_from)

        # create data and features for clustering
        css_dict = self.get_css(soup)
        features = self.extract_features(soup=soup, css_dict=css_dict, web_view=web_view)


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

        self.pdf_obj.features = features
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

        # sorting groups of clusters within pages
        page_cluster_lr_groups = defaultdict(list)
        for page, page_group in features.groupby(by="page_number"):
            # itertools would break up the clusters, when the clusters are unsorted
            # TODO right to left and up to down!

            # Assuming, also column labels have been sorted yet from left to right

            groups_of_clusters = page_group.groupby(by="column_labels")

            groups_of_clusters = sorted(groups_of_clusters, key=lambda cluster_and_cluster_group: cluster_and_cluster_group[1].x.mean() )

            page_cluster_up_bottom_groups = \
                [(new_cluster, cluster_group.sort_values(by="y"))
                 for new_cluster, (cluster, cluster_group) in enumerate(groups_of_clusters)
                 ]

            page_cluster_lr_groups[page] = \
                sorted(page_cluster_up_bottom_groups, key=lambda c: c[1].x.mean())

        features["reading_sequence"] = list(more_itertools.flatten([cluster_content["index"].tolist()
                                for page_number, page_content in page_cluster_lr_groups.items()
                                for cluster, cluster_content in page_content
                                ]))

        features["relevant"] = True


        self.pdf_obj.pages_to_column_to_text = {page_number:{cluster:" ".join([div.text for div in cluster_content["divs"].tolist()])
                                for cluster, cluster_content in page_content}
                      for page_number, page_content in page_cluster_lr_groups.items()

                      }
        return features

    FeatureStuff = namedtuple("FeatureStuff", ["divs", "coords", "data", "density_field", "labels"])
    def extract_features(self, soup, css_dict, web_view) -> FeatureStuff:
        features = pandas.DataFrame()

        real_properties = get_attributes_script = f"""
           divs = document.querySelectorAll("{self.div_selector}");
           properties = [...divs].map(function(arg) {{
                   rect = arg.getBoundingClientRect();
                   return rect}});
          return properties;"""
        print (get_attributes_script)
        edges_of_all_divs = self.browser.execute_async_script(get_attributes_script)
        print(edges_of_all_divs)
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

    div_selector = 'div[class*=x]'
    def collect_pages_dict(self, pages):
        page_to_divs = [(page_number, page.select(self.div_selector)) for page_number, page in enumerate(pages)]
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
            features[attribute] = features.tag_attributes.apply(
                lambda row: self.css_dict_lookup(row, index, css_dict, attribute))

        features.rename(columns={"bottom":"y", "left":"x"}, inplace=True)

    def css_dict_lookup(self, row, index, css_dict, attribute):
        try:
            return self.get_declaration_value(css_dict[row[index]], attribute)
        except IndexError:
            logging.warning(f"{index} is more than rows  attribute list is long {row}")
            return 0

    def point_density_frequence_per_page (self, features, **kwargs):
        # map computation to pageclusters
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
        features["column_labels"] =  numpy.hstack(other_feature_kinds_stacked.column_labels)

        return coarse_grained_field

    edges = numpy.array(
        [[0, 0], [0, config.reader_height], [config.reader_width, 0], [config.reader_width, config.reader_height]])

    def normalized(self, a):
        return a/[a_line.max()-a_line.min() for a_line in a.T]

    FeatureKinds = namedtuple("FeatureKinds", ["coarse_grained_pdfs", "coarse_grained_field", "mask", "number_of_columns", "column_labels"])

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

        mask, column_labels = self.header_footer_mask(
            fine_grained_field,
            fine_grained_pdfs,
            edges_and_points[:-4],
            number_of_columns,
            page_features.divs)
        return self.FeatureKinds(coarse_grained_pdfs, coarse_grained_field, mask, number_of_columns, column_labels)

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
                int(config.page_array_model * 0.01)):
            peaks, _ = find_peaks(density2D[height], distance=15, prominence=0.000001)
            peaks_at_height_steps.append(peaks)
        lens = [len(peaks) for peaks in peaks_at_height_steps if len(peaks) != 0]
        number_of_clumns = self.most_common_value(lens)
        if number_of_clumns == 0:
            number_of_clumns = 1
        return number_of_clumns

    def header_footer_mask(self, field, pdfs, points, number_of_culumns, divs):
        mask = numpy.full_like(pdfs, False).astype(bool)

        indexed_points = list(enumerate(points))
        indices = numpy.array(range(len(points)))
        # raw column sorting

        left_border = min(points[:,0][points[:,0]>0.05])
        x_sorted_points = [(int(((indexed_point[1][0] - left_border + 0.9)/(1-2*left_border) * number_of_culumns)), indexed_point)
                           for indexed_point in indexed_points]

        if not (len({t[0] for t in x_sorted_points}) == number_of_culumns):
            logging.info ("other number of columns detexted, than sorted to")
        # raw top down sorting
        xy_sorted_points = sorted(x_sorted_points, key=lambda x:x[0]*1000 - x[1][1][1])

        y_sorted_indexed_points = [(len(points)+1,0)] + \
                                  [(column_index_point_tuple[1][0], column_index_point_tuple[1][1][1])
                                      for column_index_point_tuple
                                      in xy_sorted_points] + \
                                  [(len(points)+2,1)]

        indexed_distances = [(i2, list((numpy.abs(b - a), numpy.abs(c - b))))
                            for (i1, a), (i2, b), (i3, c)
                            in threewise(y_sorted_indexed_points)]
        dY = list(id[1] for id in indexed_distances)
        dI = numpy.array(list(id[0] for id in indexed_distances))

        # If there are text boxes on the same height, the distance will be very small due to rounding,
        # replace them with the value for the textbox in the same line
        threshold = 0.3
        to_sanitize = list(enumerate(dY))
        self.sanitize_line_distances(dY, threshold, to_sanitize, direction = 1)
        self.sanitize_line_distances(dY, threshold, to_sanitize, direction = -1)


        norm_distance = numpy.median(list(more_itertools.flatten(dY)))
        distance_std = norm_distance * 0.1

        logging.debug(f"median {norm_distance} std {distance_std}")

        valid_points_at = numpy.logical_and(dY > norm_distance - distance_std, dY < norm_distance + distance_std).any(axis=1)
        good = indices[dI[valid_points_at]]
        mask[good] = True
        column_indices = numpy.full_like(divs, 0)
        column_indices[indices[dI]] = numpy.array([column_index for column_index, point in xy_sorted_points])

        return mask, column_indices

    def sanitize_line_distances(self, dY, threshold, to_sanitize, direction):
        dd_overwrite = 0
        if direction == 1:
            tindex = 1
        elif direction ==-1:
            tindex = 0
        for i, dyy in to_sanitize[::direction]:
            if dyy[tindex] < threshold:
                dY[i][tindex] = dd_overwrite
            else:
                dd_overwrite = dyy[tindex]


import unittest


class TestPaperReader(unittest.TestCase):
    tfu_pdf = TrueFormatUpmarkerPdf2HTMLEX(debug=True, parameterize=False)

    def _test_train(self):
        hdbscan_kwargs = {
            'algorithm': 'boruvka_balltree',
            'metric': 'hamming',
            'cluster_selection_epsilon': 0.5,
            'cluster_selection_method': 'leaf',
            'alpha': 0.95,
            'min_cluster_size': 0.2,
            'min_samples': 0.2
        }

        updating = {
                    #"cluster_selection_epsilon": (0.5, 0.5, 1.5),
                    "min_samples": (0.1, 0.5, 0.9),
                    "min_cluster_size": (0.1, 0.5, 0.9),
                    "alpha": (0.2, 0.5, 1.)
        }
        pandas.options.display.width = 0


        updater = Updater(updating, self.tfu_pdf.hdbscan_kwargs)
        files = list(pathlib.Path('test/data').glob('*.html'))
        train = True
        i = 0


        while train:
            for option in updater.options():
                score = 0
                i += 1
                param, value, kwargs = option
                self.tfu_pdf.hdbscan_kwargs.update(kwargs)

                for path in files:
                    try:
                        this_score = self.extract(path)

                        score += this_score
                        logging.info(f"PDF extraction score: {this_score}")

                    except ValueError as e:
                        logging.error(f"training error {e} ignoring")
                        continue

                updater.notate(option=(param, value), score=score, ml_kwargs=self.tfu_pdf.hdbscan_kwargs)

            updater.give_feedback()
            updater.update()

    def extract(self, path):
        path = str(path)
        kwargs = {}
        kwargs['html_read_from'] = path
        kwargs['html_write_to'] = path + ".computed.htm"
        columns = int(regex.search(r"\d", path).group(0))
        pdf_obj = self.tfu_pdf.convert_and_index(**kwargs)
        score_pdf = pdf_obj.verify(columns=columns, serious=True, test_document=True)
        return score_pdf

    def test_layout_files(self):
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

    def test_columns_and_file_existence(self):
        docs = [
            {
                'html_read_from': 'Laia Font-Ribera - Short-Term Changes in Respiratory Biomarkers after Swimmingin a Chlorinated Pool.pdf.html',
                'html_write_to': 'Laia Font-Ribera - Short-Term Changes in Respiratory Biomarkers after Swimmingin a Chlorinated Pool.pdf.pdf2htmlEX.test.html',
                'cols': 3
            },
            {
                'html_read_from': 'Sonja Vermeulen - Climate Change and Food Systems.pdf.html',
                'html_write_to': 'Sonja Vermeulen - Climate Change and Food Systems.pdf.html.pdf2htmlEX.test.html',
                'cols': 2
            },
            {
                'html_read_from': 'Wei Quian - Translating Embeddings for Knowledge Graph Completion with Relation Attention Mechanism.pdf.html',
                'html_write_to': 'Wei Quian - Translating Embeddings for Knowledge Graph Completion with Relation Attention Mechanism.test.html',
                'cols': 2
            },
            {'html_read_from': 'Ludwig Wittgenstein - Tractatus-Logico-Philosophicus.pdf.html',
             'html_write_to': 'Ludwig Wittgenstein - Tractatus-Logico-Philosophicus.pdf.html.pdf2htmlEX.test.html',
             'cols': 1
             },



            {
                'html_read_from': 'Filipe Mesquita - KnowledgeNet: A Benchmark Dataset for Knowledge Base Population.pdf.html',
                'html_write_to': 'Filipe Mesquita - KnowledgeNet: A Benchmark Dataset for Knowledge Base Population.pdf.pdf2htmlEX.test.html',
                'cols': 2
            },
            {
                'html_read_from': 'Laia Font-Ribera - Short-Term Changes in Respiratory Biomarkers after Swimmingin a Chlorinated Pool.pdf.html',
                'html_write_to': 'Laia Font-Ribera - Short-Term Changes in Respiratory Biomarkers after Swimmingin a Chlorinated Pool.pdf.pdf2htmlEX.test.html',
                'cols': 3
            },
            {
                'html_read_from': 'F. Ning - Toward automatic phenotyping of developing embryos from videos.pdf.html',
                'html_write_to': 'F. Ning - Toward automatic phenotyping of developing embryos from videos.pdf.pdf2htmlEX.test.html',
                'cols': 2
            },
            {
                'html_read_from': 'HumKno.pdf.html',
                'html_write_to': 'HumKno.pdf.pdf2htmlEX.test.html',
                'cols': 1
            }
        ]

        for kwargs in docs:
            logging.error(kwargs)
            columns = kwargs['cols']
            del kwargs['cols']
            kwargs['html_read_from'] = config.appcorpuscook_docs_document_dir + kwargs['html_read_from']
            kwargs['html_write_to'] = config.appcorpuscook_docs_document_dir + kwargs['html_write_to']
            self.tfu_pdf.convert_and_index(**kwargs)
            print (self.tfu_pdf.number_columns, columns)
            assert self.tfu_pdf.number_columns == columns
            assert self.tfu_pdf.indexed_words
            assert os.path.exists(kwargs['html_write_to'])

if __name__ == '__main__':
    unittest.main()