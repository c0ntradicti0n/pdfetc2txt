import json
import urllib
import os
from urllib.request import Request

import requests
from bs4 import BeautifulSoup
from flask import request
from flask import Flask

from helpers.programming import deprecated
import config
from anyfile2text import PaperReader
from profiler import qprofile

app = Flask(__name__)
import logging
logging.getLogger().setLevel(logging.INFO)

from CCAPI.annotate_json import AnnotateJson
from Scraper.ScrapeDifferenceBetween import DifferncenceBetweenScraper
from Topics.topics import MakeTopics
from Topics.predict_topics import PredictTopics

from layouteagle.LayoutEagle import LayoutEagle
layouteagle = LayoutEagle()
layouteagle.make_model()

difference_between_pipe = PfadAmeise('html', 'nlp.css')
pdf_pipe = PfadAmeise('pdf', 'nlp.css')



def work_out_file(path):
    meta = {'bitbtex_data':"not implemented"}

    # Parse the file, text is written in the CorpusCookApps document dir
    logging.info('Parse file')
    reader.parse_file_format(path)

    # Starting annotation
    logging.info(f"Annotating {path}: Calling CorpusCookApp to call CorpusCook")
    try:
        requests.post(
            url=f"http://localhost:{config.app_port}/annotate_certain_json_in_doc_folder",
            json=
                {'filename': reader.paths.json_path,
                 'meta': meta},
        )
    except requests.exceptions.ConnectionError:
        logging.error("CorpusCookApp offline")





def latest_difference_between(n=10):

    try:
        logging.info("downloading front page of differencebetween")

        req = Request(
            'https://differencebetween.com',
            headers={'User-Agent': 'Mozilla/5.0'})

        f = urllib.request.urlopen(req)
        page = f.read()



        soup = BeautifulSoup(page, 'html.parser')
        anchors = list(soup.find_all('a', attrs = {'rel':'bookmark'})) [:n]
        for anchor in anchors:
            text_ground_path = anchor.get_text()
            text_path = config.appcorpuscook_diff_txt_dir + text_ground_path + "_html" + ".txt"
            html_path = config.appcorpuscook_diff_document_dir + text_ground_path + ".html"

            if not os.path.exists(text_path):
                logging.info(f"downloading page for '{text_ground_path}'")
                diffpagereq = Request(
                    anchor['href'],
                    headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(diffpagereq) as response:
                    html = response.read().decode("utf-8")
                    with open(text_path, 'w') as tf:
                        tf.write(BeautifulSoup(html, features="lxml").body.text)
                    with open(html_path, 'w') as hf:
                        hf.write(html)
                work_out_file(html_path)

        t_diff.update()
    except requests.exceptions.ConnectionError:
        logging.error("Connection was refused by site")
    except urllib.error.URLError:
        logging.error("Connection was refused, no internet")



@qprofile
@app.route("/docload", methods=["POST"])
def upload():
    meta = {'bitbtex_data':"not implemented"}
    uploaded_bytes = request.data
    filename = request.args['filename']
    os.system(f"rm \"\{filename}\"")

    logging.info('File upload to folder')
    path = config.appcorpuscook_docs_document_dir + filename
    with open(path, 'wb') as f:
        f.write(uploaded_bytes)

    # Parsing, annotating, topic modelling
    work_out_file(path)

    # Updating topics
    logging.info("Updating topics")
    MakeTopics.update()

    logging.info("Finished upload, topic modelling and upmarking")
    return ""


@app.route("/recompute_all", methods=["GET"])
def recompute_all():
    #os.system("rm ../CorpusCook/cache/predictions/*.*")
    recompute(config.appcorpuscook_docs_document_dir)
    recompute(config.appcorpuscook_diff_html_dir)
    # Updating topics
    logging.info("Updating topics")
    MakeTopics.update()


def recompute(folder):
    logging.info("Recomputation started")
    files = os.listdir(folder)
    files = [f for f in files if not (f.endswith("html") or f.endswith('txt'))]
    list(pdf_pipe(files))
    logging.info("Recomputation finished")
    return ""


def get_htmls(folder=config.appcorpuscook_docs_document_dir):
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".html"):
                yield file


@app.route("/diff_paths",  methods=['GET', 'POST'])
def get_topical_paths_diff():
    logging.info("give topic modelled paths for differencebetween")
    print (t_diff.get_paths())
    return json.dumps(t_diff.get_paths())

@app.route("/docs_paths",  methods=['GET', 'POST'])
def get_topical_paths_docs():
    logging.info("give topic modelled paths for docs")
    return json.dumps(t_docs.get_paths())


@app.route("/get_doc",  methods=['GET', 'POST'])
@deprecated
def doc_html():
    ''' give file '''
    return ""


@app.route("/get_diff", methods=['GET', 'POST'])
@deprecated
def diff_html():
    ''' give file '''
    return ""


if __name__ == '__main__':
    app.debug = True
    app.run(port=config.doc_port, debug=True, use_reloader=False)

