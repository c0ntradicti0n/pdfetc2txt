import json
import urllib
import os
import requests
from bs4 import BeautifulSoup
from flask import request
from flask import Flask
from regex import regex

from progress_viewer import whats_new
import config
from anyfile2text import paper_reader
from profiler import qprofile
from webpageparser import WebPageParser

app = Flask(__name__)
import logging
logging.getLogger().setLevel(logging.INFO)

os.system(". ~/.bashrc")

def get_raw_html_doc(path):
    with open(path + ".html", 'r+') as f:
        html = f.read();
    occurrences = whats_new(html)
    html = insert_markedup_js(html, occurrences)
    return html.encode()


def get_pdf2htmlEX_doc(path):
    with open(path, 'r+') as f:
        html = f.read();
    occurrences = whats_new(html)
    html = insert_markedup_js(html, occurrences)
    return html.encode()


def insert_markedup_js(html, occurrences):
    tag = "</body"
    before_end = html.find(tag)
    html = html[:before_end] + f"""
<script onload="">
    function markup() {{
        console.log("working");
        mark_what_was_recently_annotated("{"`~`".join(occurrences).replace('"', '')}");
        }}
    $(document).ready(markup);
    $(window).load(markup);
</script>
        """ + html[before_end:]
    html = regex.sub(' +', ' ', html)
    return html


def code_detect_replace(text):
    return text

def work_out_file(path):
    meta = {'lorem':"ipsum"}

    # Parse the file, text is written in the CorpusCookApps document dir
    logging.info('Parse file')
    reader.parse_file_format(path)

    # Starting annotation
    logging.info("Annotating it: Calling CorpusCookApp to call CorpusCook")
    requests.post(url="http://localhost:5000/annotate_certain_json_in_doc_folder",
                  json={'filename': reader.json_text_extract, 'meta': meta})




reader = paper_reader(_length_limit=40000)
@qprofile
@app.route("/docload", methods=["POST"])
def upload():
    meta = {'bitbtexdata':"Not implemented yet"}
    uploaded_bytes = request.data
    filename = request.args['filename']
    os.system(f"rm \"\{filename}\"")

    logging.info('File upload to folder')
    path = config.appcorpuscook_pdf_dir + filename
    with open(path, 'wb') as f:
        f.write(uploaded_bytes)

    # Parsing, annotating, topic modelling
    work_out_file(path)

    # Updating topics
    logging.info("Updating topics")
    t_docs.update()

    logging.info("Finished upload, topic modelling and upmarking")
    return ""


@app.route("/recompute_all", methods=["GET"])
def recompute_all():
    os.system("rm ../CorpusCook/cache/predictions/*.*")
    recompute("./docs/")
    recompute("./scraped_difference_between/")
    # Updating topics
    logging.info("Updating topics")
    t_docs.update()


def recompute(folder):
    files = os.listdir(folder)
    files = [f for f in files if not (f.endswith("html") or f.endswith('txt'))]
    for f in files:
        work_out_file(config.appcorpuscook_pdf_dir + f)
    return ""


def get_htmls(folder=config.appcorpuscook_pdf_dir):
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".html"):
                yield file

import topic_modelling
t_diff = topic_modelling.Topicist(directory='scraped_difference_between')
t_docs = topic_modelling.Topicist(directory='docs')

@app.route("/diff_paths",  methods=['GET', 'POST'])
def get_topical_paths_diff():
    logging.info("give topic modelled paths for differencebetween")
    return json.dumps(t_diff.get_paths())

@app.route("/docs_paths",  methods=['GET', 'POST'])
def get_topical_paths_docs():
    logging.info("give topic modelled paths for docs")
    return json.dumps(t_docs.get_paths())


@app.route("/get_doc",  methods=['GET', 'POST'])
def doc_html():
    ''' give file '''
    if request.method == 'GET':
        path = config.appcorpuscook_pdf_dir + os.sep + request.args['path']
        pdf2htmlEX_path = path + ".pdf2htmlEX.html"

        # parsed with pdf2htmlEX
        return get_pdf2htmlEX_doc(pdf2htmlEX_path)

        # TODO fallback to TIKA return get_raw_html_doc(path)

    logging.info("no file path given")
    return ""


wpp = WebPageParser(config.scraped_difbet)
def latest_difference_between(source=config.scraped_difbet):
    logging.info("downloading front page of difference between")

    f = urllib.request.urlopen('http://differencebetween.net')
    page = f.read()
    soup = BeautifulSoup(page, 'html.parser')
    name_box = list(soup.find_all('a', attrs = {'rel':'bookmark'})) [:2]
    for anchor in name_box:
        text_ground_path = anchor['title']
        text_path = source +  text_ground_path + '.txt'

        if not os.path.exists(text_path):
            logging.info(f"downloading page for '{anchor['title']}'")

            content = urllib.request.urlopen(anchor['href'])
            text = wpp.html_to_text(content)
            logging.info(f"text starts with '{text[:100]}'")
            with open(text_path, 'w+') as text_file:
                text_file.write(text)
            work_out_file(text_ground_path, folder=source)
            t_diff.update()


@app.route("/diff_paths_list", methods=['GET', 'POST'])
def difbet_paths():
    ''' available files '''
    latest_difference_between()
    logging.info("get difbet paths")
    paths = list(get_htmls(folder=config.scraped_difbet))[:5]
    return json.dumps(paths)


@app.route("/get_diff", methods=['GET', 'POST'])
def difffs_html():
    ''' give file '''
    logging.info("get difbet html")

    if request.method == 'GET':
        path = config.scraped_difbet + os.sep + request.args['path']
        logging.info("give file " + path)
        try:
            return get_raw_html_doc(path)
        except FileNotFoundError:
            logging.info("give file " + path)
            return ""
    logging.info("no file path given")
    return ""

###########################################################################################

if __name__ == '__main__':
    import logging, logging.config, yaml

    logfile = logging.getLogger('file')
    logconsole = logging.getLogger('console')
    logfile.debug("Debug FILE")
    logconsole.debug("Debug CONSOLE")

    app.run(port=5555, debug=True)

