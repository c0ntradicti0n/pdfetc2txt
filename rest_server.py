import glob
import json
from datetime import datetime
from time import time, sleep
import os
import subprocess
import requests
from flask import request
from flask import Flask

import config
from anyfile2text import paper_reader
from profiler import qprofile
app = Flask(__name__)
import logging
logging.getLogger().setLevel(logging.INFO)
from config import htmls

def code_detect_replace(text):
    return text

def work_out_file(filename):
    meta = {'bla':"huahaua"}

    text_filename = htmls + filename + '.txt'
    html_filename = htmls + filename + '.html'

    with open(text_filename, 'r+') as f:
        text = f.read()

    logging.info("calling ccapp")
    r = requests.post(url="http://localhost:5000/save_text", json={'text':text, 'filename':filename, 'meta':meta})
    with open(html_filename, 'w+') as f:
        f.write(r.text)


reader = paper_reader(_length_limit=40000)
@qprofile
@app.route("/docload", methods=["POST"])
def upload( profile=True):
    meta = {'bla':"huahaua"}
    uploaded_bytes = request.data
    filename = request.args['filename']
    text_filename = htmls + filename + '.txt'
    html_filename = htmls + filename + '.html'

    if not os.path.isfile(text_filename):
        with open(htmls + filename, 'wb') as f:
            f.write(uploaded_bytes)
        logging.info('file uploaded to folder')
        path = htmls + filename

        reader.load_text(path)
        text = reader.analyse()
        print (text)
        with open(text_filename, 'w+') as f:
            f.write(text)

        code_detect_replace (text)
    else:
        with open(text_filename, 'r+') as f:
            text = f.read()

    logging.info("calling ccapp")
    r = requests.post(url="http://localhost:5000/save_text", json={'text':text, 'filename':filename, 'meta':meta})
    #print(r.status_code, r.reason, r.text)
    with open(html_filename, 'w+') as f:
        f.write(r.text)

    logging.info("finished")
    return ""


@app.route("/recompute_all", methods=["GET"])
def recompute_all():
    files = os.listdir(htmls)
    files = [f for f in files if not (f.endswith("html") or f.endswith('txt'))]
    for f in files:
        work_out_file(f)
    return ""


def get_htmls():
    for subdir, dirs, files in os.walk(htmls):
        for file in files:
            if file.endswith(".html"):
                yield file

@app.route("/paths",  methods=['GET', 'POST'])
def html_paths():
    ''' available files '''

    logging.info("get html paths")
    paths = list(get_htmls())
    return json.dumps(paths)

@app.route("/html",  methods=['GET', 'POST'])
def give_html():
    ''' give file '''
    if request.method == 'GET':
        path = htmls + os.sep + request.args['path']
        logging.info("give file " + path)
        try:
            with open( path, 'r+') as f:
                return f.read().encode();
        except FileNotFoundError:
            logging.info("give file " + path)
            return ""
    logging.info("no file path given")
    return ""


if __name__ == '__main__':
    import logging, logging.config, yaml

    logfile = logging.getLogger('file')
    logconsole = logging.getLogger('console')
    logfile.debug("Debug FILE")
    logconsole.debug("Debug CONSOLE")

    app.run(port=5555, debug=True)

