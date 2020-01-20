import glob
import json
import urllib
from datetime import datetime
from time import time, sleep
import os
import subprocess
import requests
from bs4 import BeautifulSoup
from flask import request
from flask import Flask

import config
from anyfile2text import paper_reader
from profiler import qprofile
from webpageparser import WebPageParser

app = Flask(__name__)
import logging
logging.getLogger().setLevel(logging.INFO)
from config import htmls

def code_detect_replace(text):
    return text

def work_out_file(filename, folder=htmls):
    meta = {'bla':"huahaua"}

    text_filename = folder + filename + '.txt'
    html_filename = folder + filename + '.html'

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
def recompute_all(folder =htmls):
    files = os.listdir(folder)
    files = [f for f in files if not (f.endswith("html") or f.endswith('txt'))]
    for f in files:
        work_out_file(f)
    return ""


def get_htmls(folder=htmls):
    for subdir, dirs, files in os.walk(folder):
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



wpp = WebPageParser(config.scraped_difbet)
def latest_difference_between(source=config.scraped_difbet):
    logging.info("downloading front page of difference between")

    f = urllib.request.urlopen('http://differencebetween.net')
    page = f.read()
    soup = BeautifulSoup(page, 'html.parser')
    name_box = soup.find_all('a', attrs = {'rel':'bookmark'})
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

@app.route("/difbet_paths", methods=['GET', 'POST'])
def difbet_paths():
    ''' available files '''
    latest_difference_between()
    logging.info("get difbet paths")
    paths = list(get_htmls(folder=config.scraped_difbet))[:5]
    return json.dumps(paths)

@app.route("/difbet_html", methods=['GET', 'POST'])
def div_between_give_html():
    ''' give file '''
    logging.info("get difbet html")

    if request.method == 'GET':
        path = config.scraped_difbet + os.sep + request.args['path']
        logging.info("give file " + path)
        try:
            with open(path, 'r+') as f:
                return f.read().encode();
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

