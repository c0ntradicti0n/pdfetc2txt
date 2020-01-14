import glob
from datetime import datetime
from time import time, sleep
import os
import requests
from flask import request
from flask import Flask

from anyfile2text import paper_reader
from profiler import qprofile
app = Flask(__name__)
import logging
logging.getLogger().setLevel(logging.INFO)
from config import htmls

def code_detect_replace(text):
    return text

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
    logging.info("finished")
    return {'html': r.text, 'path': html_filename}




if __name__ == '__main__':
    import logging, logging.config, yaml

    logfile = logging.getLogger('file')
    logconsole = logging.getLogger('console')
    logfile.debug("Debug FILE")
    logconsole.debug("Debug CONSOLE")

    app.run(port=5555, debug=True)

