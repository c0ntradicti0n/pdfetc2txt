import glob
import json
import logging
from pathlib import Path
from pprint import pprint
import os
import dariah
from multi_rake import Rake
rake = Rake(language_code="en",
            min_chars=10,
            max_words=2,
            min_freq=5,
            )

class Topicist:
    def __init__(self, directory="docs"):
        self.directory = directory
        self.update()

    def update(self):
        self.docs_paths = list(glob.glob(self.directory + "/*.*"))
        self.state_file = self.directory.replace("\\", "") + ".topicstate";
        try:
            with open(f'{self.state_file}', "r") as f:
                self.state = json.loads(f.read())
            if self.state['state'] == str (self.docs_paths):
                self.headword2doc = self.state['result']
                return
            else:
                logging.info("New state, making new topics")
        except Exception:
            raise
            logging.info ("No state file, generating a state")

        self.lda_model, self.vis = dariah.topics(directory=self.directory,
                                   stopwords=100,
                                    num_topics=8,
                                    num_iterations=100)

        print (self.lda_model.topics.iloc[:10, :5])

        self.doc2topic = self.lda_model.topic_document.idxmax(axis=0).to_dict()
        self.topic2doc = {}
        for doc, topic in self.doc2topic.items():
            if topic in self.topic2doc:
                self.topic2doc[topic].append(doc)
            else:
                self.topic2doc[topic] = [doc]

        self.headword2doc = {}
        for topic, doc_paths in self.topic2doc.items():
            self.headword2doc[self.create_headwords(doc_paths)] = doc_paths

        with open(self.state_file, "w") as f:
            state = {
                'state': str(self.docs_paths),
                'result': self.headword2doc
            }
            f.write(json.dumps(state))

    def create_headwords(self, paths):
        text = ""
        for path in paths:
            with open(self.directory + "/" + path +".txt", 'r+', errors='ignore') as f:
                text += f.read() + " "
                text = self.clean_text(text)
        poss_headwords = rake.apply(text)
        if poss_headwords:
            result = poss_headwords[0][0]
            return result
        else:
            return "no topic set"

    def read_long_text(dir, pattern):
        filepaths = Path().rglob(dir + "/" + pattern)
        for filepath in filepaths:
            with open(filepath) as f:
                yield f.read()

    def get_paths(self):
        return self.headword2doc

    def clean_text(self, text):
        allowed_chars = sorted(""" 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz""")
        return "".join(c for c in text if c in allowed_chars)

def main():
    t = Topicist()
    pprint (t.get_paths())

if __name__ == '__main__':
    main()
