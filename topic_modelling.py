import itertools
from pathlib import Path
from pprint import pprint

import dariah
import pandas
from gensim.models.fasttext import FastText
from multi_rake import Rake
rake = Rake(language_code="en",
            min_chars=3,
            max_words=2,
            min_freq=2,
            )

from helpers.nested_dict_tools import reverseDict


class Topicist:
    def __init__(self, directory="docs"):
        self.directory= directory
        self.update()

    def update(self):

        self.lda_model, self.vis = dariah.topics(directory=self.directory,
                                   stopwords=100,
                                    num_topics=8,
                                    num_iterations=1000)

        print (self.lda_model.topics.iloc[:10, :5])

        """
        filepath_pattern="*.txt"
        directory = "docs"
        sentences = [[w for w in  sent.split() if len(w)>5]
                     for text in topicist.read_long_text(directory, filepath_pattern)
                     for sent in text.split(".") if len(sent)>14
                     ]

        self.ft_model = FastText(sentences, sg=1, hs=1, size=200, workers=12, iter=2, min_count=2)

        topics =   [list(x) for x in self.lda_model.topics.to_numpy()]

        print ("finding abstractions for each topic")
        similar = self.ft_model.most_similar(positive=['man', 'code', 'master', 'human'],topn=1)
        print (similar)

        self.topic_df = self.lda_model.topics

        self.topic_df ['abstraction'] = [self.min_distant(topic_words) for topic_words in topics]

        columns = ['word' +str(i) for i in range(5)] + ['abstraction']
        with pandas.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print (self.topic_df [columns])
        """
        self.doc2topic = self.lda_model.topic_document.idxmax(axis=0).to_dict()
        self.topic2doc = {}
        for doc, topic in self.doc2topic.items():
            if topic in self.topic2doc:
                self.topic2doc[topic].append(doc)
            else:
                self.topic2doc[topic] = [doc]

        self.headword2doc = {}
        for topic, docs in self.topic2doc.items():
            self.headword2doc[self.create_headwords(docs)] = docs

    def create_headwords(self, paths):
        text = ""
        for path in paths:
            with open(self.directory + "/" + path +".txt", 'r+', errors='ignore') as f:
                text += f.read() + " "
                text = self.clean_text(text)
        poss_headwords = rake.apply(text)
        return poss_headwords[0][0]

        """
        combinations = itertools.combinations(topic_words)
        return self.ft_model.most_similar(positive=topic_words[:13],topn=1)[0][0]
        # CFLAGS="-Wno-narrowing" pip install cld2-cffi
        """

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
