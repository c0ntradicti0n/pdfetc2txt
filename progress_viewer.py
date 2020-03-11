import glob

from suffix_trees import STree
import bs4

import bio_annotation
import config
import os
import regex
import logging




def read_actual_corpus():
    annotations = []
    yet = []
    relevant_files = list(glob.iglob(config.new_corpus_increment + '/*.conll3'))
    for rf in relevant_files:
        #print (rf)
        annotations.extend(bio_annotation.BIO_Annotation.read_annotation_from_corpus(rf, different_only=True, yet=yet))
    #print (annotations)
    return annotations

new_tag_open = "<div class='new'>"
new_tag_close = "</div>"

def inner_strip(text):
    text = text.replace("  "," ")
    text =  text.replace("\n","")
    return text

def whats_new(html):
    soup = bs4.BeautifulSoup(html, "lxml").find("body")
    annotations = read_actual_corpus()
    annotation_texts = list({inner_strip(" ".join(word for word, tag in annotation)) for annotation in annotations})
    text = str(inner_strip(soup.text)).replace(" ", "")
    occurrences = [an.replace(" "," ") for an in annotation_texts if an.replace(" ", "") in text]
    logging.info (f"found {len(occurrences)} occurences ")
    return occurrences