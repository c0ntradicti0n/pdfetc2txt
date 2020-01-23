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
    relevant_files = list(glob.iglob(config.cc_corpus_collection_path + '/*.conll3'))
    for rf in relevant_files:
        print (rf)
        annotations.extend(bio_annotation.BIO_Annotation.read_annotation_from_corpus(rf))
    return annotations

new_tag_open = "<span class='new'>"
new_tag_close = "</span>"

def mark_new(html):
    soup = bs4.BeautifulSoup(html, "lxml")
    annotations = read_actual_corpus()
    annotation_texts = list({" ".join(word for word, tag in annotation) for annotation in annotations})
    text = soup.text
    occurrences = [an for an in annotation_texts if an in text]
    logging.info (f"found {len(occurrences)} occurences ")

    for an in occurrences:

        for text in soup.findAll(text=True):
            if regex.search(an, text):
                new_html = new_tag_open + text + new_tag_close
                new_soup = bs4.BeautifulSoup(new_html, features="lxml")
                text.parent.replace_with(new_soup.span)



    return str(soup)



if __name__ == '__main__':
    html = """ <span class="contrast level1 contrast1"><span class="privative"> nature are directly intuited from sense everything else worthy of the name of science follows demonstrably from these first principles . What characterizes the whole enterprise is a degree of certainty which distinguishes it most crucially from mere opinion . But Aristotle sometimes offered a second demarcation criterion , orthogonal to this one between science and opinion . Specifically , he distinguished between know-how ( the sort of knowledge which the craftsman and the engineer possess ) and what we might call know-why or demonstrative understanding ( which the scientist alone possesses ) . A shipbuilder , for instance , knows how to form pieces of wood together so as to make a seaworthy vessel but he does not have , and has no need for , a syllogistic , causal demonstration based on the primary principles or first causes of things . Thus , he needs to know that wood , when properly sealed , floats but he need not be able to show by virtue of what principles and causes wood has this property of buoyancy . By contrast , the scientist is concerned with what Aristotle calls the reasoned fact until he can show why a thing use   of   bio   accumulative   ,   persistent   ,   toxic   ,   and   otherwise   hazardous   materials   Environmental   Chemistry   Human   Health   Cleaner   air   Cleaner   water   Reduced </span></span>  <br> </br>
  <span class="contrast level1 contrast1"><span class="privative"> use  nature are directly intuited from sense everything else worthy of the name of science follows demonstrably from these first principles . What characterizes the whole enterprise is a degree of certainty which distinguishes it most crucially from mere opinion . But Aristotle sometimes offered a second demarcation criterion , orthogonal to this one between science and opinion . Specifically , he distinguished between know-how ( the sort of knowledge which the craftsman and the engineer possess ) and what we might call know-why or demonstrative understanding ( which the scientist alone possesses ) . A shipbuilder , for instance , knows how to form pieces of wood together so as to make a seaworthy vessel but he does not have , and has no need for , a syllogistic , causal demonstration based on the primary principles or first causes of things . Thus , he needs to know that wood , when properly sealed , floats but he need not be able to show by virtue of what principles and causes wood has this property of buoyancy . By contrast , the scientist is concerned with what Aristotle calls the reasoned fact until he can show why a thing of   toxic   and   hazardous   materials   and   maximum   safety   for   workers   in   the   chemical   establishment   Safer   consumer   products   of   all   types   Less   exposure   to </span></span>  <br> </br>
  <span class="contrast level1 contrast1"><span class="privative"> such nature are directly intuited from sense everything else worthy of the name of science follows demonstrably from these first principles . What characterizes the whole enterprise is a degree of certainty which distinguishes it most crucially from mere opinion . But Aristotle sometimes offered a second demarcation criterion , orthogonal to this one between science and opinion . Specifically , he distinguished between know-how ( the sort of knowledge which the craftsman and the engineer possess ) and what we might call know-why or demonstrative understanding ( which the scientist alone possesses ) . A shipbuilder , for instance , knows how to form pieces of wood together so as to make a seaworthy vessel but he does not have , and has no need for , a syllogistic , causal demonstration based on the primary principles or first causes of things . Thus , he needs to know that wood , when properly sealed , floats but he need not be able to show by virtue of what principles and causes wood has this property of buoyancy . By contrast , the scientist is concerned with what Aristotle calls the reasoned fact until he can show why a thing  toxic   chemicals   as   endocrine   disruptors   (   chemicals   that   interfere   withhormonal   systems   at   certain   doses   ) </span></span>
"""

    print (mark_new(html))
    assert ('<span class="new">' in mark_new(html))

