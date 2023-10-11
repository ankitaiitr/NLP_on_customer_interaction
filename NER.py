# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 23:11:07 2020

@author: Ankita
https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

sent = preprocess(ex)
'''Our chunk pattern consists of one rule, that a noun phrase, NP,
 should be formed whenever the chunker finds an optional determiner,
 DT, followed by any number of adjectives, JJ, and then a noun, NN.'''
pattern = 'NP: {<DT>?<JJ>*<NN>}'

cp = nltk.RegexpParser(pattern)
cs = cp.parse(sent)
print(cs)

from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
iob_tagged = tree2conlltags(cs)
pprint(iob_tagged)


'''With the function nltk.ne_chunk(), we can recognize named entities
 using a classifier, the classifier adds category labels such as PERSON,
 ORGANIZATION, and GPE.'''
ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(ex)))
print(ne_tree)

'''spacy'''


import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
pprint([(X.text, X.label_) for X in doc.ents])

pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])














