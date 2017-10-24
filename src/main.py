import os
import re

from features.pre_processment.cleaning_steps_builder import cleaning_steps_builder
from features.classifier.inverted_index import InvertedIndex
from features.classifier.ranker import Ranker

from dataset.retriever import ReutersCollection


# FUNCTIONS
def printu (str):
	print (str.encode ('utf-8'))


# SCRIPT
print ("Initializing Reuters 21578 ModApte subset, considering only ten bigger categories...")

# The dataset should only consider the documents from the ten bigger categories
collection = ReutersCollection([
	"earn", 
	"acq", 
	"money-fx", 
	"grain", 
	"crude", 
	"trade", 
	"interest", 
	"ship", 
	"wheat", 
	"corn"
])

print ("Collection initialized.")
collection.stats()
collection.detailed_stats()
# collection.preview("corn")

corpus = {}

for doc_name in collection.train_docs:
	corpus[doc_name] = collection.get_words(doc_name)
	# break

document_clean_function = lambda document: cleaning_steps_builder(document, remove_stopwords = False, do_stemming = False, do_lemmatizing = False)
document_stop_function = lambda document: cleaning_steps_builder(document, remove_stopwords = True, do_stemming = False, do_lemmatizing = False)
document_stem_function = lambda document: cleaning_steps_builder(document, remove_stopwords = False, do_stemming = True, do_lemmatizing = False)
document_stop_stem_function = lambda document: cleaning_steps_builder(document, remove_stopwords = True, do_stemming = True, do_lemmatizing = False)
document_stop_lem_stem_function = lambda document: cleaning_steps_builder(document, remove_stopwords = True, do_stemming = True, do_lemmatizing = True)

ranker_clean = Ranker(InvertedIndex(corpus, collection, document_clean_function), collection)
ranker_stop = Ranker(InvertedIndex(corpus, collection, document_stop_function), collection)
ranker_stem = Ranker(InvertedIndex(corpus, collection, document_stem_function), collection)
ranker_stop_stem = Ranker(InvertedIndex(corpus, collection, document_stop_stem_function), collection)
ranker_stop_lem_stem = Ranker(InvertedIndex(corpus, collection, document_stop_lem_stem_function), collection)

# print(collection.get_document(collection.test_docs[0]))
# words = document_clean_function(collection.get_words(collection.test_docs[0]))
# string = ''

# for word in words:
# 	string += word + ' '

# print (string)
print("\n")

results = {}
results['ranker_clean'] = []
results['ranker_stop'] = []
results['ranker_stem'] = []
results['ranker_stop_stem'] = []
results['ranker_stop_lem_stem'] = []

for doc in collection.test_docs:
	rank = ranker_stop.classify(collection.get_words(doc))
	print(rank)
	print (collection.get_categories(doc))
	print(collection.get_words(doc))
	print("\n")