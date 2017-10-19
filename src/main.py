import os
import re


from features.pre_processment import cleaning_steps_builder
from dataset.retriever import ReutersCollection
# from inverted_index import InvertedIndex


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

corpus = []

for document_id in collection.documents:
	corpus.append((document_id, collection.get_words(document_id)))
	break

document_clean_function = lambda document: cleaning_steps_builder(document, remove_stopwords = False, do_stemming = False)
document_stop_function = lambda document: cleaning_steps_builder(document, remove_stopwords = True, do_stemming = False)
document_stem_function = lambda document: cleaning_steps_builder(document, remove_stopwords = False, do_stemming = True)
document_stop_stem_function = lambda document: cleaning_steps_builder(document, remove_stopwords = True, do_stemming = True)

# ranker_clean = Ranker(InvertedIndex(corpus, document_clean_function))
# ranker_stop = Ranker(InvertedIndex(corpus, document_stop_function))
# ranker_stem = Ranker(InvertedIndex(corpus, document_stem_function))
# ranker_stop_stem = Ranker(InvertedIndex(corpus, document_stop_stem_function))