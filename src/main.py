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
	# "money-fx", 
	# "grain", 
	# "crude", 
	# "trade", 
	# "interest", 
	# "ship", 
	"wheat", 
	"corn"
])

print ("Collection initialized.")
collection.stats()
collection.detailed_stats()
# collection.preview("corn")

# Training phase
print ('\nCreating corpus...')

corpus = {}

for doc_name in collection.train_docs:
	corpus[doc_name] = collection.get_words(doc_name)
	# break


# document_clean_function = lambda document: cleaning_steps_builder(document, remove_stopwords = False, do_stemming = False, do_lemmatizing = False)
# document_stop_function = lambda document: cleaning_steps_builder(document, remove_stopwords = True, do_stemming = False, do_lemmatizing = False)
# document_stem_function = lambda document: cleaning_steps_builder(document, remove_stopwords = False, do_stemming = True, do_lemmatizing = False)
# document_stop_stem_function = lambda document: cleaning_steps_builder(document, remove_stopwords = True, do_stemming = True, do_lemmatizing = False)
document_stop_lem_stem_function = lambda document: cleaning_steps_builder(document, remove_stopwords = True, do_stemming = True, do_lemmatizing = True)

# y = document_stop_stem_function(corpus[doc_name])
# print (y)
# for x in y:
# 	print(x)

# print('\nInitializing ranker with no text pre processment...')
# ranker_clean = Ranker(InvertedIndex(corpus, collection, document_clean_function), collection)
# print('\nInitializing ranker with removing stopwords as pre processment...')
# ranker_stop = Ranker(InvertedIndex(corpus, collection, document_stop_function), collection)
# print('\nInitializing ranker with doing stemming as pre processment...')
# ranker_stem = Ranker(InvertedIndex(corpus, collection, document_stem_function), collection)
# print('\nInitializing ranker with removing stopwords and doing stemming as pre processment...')
# ranker_stop_stem = Ranker(InvertedIndex(corpus, collection, document_stop_stem_function), collection)
print('\nInitializing ranker with all pre processment steps...')
ranker_stop_lem_stem = Ranker(InvertedIndex(corpus, collection, document_stop_lem_stem_function), collection)

# print(collection.get_document(collection.test_docs[0]))
# words = document_clean_function(collection.get_words(collection.test_docs[0]))
# string = ''

# for word in words:
# 	string += word + ' '

# print (string)
# Results calculation

print('\nClassifying test documents...')

results = {}
# results['ranker_clean'] = []
# results['ranker_stop'] = []
# results['ranker_stem'] = []
# results['ranker_stop_stem'] = []
# results['ranker_stop_lem_stem'] = []

for doc in collection.test_docs:
	results[doc] = {}

	# rank = ranker_clean.classify(collection.get_words(doc))
	# results[doc]['ranker_clean'] = rank

	# rank = ranker_stop.classify(collection.get_words(doc))
	# results[doc]['ranker_stop'] = rank
	
	# rank = ranker_stem.classify(collection.get_words(doc))
	# results[doc]['ranker_stem'] = rank
	
	# rank = ranker_stop_stem.classify(collection.get_words(doc))
	# results[doc]['ranker_stop_stem'] = rank
	
	rank = ranker_stop_lem_stem.classify(collection.get_words(doc))
	results[doc]['ranker_stop_lem_stem'] = rank

	# print(rank)
	# print (collection.get_categories(doc))
	# print(collection.get_words(doc))
	# print("\n")

# Results analysis
print('\nStarting results analysis...')
corpus_true_positive = {}
corpus_false_positive = {}
corpus_true_negative = {}
corpus_false_negative = {}

doc_true_positive = {}
doc_false_positive = {}
doc_false_negative = {}
doc_true_negative = {}

cat_true_positive = {}
cat_false_positive = {}
cat_false_negative = {}
cat_true_negative = {}

labeled_positive = {}
labeled_negative = {}

for doc in results:
	doc_categories = collection.get_categories(doc)
	for ranker in results[doc]:
		doc_result = [cat for index, cat in enumerate(results[doc][ranker]) if index < len(doc_categories)]
		doc_false_result = [cat for cat in doc_result if cat[0] not in doc_categories]
		doc_result = [cat for cat in doc_result if cat[0] in doc_categories]

		doc_true_positive[ranker] = len(doc_result)
		doc_false_positive[ranker] = len(doc_categories) - len(doc_result)
		doc_false_negative[ranker] = len(doc_categories) - len(doc_result)
		doc_true_negative[ranker] = 10 - (doc_false_negative[ranker] + doc_false_positive[ranker] + doc_true_positive[ranker])


		if ranker not in labeled_positive:
			labeled_positive[ranker] = 0
			labeled_negative[ranker] = 0
			corpus_true_positive[ranker] = 0
			corpus_false_positive[ranker] = 0
			corpus_true_negative[ranker] = 0
			corpus_false_negative[ranker] = 0

		corpus_true_positive[ranker] += doc_true_positive[ranker]
		corpus_false_positive[ranker] += doc_false_positive[ranker]
		corpus_true_negative[ranker] += doc_true_negative[ranker]
		corpus_false_negative[ranker] += doc_false_negative[ranker]

		for cat in doc_result:
			if cat[0] not in cat_true_positive:
				cat_true_positive[cat[0]] = {}
				cat_false_positive[cat[0]] = {}
				cat_false_negative[cat[0]] = {}
				cat_true_negative[cat[0]] = {}

			cat_true_positive[cat[0]][ranker] = corpus_true_positive[ranker]
			cat_false_positive[cat[0]][ranker] = corpus_false_positive[ranker]
			cat_false_negative[cat[0]][ranker] = corpus_true_negative[ranker]
			cat_true_negative[cat[0]][ranker] = corpus_false_negative[ranker]

		labeled_positive[ranker] += len(doc_categories)
		labeled_negative[ranker] += 10 - len(doc_categories)

# print('Results:')

for ranker in labeled_positive:
	print ('\nResults for ', ranker, ':')
	
	precision = corpus_true_positive[ranker]/(corpus_true_positive[ranker] + corpus_false_positive[ranker])
	print('\tPrecision: %.15f' % precision)

	recall = corpus_true_positive[ranker]/(corpus_true_positive[ranker] + corpus_false_negative[ranker])
	print('\tRecall: %.15f' % recall)

	for cat in cat_true_positive:
		print ('\n\tResults for ', ranker, ' for cat ', cat, ':')
		
		precision = cat_true_positive[cat][ranker]/(cat_true_positive[cat][ranker] + cat_false_positive[cat][ranker])
		print('\t\tPrecision: %.15f' % precision)

		recall = cat_true_positive[cat][ranker]/(cat_true_positive[cat][ranker] + cat_false_negative[cat][ranker])
		print('\t\tRecall: %.15f' % recall)