import nltk
from nltk.stem import WordNetLemmatizer
from nltk import re

# Regex for text cleaning
NUMBER_REGEX = re.compile(r'[0-9]')
BEGIN_CHAR_REGEX = re.compile(r'[A-Za-z]')
BEGIN_NUMBER_REGEX = re.compile(r'^[0-9]')

english_stopwords = nltk.corpus.stopwords.words('english')
porter_stemmer = nltk.stem.PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

def cleaning_steps_builder(document_terms, remove_stopwords = False, do_stemming = False, do_lemmatizing = False):
	document_terms = [word.lower() for word in document_terms]
	document_terms = ['NUM' if re.match(BEGIN_NUMBER_REGEX, word) else word for word in document_terms]
	
	if (remove_stopwords):
		document_terms = [word for word in document_terms if word not in english_stopwords]
	
	if (do_stemming):
		document_terms = [porter_stemmer.stem(word) for word in document_terms]

	# document_terms = [NOT_CHAR_REGEX.sub('', word) for word in document_terms]
	# document_terms = [word for word in document_terms if word is not '' and not len(word) == 1]

	if (do_lemmatizing):
		document_terms = [wordnet_lemmatizer.lemmatize(word, pos='n') for word in document_terms]

	min_length = 3
	# return document_terms
	return [term for term in document_terms if BEGIN_CHAR_REGEX.match(term) and len(term) >= min_length]