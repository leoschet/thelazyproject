import nltk

english_stopwords = nltk.corpus.stopwords.words('english')
porter_stemmer = nltk.stem.PorterStemmer()

def cleaning_steps_builder(document_terms, remove_stopwords, do_stemming):
	document_terms = [word.lower() for word in document_terms]
	
	if (remove_stopwords):
		document_terms = [word for word in document_terms if word not in english_stopwords]
	
	if (do_stemming):
		document_terms = [porter_stemmer.stem(word) for word in document_terms]

	return document_terms