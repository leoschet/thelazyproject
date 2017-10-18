import os
import re


from lib.pre_processment import cleaning_steps_builder
# from inverted_index import InvertedIndex
# from ranker import Ranker


# DIRS
# rootdir ='../UnityDocumentation/en/Manual'


# REGEX
script_tag_regex = re.compile(r'<script>(.|\n)*</script>')
html_tags_regex = re.compile(r'<[^>]*>')
whitespace_regex = re.compile(r'( |\n|[^A-Za-z0-9])+')


# FUNCTIONS
def printu (str):
	print (str.encode ('utf-8'))


def clear_split_document (document):
	document = script_tag_regex.sub(' ', document)
	document = html_tags_regex.sub(' ', document)
	document = whitespace_regex.sub(' ', document)

	# printu (document)

	return document.split(' ')


# SCRIPT
print ('Running the lazy classifier')

corpus = []

# for _, _, files in os.walk (rootdir):
# 	for file in files:
# 		document = open (rootdir + '/' + file, mode='r', encoding='utf-8').read()
# 		document_terms = clear_split_document (document)

# 		corpus.append((file, document_terms))

# 		# TODO: remove break
# 		break

document_clean_function = lambda document: stopwords_stemming(document, remove_stopwords = False, do_stemming = False)
document_stop_function = lambda document: stopwords_stemming(document, remove_stopwords = True, do_stemming = False)
document_stem_function = lambda document: stopwords_stemming(document, remove_stopwords = False, do_stemming = True)
document_stop_stem_function = lambda document: stopwords_stemming(document, remove_stopwords = True, do_stemming = True)

# ranker_clean = Ranker(InvertedIndex(corpus, document_clean_function))
# ranker_stop = Ranker(InvertedIndex(corpus, document_stop_function))
# ranker_stem = Ranker(InvertedIndex(corpus, document_stem_function))
# ranker_stop_stem = Ranker(InvertedIndex(corpus, document_stop_stem_function))