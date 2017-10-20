"""
Inverted index of the received documents.
"""


class InvertedIndex:
    clean_function = None
    processed_corpus = {}
    term_documents = {}

    """
    Initializes the inverted index with the received corpus.

    :param corpus: [(str, [str])] list of document names and words
    """

    def __init__(self, corpus, clean_function = None):
        self.clean_function = clean_function
        self.__build_inverted_index(corpus)

    """
    Creates the term_document dict based on the document_list and process the corpus with the received clean function.
    
    :param corpus: [(str, [str])] list of document names and words
    """

    def __build_inverted_index(self, corpus):
        for document in corpus:
            document_name = document[0]
            document_words = self.clean_function(document[1]) if self.clean_function is not None else document[1] 
            self.processed_corpus[document_name] = document_words
            
            for word in document_words:
                if word not in self.term_documents:
                    self.term_documents[word] = (0, {}) # inicializa

                if document_name not in self.term_documents[word][1]:
                    self.term_documents[word][1][document_name] = 0

                self.term_documents[word] = (
                    self.term_documents[word][0] + 1,
                    self.term_documents[word][1]
                )
                self.term_documents[word][1][document_name] += 1
                