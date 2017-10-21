from inverted_index import InvertedIndex
from math import log, sqrt
import numpy as np

class Ranker:
    inverted_index = None
    collection = None
    categories_vectors = {}
    term_indexer = {}

    def _init_(self, inverted_index, collection):
        self.inverted_index = inverted_index
        self.collection = collection
        self.__build_term_indexer()
        self.__calculate_categories_vectors()
        
    """
    Set the inicial order for term array
    """

    def __build_term_indexer(self):
        index = 0
        
        for term in self.inverted_index.term_documents:
            self.term_indexer[term] = index
            index += 1
        
    """
    Calculate document vector for each document
    """

    def __calculate_categories_vectors(self):
        categories = None
        
        for document_name in self.inverted_index.processed_corpus:
            document_vector = __calculate_document_vector(document_name)
            __update_categories(document_name, document_vector)
            

            
    def __update_categories(self, document_name, document_vector):
        categories = self.collection.get_document_categories(document_name)

        for category_name in categories:
            if category_name not in self.categories_vectors:
                self.categories_vectors[category_name] = (0, len(self.inverted_index.term_documents.keys)*[0])

            self.categories_vectors[category_name][0] += document_vector[0]

            matrix = np.array([self.categories_vectors[category_name][1], document_vector[1]])
            self.categories_vectors[category_name][1] = matrix.sum(axis=0)

    """
    Retorna a tupla da soma dos quadrados(tf*idf) de cada termo e o vetor de tf*idf .
    """    

    def __calculate_document_vector(self, document_name):
        document_vector_mag2 = 0
        document_vector = len(self.inverted_index.term_documents.keys) * [0]

        for term in self.inverted_index.term_documents:
            (term_frequency, documents) = self.inverted_index.term_documents[term]
            # TODO: usar normalização
            tf = documents[document_name] / term_frequency # term frequency on document / total term frequency
            idf = log(len(self.inverted_index.processed_corpus) / len(documents)) 
            tfidf = tf * idf
            document_vector[self.term_indexer[term]] = tfidf
            document_vector_mag2 += tfidf * tfidf
        
        #sqrt(document_vector_mag2)
        return (document_vector_mag2, document_vector)

    def __calculate_query_vector(self, query_terms):
        query_vector = len(self.inverted_index.term_documents.keys) * [0]
        query_inverted_index = InvertedIndex([('query', query_terms)])

        for term in query_inverted_index.term_documents:
            if term not in self.inverted_index.term_documents.keys:
                continue

            (query_term_frequency, _) = query_inverted_index.term_documents[term]            
            (term_frequency, documents) = self.inverted_index.term_documents[term]

            tf = query_term_frequency / term_frequency
            idf = log(len(self.inverted_index.processed_corpus) / len(documents))

            query_vector[self.term_indexer[term]] = tf * idf

        return query_vector


    """
    Returns list of similarity of query and categories    
    """  

    def search(self, query_words):
        query_terms = self.inverted_index.clean_function(query_words)
        query_vector = __calculate_query_vector(query_terms)

        # compute similarity
        similarities = []
        
        for document_name in self.document_vectors:
            (document_vector_mag, document_vector) = document_vectors[document_name]
            similarity = self.list_similarity(query_vector, document_vector, document_vector_mag)
            similarities.append((document_name, similarity))

        similarities.sort(key = lambda tuple: tuple[1])
        return similarities

    def list_similarity(self, query_vector, document_vector, document_vector_mag):
        dot_product = 0

        for i in range(0, len(query_vector)):
            dot_product += query_vector[i] * document_vector[i]

        return dot_product / document_vector_mag
