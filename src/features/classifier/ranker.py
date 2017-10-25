import math
from functools import reduce

from features.classifier.inverted_index import InvertedIndex


class Ranker:
    """
    Receives queries and ranks the documents based on the inverted index, using vector space with TF*IFD as term width
    calculation formula and cosine as similarity metric.
    """

    def __init__(self, inverted_index, collection):
        """
        Initializes the ranker with the received inverted index.

        :param inverted_index: inverted index
        """
        self._collection = collection
        self._inverted_index = inverted_index
        self._term_index = {term: index for index, term in enumerate(self._inverted_index.term_cats)}
        self._cat_vecs = {cat: self._calculate_cat_vec(cat) for cat in self._inverted_index.proc_corpus}

    def _calculate_cat_vec(self, cat):
        """
        Calculates the TF*IFD vector for the received document.

        :param doc: document name
        :return: document vector
        """
        cat_vec = len(self._inverted_index.term_cats) * [0]
        cat_vec_mag = 0
        for term in self._inverted_index.proc_corpus[cat]:
            term_freq, term_cats = self._inverted_index.term_cats[term]
            
            # By dividing the category term frequency, we get an avarage frequency.
            # Making the data more similar to test documents
            tf = (len(term_cats[cat]) / self._collection.get_train_docs_len(cat)) / term_freq
            idf = math.log(len(self._inverted_index.proc_corpus) / len(term_cats))
            tf_idf = tf * idf
            cat_vec[self._term_index[term]] = tf_idf
            cat_vec_mag += tf_idf ** 2
        return cat_vec, cat_vec_mag ** (1 / 2)
 
    def _calculate_query_vec(self, terms):
        """
        Calculates the TF*IDF vector to the received query terms

        :param terms: list of query terms
        :return: query vector
        """
        query_vec = len(self._inverted_index.term_cats) * [0]
        query_vec_mag = 0
        query_inverted_index = InvertedIndex({'query': terms})
        for term in query_inverted_index.proc_corpus['query']:
            if term not in self._inverted_index.term_cats:
                continue
            query_term_freq, _ = query_inverted_index.term_cats[term]
            term_freq, term_cats = self._inverted_index.term_cats[term]
            tf = query_term_freq / term_freq
            idf = math.log(len(self._inverted_index.proc_corpus) / len(term_cats))
            tf_idf = tf * idf
            query_vec[self._term_index[term]] = tf_idf
            query_vec_mag += tf_idf ** 2
        return query_vec, query_vec_mag ** (1 / 2)

    def classify(self, query):
        """
        Calculates the similarity of a query to all categories

        :param query: the query
        :return: list of categories ordered by similarity
        """
        terms = self._inverted_index.clean_func(query)
        terms = [term for term in terms if term in self._inverted_index.term_cats]
        query_vec, query_vec_mag = self._calculate_query_vec(terms)
        similarities = [(cat, self._calculate_similarity(query_vec, query_vec_mag, cat)) for cat in self._inverted_index.proc_corpus.keys()]
        similarities = sorted(similarities, key=lambda tup: tup[1], reverse=True)
        
        # index = 1
        # cond = True
        # while index < len(similarities) and cond:
        #     if (similarities[index-1][1]/3)*2 > similarities[index-1][1] - similarities[index][1]:
        #         cond = False
        #     else:
        #         index += 1

        # return [sim for i, sim in enumerate(similarities) if i < index]
        # return [sim for sim in similarities if sim[1] > 0.00099]
        return similarities

    def _calculate_similarity(self, query_vec, query_vec_mag, cat):
        """
        Calculates the cosine similarity between the received query vector and the cat.

        :param query_vec: query vector
        :param query_vec_mag: query vector magnitude
        :param cat: category name
        :return: similarity between query and caegory
        """
        cat_vec, cat_vec_mag = self._cat_vecs[cat]
        denominator = query_vec_mag * cat_vec_mag
        return 0 if denominator == 0 else sum(map(lambda x, y: x * y, query_vec, cat_vec)) / denominator


    @staticmethod
    def _find_consecutive_values(list1, list2):
        return [y for x in list1 for y in list2 if y > x and y - x <= 2]
