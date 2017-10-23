class InvertedIndex:
    """
    Inverted index of the received docs.

    Contains the following fields:

    :clean_func: used to clean and filter the received corpus doc terms
    :proc_corpus: resultant corpus from applying clean_func in the original corpus
    :term_docs: dict of term -> (term_frequency, {doc: [term_doc_indices]})
    """

    def __init__(self, corpus, collection=None, clean_func=None):
        """
        Initializes the inverted index with the received corpus.

        :param corpus: docs to terms dict
        :param clean_func: function the process a list of words
        """
        self._collection = collection
        self.clean_func = clean_func if clean_func is not None else lambda word: word

        # Process corpus
        if collection is None:
            self.proc_corpus = {doc: self.clean_func(corpus[doc]) for doc in corpus}
        else:
            self.proc_corpus = {}
            for doc in corpus:
                cats = self._collection.get_categories(doc)
                for cat in cats:
                    if cat not in self.proc_corpus:
                        self.proc_corpus[cat] = []

                    self.proc_corpus[cat] += (self.clean_func(corpus[doc]))

        # self.term_docs = self._build_term_docs(self.proc_corpus)
        self.term_cats = self._build_term_cats(self.proc_corpus)

    @staticmethod
    def _build_term_cats(corpus):
        """
        Creates the term_cats dict based on the received corpus.

        :param corpus: cats to terms dict
        """
        term_cats = {}
        for cat in corpus:
            terms = corpus[cat]
            for index, term in enumerate(terms):
                if term not in term_cats:
                    term_cats[term] = (0, {})
                if cat not in term_cats[term][1]:
                    term_cats[term][1][cat] = []
                term_cats[term] = (term_cats[term][0] + 1, term_cats[term][1])
                term_cats[term][1][cat].append(index)
        return term_cats