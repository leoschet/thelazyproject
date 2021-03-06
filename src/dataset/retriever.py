"""
Subset of reuters 21578, "ModApte", considering only received categories.

util: 
- http://www.nltk.org/book/ch02.html
- https://miguelmalvarez.com/2015/03/20/classifying-reuters-21578-collection-with-python-representing-the-data/

might be useful:
- http://www.nltk.org/howto/corpus.html
- https://miguelmalvarez.com/2016/11/07/classifying-reuters-21578-collection-with-python/
"""


from nltk.corpus import reuters
from nltk import word_tokenize

class ReutersCollection:
    interest_categories = []
    documents = []
    train_docs = []
    test_docs = []

    """
    Initializes the collection considering only the received categories.

    :param categories: [str] list of categories names
    """

    def __init__(self, categories):
        self.interest_categories = categories
        self.documents = reuters.fileids(categories)
        self.train_docs = list(filter(lambda doc: doc.startswith("train"), self.documents))
        self.test_docs = list(filter(lambda doc: doc.startswith("test"), self.documents))


    def get_words(self, document_id):
        return word_tokenize(reuters.raw(document_id))


    def get_document(self, document_id):
        return reuters.raw(document_id)

    def get_categories(self, document_id):
        return [cat for cat in reuters.categories(document_id) if cat in self.interest_categories]
    
    def get_train_docs_len(self, category):
        category_docs = reuters.fileids(category)        
        return len(list(filter(lambda doc: doc.startswith("train"), category_docs)))
    

    """
    Prints the size of training and test set, the total amount of documents 
    and categories used to retrieve the documents.
    """

    def stats(self):
        # print (reuters.categories())
        print ("Collection stats:")

        # List of documents
        print ("\t" + str(len(self.documents)) + " documents")
        print ("\t" + str(len(self.train_docs)) + " total train documents")
        print ("\t" + str(len(self.test_docs)) + " total test documents")

        # List of categories
        all_categories = reuters.categories()
        print ("\tConsidering " + str(len(self.interest_categories)) + " from a total of " + str(len(all_categories)) + " categories")


    """
    Prints the stats of every category of interest
    """

    def detailed_stats(self):
        for cat in self.interest_categories:
            self.category_stats(cat)


    """
    Prints the size of training and test set, the total amount of documents 
    of a specific category
    """

    def category_stats(self, category):
        print ("Stats of " + category + ":")
        category_docs = reuters.fileids(category)

        category_train_docs = list(filter(lambda doc: doc.startswith("train"), category_docs))
        category_test_docs = list(filter(lambda doc: doc.startswith("test"), category_docs))

        # List of documents
        print ("\t" + str(len(category_docs)) + " documents")
        print ("\t" + str(len(category_train_docs)) + " total train documents")
        print ("\t" + str(len(category_test_docs)) + " total test documents\n")


    """
    Prints a preview of the category, including some words and a raw document

    :param categories: str name of desired category
    """

    def preview(self, category):
        # Documents in a category
        category_docs = reuters.fileids(category)
        
        # Words for a document
        document_id = category_docs[0]
        document_words = reuters.words(document_id)
        for word in (document_words):
            print (word)
    
        # Raw document
        print(reuters.raw(document_id))