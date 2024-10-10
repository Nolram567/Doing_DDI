import os
from corpus_manager import CorpusManager
import spacy
from nltk.corpus import stopwords
import nltk

class CorpusPreprocessor(CorpusManager):

    def __init__(self, corpus_manager: CorpusManager):
        """
        The constructor of the class CorpusPreprocessor.

        Args:
            corpus_manager: A CorpusManager object.
        """

        self.corpus = corpus_manager.corpus
        self.name = corpus_manager.name

    def normalize(self) -> None:
        """
        This method normalizes the full text of a document to lowercase.
        """
        for doc in self.corpus:
            self.corpus[doc]["fulltext"] = self.corpus[doc]["fulltext"].lower()

    def tokenize(self) -> None:
        """
        This method tokenizes the full text of a document by whitespaces. The resulting full text is a list of strings.
        """
        for doc in self.corpus:
            self.corpus[doc]["fulltext"] = self.corpus[doc]["fulltext"].split(" ")

    def lemmatize(self) -> None:
        """
        This method lemmatizes the full text of a document. It uses the large version of spacy's
        de_core_news language model. This method also removes stop words utilizing nltk's german stop word list plus
        digits and punctuation marks.
        """
        try:
            german_model = spacy.load('de_core_news_lg', disable=['parser', 'ner'])
        except IOError:
            os.system("python -m spacy download de_core_news_lg")
            german_model = spacy.load('de_core_news_lg', disable=['parser', 'ner'])

        german_model.max_length = 10000000

        nltk.download('stopwords')
        german_stop_words = set(stopwords.words('german'))

        for doc in self.corpus:
            current_doc = german_model(self.corpus[doc]["fulltext"])

            lemmatised_doc = [token.lemma_ for token in current_doc if
                              token.text.lower() not in german_stop_words
                              and not token.is_punct
                              and not token.is_digit]

            self.corpus[doc]["fulltext"] = ' '.join(lemmatised_doc)

    def prepare_for_topic_modeling(self):
        pass


if __name__ == "__main__":

    corpus_dateninstitut = CorpusManager(name="dateninstitut", filename="dateninstitut_fulltext.xml")

    corpus_dateninstitut.filter_by_title("Dateninstitut")

    corpus_dateninstitut_preprocessor = CorpusPreprocessor(corpus_dateninstitut)

    corpus_dateninstitut_preprocessor.normalize()

    corpus_dateninstitut_preprocessor.lemmatize()

    corpus_dateninstitut_preprocessor.tokenize()

    for i, e in enumerate(corpus_dateninstitut_preprocessor.corpus.keys()):

        print(f"{corpus_dateninstitut_preprocessor.corpus[e]['fulltext']}\n")

        if i == 2:
            break