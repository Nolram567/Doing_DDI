import os
import re
import string

from corpus_manager import CorpusManager
import spacy
from nltk.corpus import stopwords
import nltk


class CorpusPreprocessor(CorpusManager):
    """
    This class provides methods to preprocess a corpus as CorpusManager object in order to prepare it for statistical modeling.
    """

    # Load german stop words as class variable.
    try:
        german_stop_words = set(stopwords.words('german'))
    except LookupError:
        nltk.download('stopwords')
        german_stop_words = set(stopwords.words('german'))

    def __init__(self, corpus_manager: CorpusManager):
        """
        The constructor of the class CorpusPreprocessor. We add a new key for the processed corpus to preserve the unprocessed
        full text.

        Args:
            corpus_manager: A CorpusManager object.
        """

        self.corpus = corpus_manager.corpus
        self.name = corpus_manager.name

        for doc in self.corpus:
            self.corpus[doc]['processed_text'] = self.corpus[doc]['fulltext']

    def normalize(self) -> None:
        """
        This method normalizes the full text of a document to lowercase.
        """
        for doc in self.corpus:
            self.corpus[doc]['processed_text'] = self.corpus[doc]['processed_text'].lower()

    def tokenize(self) -> None:
        """
        This method tokenizes the full text of a document by whitespaces. The resulting full text is a list of strings.
        """
        for doc in self.corpus:
            self.corpus[doc]['processed_text'] = self.corpus[doc]['processed_text'].split(" ")

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

        for doc in self.corpus:
            current_doc = german_model(self.corpus[doc]['processed_text'])

            lemmatised_doc = [token.lemma_ for token in current_doc if
                              token.text.lower() not in CorpusPreprocessor.german_stop_words]

            self.corpus[doc]['processed_text'] = ' '.join(lemmatised_doc)

    def clean(self) -> None:
        """
        This method cleans a tokenized corpus.
        """
        for doc in self.corpus:
            self.corpus[doc]['processed_text'] = [token for token in self.corpus[doc]['processed_text'] if
                                                  not (all(char in string.punctuation for char in
                                                           token) or  # Remove all token that are or consist of punctuation marks.
                                                       token.isdigit() or  # Remove digits.
                                                       not any(char.isalpha() for char in
                                                               token) or  # Remove all token who do not contain at least one alphabetic character.
                                                       "@" in token or  # Remove E-Mail-addresses.
                                                       "+" in token  # Remove Phonenumbers.
                                                       )
                                                  ]

    def pre_clean(self) -> None:
        """
        This method cleans the corpus as full text from not well-formed sentences.
        """
        pattern = r'([.:?!])([^\s])'

        for doc in self.corpus:
            self.corpus[doc]['processed_text'] = re.sub(pattern, r'\1 \2', self.corpus[doc]['processed_text'])

    def prepare_for_topic_modeling(self) -> None:
        """
        This method prepares a corpus for topic modeling.
        """

        self.pre_clean()

        self.lemmatize()

        self.normalize()

        self.tokenize()

        self.clean()


if __name__ == "__main__":
    corpus_dateninstitut = CorpusManager(name="dateninstitut", filename="dateninstitut_fulltext.xml")

    corpus_dateninstitut.filter_by_title("Dateninstitut")

    corpus_dateninstitut_preprocessor = CorpusPreprocessor(corpus_dateninstitut)

    corpus_dateninstitut_preprocessor.prepare_for_topic_modeling()

    corpus_dateninstitut_preprocessor.serialize_corpus("dateninstitut_processed")
