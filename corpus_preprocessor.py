import os
import re
import string
import csv
from collections import Counter
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
            if 'processed_text' in self.corpus[doc]:
                break
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

    def lemmatize(self, remove_stopwords: bool = True, max_length: int = 10000000000) -> None:
        """
        This method lemmatizes the full text of a document. It uses the large version of spacy's
        de_core_news language model. This method optionally removes stop words utilizing nltk's german stop word list plus
        digits and punctuation marks.

        Args:
            remove_stopwords: If true, the method removes the stop words from nltk's german stop word list.
            max_length: Choose max_length, depending on corpus size and available memory.
        """
        try:
            german_model = spacy.load('de_core_news_lg', disable=['parser', 'ner'])
        except IOError:
            os.system("python -m spacy download de_core_news_lg")
            german_model = spacy.load('de_core_news_lg', disable=['parser', 'ner'])

        # Choose max_length depending one corpus size and available memory
        german_model.max_length = max_length

        for doc in self.corpus:
            current_doc = german_model(self.corpus[doc]['processed_text'])

            lemmatised_doc = [token.lemma_ for token in current_doc if
                              (token.text.lower() not in CorpusPreprocessor.german_stop_words) and remove_stopwords]

            self.corpus[doc]['processed_text'] = ' '.join(lemmatised_doc)

    def clean(self, custom_stopwords: bool = False, remove_singular_terms: bool = False) -> None:
        """
        This method cleans a tokenized corpus.

        Args:
            custom_stopwords: If true, custom stop words will be removed.
            remove_singular_terms: If true, terms that occur singularly, will be removed.
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
            if remove_singular_terms:
                self.remove_singular_terms()
            if custom_stopwords:
                self.remove_custom_stopwords()

    def remove_singular_terms(self) -> None:
        """
        This method removes all singularly occurring terms in a tokenized, lemmatized and normalized corpus.
        """
        term_counter = Counter()

        # Count all terms in the entire corpus
        for doc in self.corpus:
            term_counter.update(self.corpus[doc]['processed_text'])

        # Collect singular terms
        singular_terms = {term for term, freq in term_counter.items() if freq == 1}

        # Remove singular terms from each document
        for doc in self.corpus:
            self.corpus[doc]['processed_text'] = [token for token in self.corpus[doc]['processed_text'] if
                                                  token not in singular_terms]

    def remove_custom_stopwords(self, path: str = "data_outputs/stopwords_di_unfiltered.txt") -> None:
        """
        This method removes stop words from a custom stop word list specified by the path parameter. This method ought to
        be used on an already tokenized, lemmatized and normalized corpus.

        Args:
            path: The path and filename of the stop word list.
        """
        with open(path, 'r', encoding='utf-8') as f:
            stopwords = set(f.read().splitlines())

        for doc in self.corpus:
            self.corpus[doc]['processed_text'] = [token for token in self.corpus[doc]['processed_text'] if
                                                  token not in stopwords
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

        self.clean(custom_stopwords=True, remove_singular_terms=True)

    def mine_term_frequency(self, output_path: str = "data_outputs/term_frequency.csv") -> None:
        """
        This method calculates the TF in a corpus, that is already tokenized. The calculated frequencies are serialized
        as csv file under the specified output_path.

        Args:
            output_path: The output path and filename for the csv file.
        """
        term_counter = Counter()

        for doc in self.corpus:
            term_counter.update(self.corpus[doc]['processed_text'])

        sorted_terms = term_counter.most_common()

        with open(output_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Term', 'Frequency'])
            writer.writerows(sorted_terms)


if __name__ == "__main__":
    corpus_dateninstitut = CorpusManager(name="dateninstitut", filename="dateninstitut_fulltext.xml")

    # corpus_dateninstitut.filter_by_title("Dateninstitut")

    corpus_dateninstitut_preprocessor = CorpusPreprocessor(corpus_dateninstitut)

    corpus_dateninstitut_preprocessor.prepare_for_topic_modeling()

    corpus_dateninstitut_preprocessor.serialize_corpus("dateninstitut_unfiltered_processed")
