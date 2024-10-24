import json
import os
import re
import string
from collections import Counter
from corpus_analyzer import CorpusAnalyzer
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

    def lemmatize(self, remove_stopwords: bool = True, max_length: int = 9131400) -> None:
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

        # Set maximum document length for the model
        german_model.max_length = max_length

        for doc in self.corpus:
            try:
                # Process the whole document at once
                current_doc = german_model(self.corpus[doc]['processed_text'])

                lemmatized_doc = [token.lemma_ for token in current_doc if
                                  (token.text.lower() not in CorpusPreprocessor.german_stop_words) and remove_stopwords]

                self.corpus[doc]['processed_text'] = ' '.join(lemmatized_doc)

            except MemoryError:
                # If memory error occurs, process the document in chunks
                print(f"MemoryError: Processing document '{doc}' in smaller chunks.")

                lemmatized_tokens = []
                processed_text = self.corpus[doc]['processed_text']

                # Split the text into chunks of 1000 words each
                chunk_size = 1000
                for chunk in range(0, len(processed_text), chunk_size):
                    chunk_text = processed_text[chunk:chunk + chunk_size]
                    current_chunk = german_model(chunk_text)

                    lemmatized_chunk = [token.lemma_ for token in current_chunk if
                                        (
                                                    token.text.lower() not in CorpusPreprocessor.german_stop_words) and remove_stopwords]

                    lemmatized_tokens.extend(lemmatized_chunk)

                self.corpus[doc]['processed_text'] = ' '.join(lemmatized_tokens)

    def n_gram_inclusion(self) -> None:
        """
        This method includes Mulitword Expressions into the corpus.
        """

        with open('data_preprocessing/MWE.json', 'r', encoding='utf-8') as json_file:
            MWE = json.load(json_file)
        with open('data_preprocessing/MWE_reversed.json', 'r', encoding='utf-8') as json_file:
            MWE_reversed = json.load(json_file)

        mutliword_expressions = []
        for entry in MWE.values():
            mutliword_expressions.append(entry)

        for key, value in self.corpus.items():
            new_doc = []
            skipped = False
            for i, token in enumerate(value.get('processed_text')):
                if skipped:
                    skipped = False
                    continue
                if i < len(value.get('processed_text')) - 1:
                    bigram = [token, value.get('processed_text')[i + 1]]
                    if bigram in mutliword_expressions:
                        new_doc.append(MWE_reversed[str(bigram)])
                        skipped = True
                    else:
                        new_doc.append(token)
                else:
                    new_doc.append(token)

            self.corpus[key]['processed_text'] = new_doc

    def clean(self, custom_stopwords: bool = False, remove_rare_terms: int = 1) -> None:
        """
        This method cleans a tokenized corpus.

        Args:
            custom_stopwords: If true, custom stop words will be removed.
            remove_rare_terms: If value>0, terms that occur as often or less than specified, will be removed.
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
            if remove_rare_terms:
                self.remove_rare_terms(n=remove_rare_terms)
            if custom_stopwords:
                self.remove_custom_stopwords()

    def remove_rare_terms(self, n: int = 1) -> None:
        """
        This method removes all terms that occur as often or less than n in a tokenized, lemmatized and normalized corpus.
        """
        term_counter = Counter()

        # Count all terms in the entire corpus
        for doc in self.corpus:
            term_counter.update(self.corpus[doc]['processed_text'])

        # Collect terms with insufficient frequency
        singular_terms = {term for term, freq in term_counter.items() if freq <= n}

        # Remove all collected terms
        for doc in self.corpus:
            self.corpus[doc]['processed_text'] = [token for token in self.corpus[doc]['processed_text'] if
                                                  token not in singular_terms]

    def remove_custom_stopwords(self, path: str = "data_preprocessing/stopwords_di_unfiltered.txt") -> None:
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

        self.n_gram_inclusion()

        self.clean(custom_stopwords=True, remove_rare_terms=3)


if __name__ == "__main__":

    corpus_dateninstitut = CorpusManager(name="dateninstitut", filename="dateninstitut_fulltext.xml", from_xml=True)

    corpus_dateninstitut_preprocessor = CorpusPreprocessor(corpus_dateninstitut)

    corpus_dateninstitut_preprocessor.prepare_for_topic_modeling()

    corpus_dateninstitut_analyzer = CorpusAnalyzer(corpus_dateninstitut)

    corpus_dateninstitut_analyzer.calculate_term_relevance(term="dateninstitut")

    corpus_dateninstitut.serialize_corpus("dateninstitut_full_final.json")

