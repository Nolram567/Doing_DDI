import csv
import os
from collections import Counter, defaultdict
from corpus_manager import CorpusManager
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import bigrams
import pandas as pd



class CorpusAnalyzer(CorpusManager):

    def __init__(self, corpus_manager: CorpusManager):
        """
        The constructor of the class CorpusAnalyzer.

        Args:
            corpus_manager: A CorpusManager object.
        """
        self.corpus = corpus_manager.corpus
        self.name = corpus_manager.name

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

    def calculate_term_relevance(self, term: str) -> None:
        """
        This method calculates the tfidf for a given term and saves it in the corpus. The tfidf is saved under the key
        f"relevance_{term}" for every document.

        Args:
            term: The term for which the tfidf is calculated.
        """

        # Extract documents
        corpus = [self.corpus[doc]['processed_text'] for doc in self.corpus]

        # Join tokens
        corpus_as_strings = [' '.join(tokens) for tokens in corpus]

        vectorizer = TfidfVectorizer()

        # Fitting und Transformation
        tfidf_matrix = vectorizer.fit_transform(corpus_as_strings)

        # Extract terms
        terms = vectorizer.get_feature_names_out()

        # check if term is part of the vocabulary
        if term not in terms:
            print(f"Term '{term}' not found")
            return None

        # Extract the tfidf for the term from the matrix.
        tfidf_values = tfidf_matrix[:, terms.tolist().index(term)].todense().tolist()

        # save the term relevance for every document in the corpus.
        for doc_i, doc in enumerate(self.corpus.keys()):
            self.corpus[doc][f"relevance_{term}"] = tfidf_values[doc_i][0]

    def calculate_temporal_term_occurrence(self, output_filename='term_occurrence.json') -> None:
        """
        This method generates a json file which term salience within every quarter year for the data dashboard.

        Args:
            output_filename: The filename for the json file.
        """
        term_occurrence = defaultdict(lambda: defaultdict(int))  # Structure: {year-quarter: {term: count}}

        # Sort the corpus by document_date and ignore entries without valid datetime
        sorted_corpus = sorted(
            (item for item in self.corpus.items() if isinstance(item[1].get('document_date'), datetime)),
            key=lambda item: item[1].get('document_date')
        )

        # Iterate through each document in the sorted corpus
        for doc_name, doc_data in sorted_corpus:
            document_date = doc_data.get('document_date')
            processed_text = doc_data.get('processed_text', [])

            if document_date and processed_text:
                # Get year and quarter from document_date
                year = document_date.year
                quarter = (document_date.month - 1) // 3 + 1
                year_quarter = f"{year}-Q{quarter}"

                # Count each term in the processed_text for the given year-quarter
                for term in processed_text:
                    term_occurrence[year_quarter][term] += 1

        # Convert the term_occurrence dictionary into a list of dictionaries for JSON export
        json_data = [
            {"term": term, "date": year_quarter, "count": count}
            for year_quarter, terms in term_occurrence.items()
            for term, count in terms.items()
        ]

        # Write the results to a JSON file
        with open(os.path.join("data_outputs", output_filename), 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=4, ensure_ascii=False)

        print(f"Term occurrence data has been saved to 'data_outputs/{output_filename}'")

    def calculate_cooccurrence(self) -> None:
        """
        This method calculates all possible 2-Gram of a given corpus and serializes the result as csv.
        """
        temp = []

        for value in self.corpus.values():
            temp += value.get('processed_text')

        bi_grams = list(bigrams(temp))

        bigram_counts = Counter(bi_grams)

        cooccurrence = pd.DataFrame(columns=["Bigramm", "Count"])
        rows = []
        for bigram, count in bigram_counts.items():
            rows.append({"Bigramm": bigram, "Count": count})

        cooccurrence = pd.concat([cooccurrence, pd.DataFrame(rows)], ignore_index=True)

        cooccurrence.to_csv("data_outputs/cooccurrence.csv", index=False)
