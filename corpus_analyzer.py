import csv
from collections import Counter
from corpus_manager import CorpusManager

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

    def calculate_temporal_term_occurrence(self, output_filename='term_occurrence.json'):
        pass
