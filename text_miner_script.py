from corpus_manager import CorpusManager

if __name__ == "__main__":

    corpus = CorpusManager(name="dateninstitut", filename="dateninstitut_fulltext.xml")

    print(len(corpus.corpus.keys()))  # The corpus comprises 271 documents.

    corpus.filter_by_title("Dateninstitut")  # 246 entries in the corpus were deleted.

    print(len(corpus.corpus.keys()))  # The corpus comprises 25 documents.

    length = 0

    for key in corpus.corpus:
        length += len(corpus.corpus[key]["fulltext"])

    print(length)  # The corpus comprises 236514 characters.
