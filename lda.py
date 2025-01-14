import json
import os
import logging
from gensim.corpora import MmCorpus
from corpus_manager import CorpusManager
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import statistics

def visualize_model(lda_model: LdaModel, bag_of_words_model: list, dictionary: corpora.dictionary, filename: str) -> None:
    """
    Visualizes an LDA model and saves the visualization as an HTML file.

    Args:
        lda_model (LdaModel): The trained LDA model to visualize.
        bag_of_words_model (list): The bag-of-words representation of the corpus.
        dictionary (corpora.dictionary): The dictionary used to create the bag-of-words model.
        filename (str): The name of the file to save the HTML visualization.

    Returns:
        None
    """
    
    vis_data = gensimvis.prepare(lda_model, bag_of_words_model, dictionary)
    pyLDAvis.save_html(vis_data, os.path.join('data_outputs/lda_visualisation', filename))


def save_model(lda_model: LdaModel, bag_of_words_model: list, dictionary: corpora.dictionary, filename: str) -> None:
    """
    Save the LDA model, bag of words model, and dictionary to disk.

    Args:
        lda_model (LdaModel): The trained LDA model to be saved.
        bag_of_words_model (list): The bag of words model to be serialized and saved.
        dictionary (corpora.dictionary): The dictionary to be saved.
        filename (str): The base filename to use for saving the models and dictionary.

    Returns:
        None
    """
    dictionary.save(os.path.join('data_outputs/models', f'dictionary_{filename}.dict'))
    lda_model.save(os.path.join('data_outputs/models', f'topic_model_{filename}.lda'))
    MmCorpus.serialize(os.path.join(f'data_outputs/models', f'bow_corpus_{filename}.mm'), bag_of_words_model)


if __name__ == "__main__":

    # Enable logging to track conversion time to monitor if the parameters iterations and passes are sufficiently high.
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    corpus_dateninstitut = CorpusManager(name="dateninstitut", filename="dateninstitut_full_final.json", from_xml=False)

    relevance = []
    for doc in corpus_dateninstitut.corpus.values():
        relevance.append(doc.get('relevance_dateninstitut'))

    median_relevance = statistics.median(relevance)
    print(median_relevance)

    corpus_dateninstitut.filter_by_relevance(threshold=median_relevance, term='dateninstitut')

    corpus_dateninstitut.filter_by_length(threshold=150)

    doc_lengths = []
    for doc in corpus_dateninstitut.corpus:
        doc_lengths.append(len(corpus_dateninstitut.corpus[doc]['processed_text']))

    print(
        f"Das vorverarbeitete Korpus umfasst {len(list(corpus_dateninstitut.corpus.keys()))} Dokumente mit durchschnittlich {sum(doc_lengths) / len(doc_lengths)} Token.")

    processed_texts = []
    document_dates = []

    for doc_id, doc_data in corpus_dateninstitut.corpus.items():
        processed_texts.append(doc_data['processed_text'])

    dictionary = corpora.Dictionary(processed_texts)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_texts]

    coherence_map = {}
    my_models = {}
    # We implement the interval of k as for loop.
    for k in range(15, 35):
        model = LdaModel(
            corpus=bow_corpus,
            id2word=dictionary,
            num_topics=k,
            iterations=300,  # Check if documents converge with the given parameters.
            passes=30,  # Check if documents converge with the given parameters.
            chunksize=64,  # Choose chunksize according to the available memory and corpus size. Chunksize has minor ramifications on the calculated distributions.
            alpha='asymmetric',  # Topic Distribution per document
            eta='auto',  # Automatic distribution of terms per topic
            eval_every=1,  # Evaluation after every iteration
            random_state=42
        )

        # We calculate the semantic coherence of the topic model with k topics with the coherence metric C_V of Röder et al. (2015).
        coherence_model = CoherenceModel(model=model, texts=processed_texts, dictionary=dictionary,
                                         coherence='c_v')

        my_models[coherence_model.get_coherence()] = model  # collisions are unlikely.

        # We save the coherence for every given k.
        coherence_map[k] = coherence_model.get_coherence()
        print(f'coherence score C_v with {k} topics: {coherence_model.get_coherence()}')

    with open("data_outputs/coherence_map_big_I", "w", encoding="utf-8") as f:
        json.dump(coherence_map, f, indent=2, ensure_ascii=False)

    # determine the best model of the first run
    max_coherence_k = my_models[max(my_models.keys())].num_topics

    coherence_map = {}
    my_models = {}

    # narrow the interval using the data from the first run
    for k in range(max_coherence_k - 3, max_coherence_k + 3):
        # we use the same parameters besides k
        model = LdaModel(
            corpus=bow_corpus,
            id2word=dictionary,
            num_topics=k,
            iterations=300,
            passes=30,
            chunksize=64,
            alpha='asymmetric',
            eta='auto',
            eval_every=1,
            random_state=42
        )

        # We calculate the semantic coherence of the topic model with k topics with the coherence metric C_V of Röder et al. (2015).
        coherence_model = CoherenceModel(model=model, texts=processed_texts, dictionary=dictionary,
                                         coherence='c_v')

        my_models[coherence_model.get_coherence()] = model  # collisions are unlikely.

        # We save the coherence for every given k.
        coherence_map[k] = coherence_model.get_coherence()
        print(f'coherence score C_v with {k} topics: {coherence_model.get_coherence()}')

    with open("data_outputs/coherence_map_big_II", "w", encoding="utf-8") as f:
        json.dump(coherence_map, f, indent=2, ensure_ascii=False)

    # determine the best performing model
    max_coherence = max(my_models.keys())

    most_coherent_model = my_models[max_coherence]

    # save the best performing model
    save_model(most_coherent_model, bow_corpus, dictionary,
               filename=f"k{most_coherent_model.num_topics}_c_v_{max_coherence}")

    # visualize the best performing model and save the figure as html document
    visualize_model(most_coherent_model, bow_corpus, dictionary,
                    filename=f"k{most_coherent_model.num_topics}_c_v_{max_coherence}.html")
