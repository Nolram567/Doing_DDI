import json
import os
import logging
from gensim.corpora import MmCorpus
from corpus_manager import CorpusManager
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis


def visualize_model(lda_model: LdaModel, bag_of_words_model: list, dictionary: corpora.dictionary, filename: str, save: bool = True) -> None:
    vis_data = gensimvis.prepare(lda_model, bag_of_words_model, dictionary)
    if save:
        pyLDAvis.save_html(vis_data, os.path.join('data_outputs/lda_visualisation', filename))


def save_model(lda_model: LdaModel, bag_of_words_model: list, dictionary: corpora.dictionary, filename: str) -> None:
    dictionary.save(os.path.join('data_outputs/models', f'dictionary_{filename}.dict'))
    lda_model.save(os.path.join('data_outputs/models', f'topic_model_{filename}.lda'))
    MmCorpus.serialize(os.path.join(f'data_outputs/models', f'bow_corpus_{filename}.mm'), bag_of_words_model)


if __name__ == "__main__":

    # Enable logging to track conversion time to monitor if the parameters iterations and passes are sufficiently high.
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    corpus_dateninstitut = CorpusManager(name="dateninstitut", filename="dateninstitut_processed", from_xml=False)

    processed_texts = []
    document_dates = []

    for doc_id, doc_data in corpus_dateninstitut.corpus.items():
        processed_texts.append(doc_data['processed_text'])
        document_dates.append(doc_data['document_date'])  # Datetime-Objekte

    dictionary = corpora.Dictionary(processed_texts)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_texts]

    chunksize = 2000  # Choose chunksize according to the available memory. Chunksize has minor ramifications on the calculated distributions.
    iterations = 100  # Check if documents converge with the given parameters.
    passes = 30  # Check if documents converge with the given parameters.
    eval_every = None  # We do not evaluate perplexity.

    alpha = 'auto'  # The hyperparameter alpha is learned by the model.
    eta = 'auto'  # The hyperparameter eta is learned by the model.

    coherence_map = {}
    my_models = {}
    # We implement the interval of k as for loop.
    for k in range(5, 31):
        model = LdaModel(
            corpus=bow_corpus,
            id2word=dictionary,
            chunksize=chunksize,
            num_topics=k,
            alpha=alpha,
            eta=eta,
            iterations=iterations,
            passes=passes,
            eval_every=eval_every,
            random_state=42
        )

        # We calculate the semantic coherence of the topic model with k topics with the coherence metric C_V of Röder et al. (2015).
        coherence_model = CoherenceModel(model=model, texts=processed_texts, dictionary=dictionary,
                                         coherence='c_v')

        my_models[coherence_model.get_coherence()] = model  # collisions are unlikely.

        # We save the coherence for every given k.
        coherence_map[k] = coherence_model.get_coherence()
        print(f'Kohärenzscore C_v mit {k} Themen: {coherence_model.get_coherence()}')

    with open("data_outputs/coherence_map", "w", encoding="utf-8") as f:
        json.dump(coherence_map, f, indent=2, ensure_ascii=False)

    max_coherence = max(my_models.keys())

    most_coherent_model = my_models[max_coherence]

    save_model(most_coherent_model, bow_corpus, dictionary,
               filename=f"k{most_coherent_model.num_topics}_c_v_{max_coherence}")

    visualize_model(most_coherent_model, bow_corpus, dictionary,
                    filename=f"k{most_coherent_model.num_topics}_c_v_{max_coherence}", save=True)
