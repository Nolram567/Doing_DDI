from gensim.models import LdaModel
import logging

if __name__ == "__main__":

    # Enable logging to track conversion time to monitor if the parameters iterations and passes are sufficiently high.
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    chunksize = 2000  # Choose chunksize according to the available memory. Chunksize has ramifications on the calculated distributions.
    iterations = 0
    passes = 0
    eval_every = None  # We do not evaluate perplexity.

    # Define topics as range to determine the optimal k through iterative computation according to our metric (e.g. coherence).
    num_topics = range(10, 100)
    alpha = 'auto'  # The hyperparameter alpha is learned by the model.
    eta = 'auto'  # The hyperparameter eta is learned by the model.

    model = LdaModel(
        corpus=None,
        id2word=None,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every,
        random_state=0
    )



