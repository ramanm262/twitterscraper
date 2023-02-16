from processing import *
from analysis import *
from gensim.models import Word2Vec

mode = "loading"  # "training" or "loading"

stream_file = "resources/stream_output.txt"
punctuation_dict = {',': ' ', ". ": ' ', "&": '', '"': '', '(': '', ')': '', '’': "'", '*': '', '+': '', '[': '',
                    ']': '', '|': '', ';': '', "..": "", "...": ""}
additional_stopwords = ['like', "i'm", 'get', 'going', 'go']

corpus = preprocess_stream(stream_file, punctuation_dict, additional_stopwords=additional_stopwords,
                           stem_words=False, most_common=100)

if mode == "training":
    print("Creating embeddings")
    model = Word2Vec(sentences=corpus, vector_size=100, min_count=1, seed=0, workers=1)  # Set workers=1 to keep training deterministic
    model.save(f"resources/word2vec.model")

elif mode == "loading":
    model = Word2Vec.load("resources/word2vec.model")

print("-"*8 + f"\nModel has a vocabulary size of {len(model.wv)}")
# print(model.wv.most_similar("obama"))

tweet_vectors = average_vectors(tweets_list=corpus, model=model)

