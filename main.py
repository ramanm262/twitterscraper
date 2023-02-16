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
tweet_strings = [" ".join(tweet) for tweet in corpus]

if mode == "training":
    print("Creating embeddings")
    model = Word2Vec(sentences=corpus, vector_size=100, min_count=1, seed=0, workers=1)  # Set workers=1 to keep training deterministic
    model.save(f"resources/word2vec.model")

elif mode == "loading":
    model = Word2Vec.load("resources/word2vec.model")

print("-"*8 + f"\nModel has a vocabulary size of {len(model.wv)}")
# print(model.wv.most_similar("obama"))

tweet_vectors = average_vectors(tweets_list=corpus, model=model)

# Determine the best number of clusters to use
n_clusters = optimum_n_clusters(tweet_vectors, cluster_range=range(3, 15), save_silhouette=True)
# n_clusters = 10  # Or set it manually

# Instantiate and fit KMeans model
km_model = KMeans(n_clusters=n_clusters, n_init="auto")
km_model.fit(tweet_vectors)
cluster_assignments = km_model.labels_

# Find exemplar tokens
for cluster in range(n_clusters):
    exemplars = ""
    exemplar_list = model.wv.most_similar(positive=[km_model.cluster_centers_[cluster]], topn=5)
    for token in exemplar_list:
        exemplars += f"{token[0]} "
    print(f"Cluster {cluster}: ", exemplars)

# Find exemplar tweets
for cluster in range(n_clusters):
    print(f"\nCluster {cluster}:")
    exemplars = np.argsort(np.linalg.norm(tweet_vectors - km_model.cluster_centers_[cluster], axis=1))
    for exemplar in exemplars[:5]:
        print(tweet_strings[exemplar])
    print("-"*16)
