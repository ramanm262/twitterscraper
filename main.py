from processing import *
from gensim.models import Word2Vec
from sklearn import metrics
from sklearn.cluster import KMeans

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

# Determine how many clusters maximizes the silhouette coefficient (Rousseeuw 1987)
km_model = KMeans(n_init="auto")
silhouette_coeffs = []
cluster_range = range(2, 15)
for n_clusters in tqdm.tqdm(cluster_range, desc="Finding best number of clusters"):
    km_model.set_params(n_clusters=n_clusters)
    km_model.fit(tweet_vectors)
    silhouette_coeffs.append(metrics.silhouette_score(tweet_vectors, km_model.labels_, sample_size=int(1e4)))  # Set sample_size=None if you have lots of processing power

plt.figure()
plt.bar(cluster_range, silhouette_coeffs)
plt.xticks(list(cluster_range), list(cluster_range))
plt.title("Optimization: silhouette scores per number of K-Means clusters")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette coefficient")
plt.savefig("outputs/silhouette_scores.png")

n_clusters = cluster_range[silhouette_coeffs.index(np.max(silhouette_coeffs))]

