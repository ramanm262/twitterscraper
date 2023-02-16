from sklearn import metrics
from sklearn.cluster import KMeans
import tqdm
import matplotlib.pyplot as plt


def optimum_n_clusters(tweet_vectors, cluster_range=range(2, 15), save_silhouette=True):
    """
    Function to determine how many clusters maximizes the silhouette coefficient (Rousseeuw 1987). Also plots the
    various coefficients that were determine for different numbers of clusters.
    :param tweet_vectors: List of list of floats. Each sub-list is the vector corresponding to a single tweet. Ideally,
    this comes straight from the output of `processing.average_vectors()`.
    :param cluster_range: List-like of ints. Each element is a number of clusters to test to see if that configuration
    has a high silhouette score. Elements with high values may make the routine take quite a while to run.
    :param save_silhouette: Bool. If true, will plot and save a silhouette coefficient bar chart to `outputs/`.
    :return: n_clusters: The optimum number of clusters; i.e. the number of clusters from within `cluster_range` that
    is associated with the highest silhouette coefficient.
    """
    km_model = KMeans(n_init="auto")
    silhouette_coeffs = []
    for n_clusters in tqdm.tqdm(cluster_range, desc="Finding optimum number of clusters"):
        km_model.set_params(n_clusters=n_clusters)
        km_model.fit(tweet_vectors)
        silhouette_coeffs.append(metrics.silhouette_score(tweet_vectors, km_model.labels_, sample_size=int(1e4)))  # Set sample_size=None if you have lots of processing power

    if save_silhouette:
        plt.figure()
        plt.bar(cluster_range, silhouette_coeffs)
        plt.xticks(list(cluster_range), list(cluster_range))
        plt.title("Optimization: silhouette scores per number of K-Means clusters")
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette coefficient")
        plt.savefig("outputs/silhouette_scores.png")

    n_clusters = cluster_range[silhouette_coeffs.index(max(silhouette_coeffs))]

    return n_clusters
