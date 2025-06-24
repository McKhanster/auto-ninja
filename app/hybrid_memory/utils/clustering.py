from sklearn.cluster import KMeans

def cluster_embeddings(embeddings: list[list[float]], n_clusters: int) -> list[int]:
    if len(embeddings) < n_clusters:
        n_clusters = max(1, len(embeddings))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels.tolist()