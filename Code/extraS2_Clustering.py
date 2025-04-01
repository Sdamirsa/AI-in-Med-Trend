import os
import numpy as np
import logging
from hdbscan import HDBSCAN

# Set up logging.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("extraS2_Clustering")

def load_embeddings_from_faiss(faiss_input_path: str) -> np.ndarray:
    """
    Loads embeddings from a FAISS index file.

    Parameters:
        faiss_input_path (str): Path to the FAISS index file.

    Returns:
        np.ndarray: Array of embeddings.
    """
    try:
        import faiss
    except ImportError:
        logger.error("faiss library is not installed. Please install it.")
        raise

    logger.info(f"Loading FAISS index from: {faiss_input_path}")
    index = faiss.read_index(faiss_input_path)
    n = index.ntotal
    d = index.d
    logger.info(f"Loaded index with {n} vectors of dimension {d}.")

    # Try extracting embeddings from a flat index.
    try:
        embeddings = faiss.vector_to_array(index).reshape(n, d)
    except AssertionError:
        logger.warning("FAISS index is not a flat vector index; falling back to manual reconstruction.")
        embeddings = np.vstack([index.reconstruct(i) for i in range(n)])
    
    logger.info(f"Extracted embeddings with shape: {embeddings.shape}")
    return embeddings

def save_cluster_labels_as_faiss(cluster_labels, save_path: str):
    """
    Saves a 1D array of cluster labels as a FAISS index.

    Parameters:
        cluster_labels (np.ndarray): 1D array of integer cluster labels.
        save_path (str): File path to save the FAISS index.
    """
    try:
        import faiss
    except ImportError:
        logger.error("faiss library is not installed. Please install it.")
        raise

    # Ensure directory exists.
    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Convert cluster labels to float32 and reshape to (n, 1) for FAISS.
    labels_array = np.array(cluster_labels, dtype=np.float32).reshape(-1, 1)
    d = 1
    index = faiss.IndexFlatIP(d)  # Using inner product on 1D data.
    index.add(labels_array)
    faiss.write_index(index, save_path)
    logger.info(f"Cluster labels saved to FAISS index: {save_path}")

def load_cluster_labels_from_faiss(cluster_save_path: str) -> np.ndarray:
    """
    Loads cluster labels from a FAISS index file and converts them to integers.
    
    Parameters:
        cluster_save_path (str): Path to the FAISS index file with cluster labels.
        
    Returns:
        np.ndarray: 1D array of integer cluster labels.
    """
    try:
        import faiss
    except ImportError:
        logger.error("faiss library is not installed. Please install it.")
        raise

    logger.info(f"Loading cluster labels from FAISS index: {cluster_save_path}")
    index = faiss.read_index(cluster_save_path)
    n = index.ntotal
    d = index.d
    try:
        labels_float = faiss.vector_to_array(index).reshape(n, d)
    except AssertionError:
        logger.warning("FAISS index is not a flat vector index; falling back to manual reconstruction for cluster labels.")
        labels_float = np.vstack([index.reconstruct(i) for i in range(n)])
    cluster_labels = labels_float.astype(np.int32).flatten()
    logger.info(f"Loaded cluster labels with shape: {cluster_labels.shape}")
    return cluster_labels

def extraS2_Clustering_main(faiss_input_path: str,
                            cluster_save_path: str = None,
                            min_cluster_size: int = 50,
                            min_samples: int = None,
                            cluster_selection_epsilon: float = 0.5,
                            alpha: float = 1.0,
                            cluster_selection_method: str = "eom",
                            metric: str = "euclidean") -> np.ndarray:
    """
    Loads embeddings from a FAISS index, clusters them using HDBSCAN (via a scikit-learn-like API),
    and saves the resulting cluster labels as a FAISS index.

    Parameters:
        faiss_input_path (str): Path to the FAISS index file containing embeddings.
        cluster_save_path (str): Path to save cluster labels as a FAISS index.
            Defaults to "Results/extraS2_Clustering/clustering_labels.index".
        min_cluster_size (int): Minimum number of points in a cluster.
        min_samples (int): The k-value for determining point density. When None, defaults to min_cluster_size.
        cluster_selection_epsilon (float): Distance threshold for merging clusters.
        alpha (float): Scaling factor for robust single linkage.
        cluster_selection_method (str): Method to select clusters from the hierarchy ("eom" or "leaf").
        metric (str): Distance metric; for normalized embeddings, "euclidean" is effective.
    
    Returns:
        np.ndarray: Array of cluster labels.
    """
    # Load embeddings from FAISS.
    embeddings = load_embeddings_from_faiss(faiss_input_path)
    
    if min_samples is None:
        min_samples = min_cluster_size

    logger.info("Clustering embeddings using HDBSCAN with parameters:")
    logger.info(f"min_cluster_size: {min_cluster_size}, min_samples: {min_samples}, "
                f"alpha: {alpha}, cluster_selection_epsilon: {cluster_selection_epsilon}, "
                f"metric: {metric}, cluster_selection_method: {cluster_selection_method}")
    
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_epsilon=cluster_selection_epsilon,
                        alpha=alpha,
                        cluster_selection_method=cluster_selection_method,
                        metric=metric)
    
    cluster_labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    logger.info(f"HDBSCAN found {n_clusters} clusters (excluding noise).")
    
    if cluster_save_path is None:
        cluster_save_path = os.path.join("Results", "extraS2_Clustering", "clustering_labels.index")
    save_cluster_labels_as_faiss(cluster_labels, cluster_save_path)
    
    return cluster_labels

def extraS2_Clustering_report(cluster_save_path: str,
                              embeddings_faiss_path: str,
                              cluster_centers_save_path: str = None):
    """
    Reads the cluster labels saved as a FAISS index in cluster_save_path, prints a summary report, 
    plots a histogram of cluster sizes, and computes and saves cluster centers as a FAISS index.

    Parameters:
        cluster_save_path (str): Path to the FAISS index file with cluster labels.
        embeddings_faiss_path (str): Path to the FAISS file with embeddings (used to compute cluster centers).
        cluster_centers_save_path (str): Path to save the FAISS index of cluster centers.
            Defaults to "Results/extraS2_Clustering/cluster_centers.index".
    """
    # Load cluster labels from FAISS.
    cluster_labels = load_cluster_labels_from_faiss(cluster_save_path)
    total_points = len(cluster_labels)
    noise_count = np.sum(cluster_labels == -1)
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    print("=== Clustering Report ===")
    print(f"Total points: {total_points}")
    print(f"Noise points (-1): {noise_count}")
    print(f"Number of clusters (excluding noise): {n_clusters}")
    print("Cluster sizes:")
    for label, size in zip(unique_labels, counts):
        label_name = "Noise" if label == -1 else f"Cluster {label}"
        print(f"  {label_name}: {size}")

    # Plot a histogram of cluster sizes (excluding noise).
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is not installed. Please install it to generate visualizations.")
        raise

    valid_labels = unique_labels[unique_labels != -1]
    sizes = [counts[unique_labels == label][0] for label in valid_labels]
    plt.figure(figsize=(10,6))
    plt.bar(valid_labels, sizes)
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Points")
    plt.title("Cluster Size Distribution (Excluding Noise)")
    results_dir = os.path.join("Results", "extraS2_Clustering")
    os.makedirs(results_dir, exist_ok=True)
    hist_path = os.path.join(results_dir, "cluster_size_histogram.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    logger.info(f"Cluster size histogram saved to: {hist_path}")

    # Compute cluster centers: load embeddings.
    embeddings = load_embeddings_from_faiss(embeddings_faiss_path)
    centers = []
    cluster_ids = []
    for label in unique_labels:
        if label == -1:
            continue
        mask = (cluster_labels == label)
        # Compute the mean embedding for the cluster.
        cluster_center = embeddings[mask].mean(axis=0)
        centers.append(cluster_center)
        cluster_ids.append(label)
    centers = np.vstack(centers)
    
    # Save cluster centers as a FAISS index.
    try:
        import faiss
    except ImportError:
        logger.error("faiss library is not installed. Please install it.")
        raise
    d = centers.shape[1]
    index_centers = faiss.IndexFlatIP(d)
    index_centers.add(centers)
    if cluster_centers_save_path is None:
        cluster_centers_save_path = os.path.join(results_dir, "cluster_centers.index")
    out_dir = os.path.dirname(cluster_centers_save_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index_centers, cluster_centers_save_path)
    logger.info(f"Cluster centers saved as FAISS index to: {cluster_centers_save_path}")

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Cluster embeddings using HDBSCAN and generate a report.")
#     parser.add_argument("--faiss_input_path", type=str, required=True, help="Path to the FAISS index file with embeddings.")
#     parser.add_argument("--cluster_save_path", type=str, default=None, help="Path to save the cluster labels as a FAISS index.")
#     parser.add_argument("--min_cluster_size", type=int, default=50, help="Minimum cluster size for HDBSCAN.")
#     parser.add_argument("--min_samples", type=int, default=None, help="Minimum samples for HDBSCAN (default: min_cluster_size).")
#     parser.add_argument("--cluster_selection_epsilon", type=float, default=0.5, help="Cluster selection epsilon.")
#     parser.add_argument("--alpha", type=float, default=1.0, help="Alpha for robust single linkage.")
#     parser.add_argument("--cluster_selection_method", type=str, default="eom", help="Cluster selection method ('eom' or 'leaf').")
#     parser.add_argument("--metric", type=str, default="euclidean", help="Distance metric.")
#     parser.add_argument("--embeddings_faiss_path", type=str, required=True, help="Path to the FAISS file with embeddings (for computing cluster centers).")
#     parser.add_argument("--cluster_centers_save_path", type=str, default=None, help="Path to save cluster centers as a FAISS index.")
    
#     args = parser.parse_args()
#     labels = extraS2_Clustering_main(faiss_input_path=args.faiss_input_path,
#                                      cluster_save_path=args.cluster_save_path,
#                                      min_cluster_size=args.min_cluster_size,
#                                      min_samples=args.min_samples,
#                                      cluster_selection_epsilon=args.cluster_selection_epsilon,
#                                      alpha=args.alpha,
#                                      cluster_selection_method=args.cluster_selection_method,
#                                      metric=args.metric)
#     extraS2_Clustering_report(cluster_save_path=(args.cluster_save_path if args.cluster_save_path 
#                                                    else os.path.join("Results", "extraS2_Clustering", "clustering_labels.index")),
#                               embeddings_faiss_path=args.embeddings_faiss_path,
#                               cluster_centers_save_path=args.cluster_centers_save_path)
