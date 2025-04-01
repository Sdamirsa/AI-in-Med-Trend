import os
import numpy as np
import umap
import logging

logger = logging.getLogger("extraS2_DimenReduction")

def save_embeddings(embeddings, save_path, save_format="faiss"):
    logger.info(f"Saving embeddings using format: {save_format}")
    if save_format == "faiss":
        try:
            import faiss
        except ImportError:
            logger.error("faiss library is not installed. Please install it or use save_format='npy'.")
            raise
        
        # Ensure the directory exists.
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        faiss.write_index(index, save_path)
        logger.info(f"FAISS index saved to {save_path}")
    elif save_format == "npy":
        # Ensure the directory exists.
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        np.save(save_path, embeddings)
        logger.info(f"Numpy embeddings saved to {save_path}")
    else:
        logger.error("Unsupported save format. Use 'faiss' or 'npy'.")
        raise ValueError("Unsupported save format. Use 'faiss' or 'npy'.")


def extraS2_DimenReduction_main(faiss_input_path: str,
                            faiss_output_path: str = None,
                            umap_metric: str = None,
                            umap_min_dist = None,
                            umap_n_components = None,
                            umap_n_neighbors = None):
    """
    Loads an existing FAISS index file containing embeddings (saved with your generate_embeddings/FAISS method),
    reduces the dimensionality using UMAP, and saves the reduced embeddings to a new FAISS index file.
    
    Parameters:
        faiss_input_path (str): Path to the existing FAISS file with embeddings.
        faiss_output_path (str): Path to save the reduced FAISS index.
            Defaults to os.getenv("Save_FAISS_DimenReduction_Path").
        umap_metric (str): (Optional) Metric to use for UMAP. Defaults to os.getenv("UMAP_METRIC", "cosine").
        umap_min_dist: (Optional) Minimum distance for UMAP.
            Defaults to os.getenv("UMAP_MIN_DIST", "0.07").
        umap_n_components: (Optional) Number of dimensions in the reduced embedding.
            Defaults to os.getenv("UMAP_N_COMPONENTS", "48").
        umap_n_neighbors: (Optional) Number of neighbors for UMAP.
            Defaults to os.getenv("UMAP_N_NEIGHBORS", "30").
    
    Returns:
        np.ndarray: The reduced embeddings.
    """
    # Set default output path if not provided.
    if faiss_output_path is None:
        faiss_output_path = os.getenv("Save_FAISS_DimenReduction_Path", "faiss_embeddings_dimenreduction.index")
    
    # Get UMAP parameters from environment if not provided.
    if umap_metric is None:
        umap_metric = os.getenv("UMAP_METRIC", "cosine")
    
    # Convert parameters to the appropriate types if they are not already numeric.
    if umap_min_dist is None:
        umap_min_dist = float(os.getenv("UMAP_MIN_DIST", "0.07"))
    else:
        umap_min_dist = float(umap_min_dist)
        
    if umap_n_components is None:
        umap_n_components = int(os.getenv("UMAP_N_COMPONENTS", "48"))
    else:
        umap_n_components = int(umap_n_components)
        
    if umap_n_neighbors is None:
        umap_n_neighbors = int(os.getenv("UMAP_N_NEIGHBORS", "30"))
    else:
        umap_n_neighbors = int(umap_n_neighbors)
    
    logger.info(f"Loading FAISS index from: {faiss_input_path}")
    
    try:
        import faiss
    except ImportError:
        logger.error("faiss library is not installed. Please install it or use an alternative save format.")
        raise
    
    # Load the FAISS index.
    index = faiss.read_index(faiss_input_path)
    n = index.ntotal
    d = index.d
    logger.info(f"Loaded index with {n} vectors of dimension {d}.")
    
    # Extract embeddings; if the index is flat, this should work:
    try:
        embeddings = faiss.vector_to_array(index).reshape(n, d)
    except AssertionError:
        logger.warning("FAISS index is not a flat vector index; falling back to manual reconstruction.")
        embeddings = np.vstack([index.reconstruct(i) for i in range(n)])
    
    logger.info(f"Extracted embeddings with shape: {embeddings.shape}")
    
    # Perform UMAP reduction.
    logger.info(f"Performing UMAP reduction with parameters: metric={umap_metric}, min_dist={umap_min_dist}, "
                f"n_components={umap_n_components}, n_neighbors={umap_n_neighbors}.")
    reducer = umap.UMAP(
        metric=umap_metric,
        min_dist=umap_min_dist,
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors
    )
    reduced_embeddings = reducer.fit_transform(embeddings)
    reduced_embeddings = np.asarray(reduced_embeddings, dtype=np.float32)
    logger.info(f"UMAP reduction complete. New shape: {reduced_embeddings.shape}")
    
    # Save the reduced embeddings to a new FAISS index.
    save_embeddings(reduced_embeddings, faiss_output_path, save_format="faiss")
    logger.info(f"Reduced embeddings saved to: {faiss_output_path}")
    
    return reduced_embeddings