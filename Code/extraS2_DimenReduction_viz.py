import os
import numpy as np
import faiss
import umap
import umap.plot
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger("extraS2_UMAP_visualization")
logger.setLevel(logging.INFO)

def extraS2_DimenReduction_viz_main(faiss_path: str, label_for_figure: str = "afterUMAP") -> tuple:
    """
    Loads the FAISS index containing reduced embeddings, further reduces them to 2D for visualization
    using UMAP, generates several diagnostic plots using umap.plot, and saves the visualizations in
    the "Results/extraS2_DimenReduction/" folder.

    Parameters:
        faiss_path (str): Path to the FAISS file with reduced embeddings (e.g. os.getenv("Save_FAISS_DimenReduction_Path")).

    Returns:
        mapper: The UMAP model fitted to the embeddings.
        embeddings: The loaded (and possibly reduced) embeddings from the FAISS index.
    """
    # Create results folder if it doesn't exist.
    results_dir = os.path.join("Results", "extraS2_DimenReduction")
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved in: {results_dir}")
    
    # Load the FAISS index.
    logger.info(f"Loading FAISS index from: {faiss_path}")
    index = faiss.read_index(faiss_path)
    n = index.ntotal
    d = index.d
    logger.info(f"Loaded index with {n} vectors of dimension {d}.")
    
    # Extract embeddings from the FAISS index.
    try:
        embeddings = faiss.vector_to_array(index).reshape(n, d)
    except AssertionError:
        logger.warning("FAISS index is not a flat vector index; falling back to manual reconstruction.")
        embeddings = np.vstack([index.reconstruct(i) for i in range(n)])
    logger.info(f"Extracted embeddings with shape: {embeddings.shape}")
    
    # Further reduce embeddings to 2 dimensions for visualization.
    viz_umap = umap.UMAP(n_components=2, metric="cosine", min_dist=0.1, n_neighbors=30)
    mapper = viz_umap.fit(embeddings)
    logger.info("UMAP 2D reduction complete for visualization.")
    
    # Generate a scatter plot of the UMAP results.
    fig_points = umap.plot.points(mapper, show_legend=False)
    points_path = os.path.join(results_dir, f"{label_for_figure}_points.png")
    fig_points.figure.savefig(points_path, dpi=300)
    logger.info(f"UMAP points plot saved to: {points_path}")
    
    # Generate a diagnostic plot (PCA-based diagnostic) of the UMAP embedding.
    fig_diag = umap.plot.diagnostic(mapper, diagnostic_type='pca')
    diag_path = os.path.join(results_dir, f"{label_for_figure}_diagnostic_pca.png")
    fig_diag.figure.savefig(diag_path, dpi=300)
    logger.info(f"UMAP diagnostic (PCA) plot saved to: {diag_path}")
    
    # Generate a connectivity plot to visualize the topological connectivity.
    fig_connect = umap.plot.connectivity(mapper, show_points=True)
    connect_path = os.path.join(results_dir, f"{label_for_figure}_connectivity.png")
    fig_connect.figure.savefig(connect_path, dpi=300)
    logger.info(f"UMAP connectivity plot saved to: {connect_path}")
    
    # Optionally, display the plots (especially useful when running in a Jupyter notebook).
    plt.show()
    
    return mapper, embeddings

# Example usage:
# from Code import extraS2_UMAP_visualization
# viz_mapper, viz_embeddings = extraS2_UMAP_visualization(os.getenv("Save_FAISS_DimenReduction_Path"))
