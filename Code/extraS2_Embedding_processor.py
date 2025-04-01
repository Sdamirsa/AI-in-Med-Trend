import os
import json
import logging
from glob import glob
from tqdm import tqdm
import torch

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extraS2_Embedding_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("S2_prepare_and_label_main")

# Import the embedding generation and saving functions.
from .extraS2_Embedding import generate_embeddings, save_embeddings

def extraS2_Embedding_processor_main(folder_path, filter_startstring="", save_embedding_path="embeddings_faiss.index", batch_size=128, save_format="faiss", model_name=os.getenv("HF_ST_model_for_clustering", "Alibaba-NLP/gte-Qwen2-1.5B-instruct" )):
    """
    Process JSON files from a folder to generate and save embeddings.
    
    This function:
      - Reads all .json files in the given folder (optionally filtered by a start string).
      - For each JSON file, retrieves the list of articles from the "articles" key.
      - For each article, extracts the "llm_input" text.
      - Generates embeddings for all collected texts.
      - Saves the embeddings in the specified format and location.
    
    Parameters:
        folder_path (str): Path to the folder containing JSON files.
        filter_startstring (str): If provided, only process JSON files starting with this string.
        save_embedding_path (str): File path to save the embeddings.
        batch_size (int): Batch size used in embedding generation.
        save_format (str): Format to save embeddings ("faiss" or "npy").
    """
    logger.info(f"Processing folder: {folder_path} with filter: '{filter_startstring}'")
    
    # Construct the file search pattern.
    pattern = os.path.join(folder_path, f"{filter_startstring}*.json") if filter_startstring else os.path.join(folder_path, "*.json")
    json_files = glob(pattern)
    logger.info(f"Found {len(json_files)} JSON files.")
    
    all_texts = []
    
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading {json_file}: {e}")
            continue
        
        if "articles" not in data:
            logger.warning(f"'articles' key not found in {json_file}. Skipping file.")
            continue
        
        articles = data["articles"]
        for article in articles:
            if "llm_input" in article:
                all_texts.append(article["llm_input"])
            else:
                logger.warning(f"'llm_input' not found in an article in {json_file}.")
    
    logger.info(f"Total texts to process: {len(all_texts)}")
    
    # Generate embeddings for the collected texts.
    embeddings = generate_embeddings(all_texts, batch_size=batch_size, model_name=model_name)
    
    # Save the embeddings to disk.
    save_embeddings(embeddings, save_embedding_path, save_format=save_format)
    logger.info("Embedding generation and saving completed.")

if __name__ == "__main__":
    # Command-line interface for processing.
    import argparse
    parser = argparse.ArgumentParser(description="Process JSON files and generate embeddings for clustering.")
    parser.add_argument("--folder_path", type=str, required=True, help="Folder containing JSON files.")
    parser.add_argument("--filter_startstring", type=str, default="", help="Filter JSON files starting with this string.")
    parser.add_argument("--save_embedding_path", type=str, required=True, help="Path to save the embeddings.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for embedding generation.")
    parser.add_argument("--save_format", type=str, choices=["faiss", "npy"], default="faiss", help="Format to save embeddings.")
    parser.add_argument("--model_name", type=str, default=os.getenv("HF_ST_model_for_clustering", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"), help="Model name for embedding generation.")
    
    args = parser.parse_args()
    extraS2_Embedding_processor_main(
        folder_path=args.folder_path,
        filter_startstring=args.filter_startstring,
        save_embedding_path=args.save_embedding_path,
        batch_size=args.batch_size,
        save_format=args.save_format,
        model_name=args.model_name
    )
