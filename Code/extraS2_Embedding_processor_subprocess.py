import os
import subprocess
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extraS2_Embedding_processor_subprocess_main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("extraS2_Embedding_processor_subprocess_main")

# Import the embedding processor main function.
# (This import ensures the module is accessible and verifies its interface.)
from .extraS2_Embedding_processor import extraS2_Embedding_processor_main

def extraS2_Embedding_processor_subprocess_main(
    folder_path,
    filter_startstring="",
    save_embedding_path="embeddings_faiss.index",
    batch_size=128,
    save_format="faiss",
    model_name=os.getenv("HF_ST_model_for_clustering", "Alibaba-NLP/gte-Qwen2-1.5B-instruct" )
):
    """
    This function retrieves the python executable path from the environment variable 
    'Path_to_venv_torch_for_subprocess_call' (with a default fallback) and calls the 
    extraS2_Embedding_processor_main via a subprocess call, passing the given parameters.

    Parameters:
        folder_path (str): Path to the folder containing JSON files to be processed.
        filter_startstring (str): (Optional) Filter for JSON files starting with this string.
        save_embedding_path (str): (Optional) Path to save the generated embeddings.
        batch_size (int): (Optional) Batch size for embedding generation.
        save_format (str): (Optional) Format to save embeddings ("faiss" or "npy").
    """
    # Get the python executable path from environment variable.
    python_executable = os.getenv("Path_to_venv_torch_for_subprocess_call",
                                    r"C:\Users\LEGION\Documents\GIT\AI-in-Med-Trend\venv_torch\Scripts\python.exe")
    
    # Build the command to execute the embedding processor.
    # Note: The module is referenced as part of the 'Code' package.
    command = [
        python_executable,
        "-m", "Code.extraS2_Embedding_processor",
        "--folder_path", folder_path,
        "--filter_startstring", filter_startstring,
        "--save_embedding_path", save_embedding_path,
        "--batch_size", str(batch_size),
        "--save_format", save_format,
        "--model_name", model_name
    ]
    
    logger.info(f"Executing command: {' '.join(command)}")
    
    # Run the command as a subprocess.
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Subprocess failed with error:\n{result.stderr}")
    else:
        logger.info(f"Subprocess completed successfully:\n{result.stdout}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run extraS2_Embedding_processor_main in a subprocess using a specified python executable."
    )
    parser.add_argument("--folder_path", type=str, required=True, help="Folder path containing JSON files to process.")
    parser.add_argument("--filter_startstring", type=str, default="", help="Filter JSON files starting with this string.")
    parser.add_argument("--save_embedding_path", type=str, default="embeddings_faiss.index", help="Path to save the embeddings.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for embedding generation.")
    parser.add_argument("--save_format", type=str, choices=["faiss", "npy"], default="faiss", help="Format to save embeddings.")
    parser.add_argument("--model_name", type=str, default=os.getenv("HF_ST_model_for_clustering", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"), help="Model name for embedding generation.") 
    
    args = parser.parse_args()
    extraS2_Embedding_processor_subprocess_main(
        folder_path=args.folder_path,
        filter_startstring=args.filter_startstring,
        save_embedding_path=args.save_embedding_path,
        batch_size=args.batch_size,
        save_format=args.save_format,
        model_name=args.model_name
    )
