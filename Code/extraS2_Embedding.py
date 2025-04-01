import os
import json
import logging
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

# Determine the appropriate tqdm import based on the execution environment
from IPython import get_ipython
try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':  # Jupyter Notebook or qtconsole
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extraS2_Embedding.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("extraS2_Embedding")

def load_model_and_tokenizer(model_name):
    """
    Loads the tokenizer, model, and determines the computation device.
    Returns:
        tokenizer: The Hugging Face tokenizer.
        model: The Hugging Face model (set to eval mode).
        device: The torch.device used by the model.
    """
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
    model.eval()
    device = next(model.parameters()).device
    return tokenizer, model, device

def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on the token embeddings.
    
    Parameters:
        model_output: Model output containing the last hidden state.
        attention_mask: Attention mask tensor.
        
    Returns:
        Tensor: Mean pooled embeddings.
    """
    token_embeddings = model_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def encode_batch(texts, model_name):
    """
    Encode a batch of texts into embeddings using mean pooling.
    
    Parameters:
        texts (list[str]): List of text strings.
        
    Returns:
        np.ndarray: Array of normalized embeddings.
    """
    tokenizer, model, device = load_model_and_tokenizer(model_name)
    max_length = int(os.getenv("HF_ST_MaxToken_for_clustering", "512"))
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = mean_pooling(outputs, inputs['attention_mask'])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

def generate_embeddings(texts, batch_size,model_name):
    """
    Generate embeddings for a list of texts in batches.
    
    Parameters:
        texts (list[str]): List of texts to embed.
        batch_size (int): Batch size for processing.
        
    Returns:
        np.ndarray: Array of embeddings.
    """
    logger.info("Starting embedding generation...")
    embeddings_list = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = encode_batch(batch_texts, model_name)
        embeddings_list.append(batch_embeddings)
        if i % (batch_size * 10) == 0:
            logger.info(f"Processed {i} / {len(texts)} texts")
    embeddings = np.vstack(embeddings_list)
    logger.info("Completed embedding generation.")
    return embeddings

def save_embeddings(embeddings, save_path, save_format="faiss"):
    """
    Save embeddings to disk.
    
    Parameters:
        embeddings (np.ndarray): Array of embeddings.
        save_path (str): File path to save the embeddings.
        save_format (str): 'faiss' to save as a FAISS index, 'npy' to save as a NumPy file.
    """
    logger.info(f"Saving embeddings using format: {save_format}")
    if save_format == "faiss":
        try:
            import faiss
        except ImportError:
            logger.error("faiss library is not installed. Please install it or use save_format='npy'.")
            raise
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)  # Using inner product on normalized vectors approximates cosine similarity.
        index.add(embeddings)
        faiss.write_index(index, save_path)
        logger.info(f"FAISS index saved to {save_path}")
    elif save_format == "npy":
        np.save(save_path, embeddings)
        logger.info(f"Numpy embeddings saved to {save_path}")
    else:
        logger.error("Unsupported save format. Use 'faiss' or 'npy'.")
        raise ValueError("Unsupported save format. Use 'faiss' or 'npy'.")
