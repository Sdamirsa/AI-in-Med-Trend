"""
PubMed XML Batch Cleaner

This script processes all PubMed batch files in the cache directory,
extracts and cleans the XML data, and saves the cleaned data with
a "cleaned_" prefix in the same directory.

Enviroment variables used: COMBINE_ALL_AFTER_S2
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any
from Code.S2_Pubmed_XML_Cleaner import PubMedXMLCleaner

# Load environment variables
# load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("S2_Pubmed_XML_Cleaner_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("S2_Cleaner_processor_main")
def create_and_copy_folder(source_name, destination_folder):
    """
    Creates a destination folder and copies contents from a source folder,
    only transferring files that do not already exist in the destination.
    
    Args:
        source_name: The name of the source folder
        destination_folder: The path to the destination folder
    
    Returns:
        Path: The path to the destination folder
    """
    import os
    import shutil
    from pathlib import Path
    
    # Turning source_name into a Path object
    source_folder = Path(source_name)
    destination_folder = Path(destination_folder)
    
    # Check if the destination is inside the source
    is_subdirectory = False
    try:
        is_subdirectory = destination_folder.resolve().is_relative_to(source_folder.resolve())
    except Exception:
        # For Python < 3.9 compatibility or other path resolution errors
        pass
    
    # Create the destination directory if it doesn't exist
    if not destination_folder.exists():
        destination_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created destination directory: {destination_folder}")
    
    # Copy contents from source to destination
    if source_folder.exists():
        for item in source_folder.glob('*'):
            # Skip if the item is the destination folder
            if is_subdirectory and destination_folder.name == item.name:
                logger.info(f"Skipping destination folder: {item}")
                continue
                
            dest_item = destination_folder / item.name
            if item.is_file() and not dest_item.exists():
                shutil.copy2(item, dest_item)
                logger.info(f"Copied file {item} to {dest_item}")
            elif item.is_dir() and not dest_item.exists():
                shutil.copytree(item, dest_item, dirs_exist_ok=True)
                logger.info(f"Copied directory {item} to {dest_item}")
    else:
        logger.warning(f"Source directory {source_folder} does not exist")
    
    return destination_folder

def find_batch_files(cache_dir: str) -> List[str]:
    """
    Find all PubMed batch files in the cache directory.
    
    Args:
        cache_dir: Path to the cache directory
        
    Returns:
        List of file paths
    """
    if not os.path.exists(cache_dir):
        logger.warning(f"Cache directory {cache_dir} does not exist")
        return []
    
    batch_files = []
    for file in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, file)
        if (file.startswith("pubmed_batch") and 
            file.endswith(".json") and 
            os.path.isfile(file_path) and
            not os.path.basename(file).startswith("cleaned_")):
            batch_files.append(file_path)
    
    logger.info(f"Found {len(batch_files)} batch files in {cache_dir}")
    return batch_files

def process_batch_file(file_path: str, cleaner: PubMedXMLCleaner) -> str:
    """
    Process a single batch file and save the cleaned data.
    
    Args:
        file_path: Path to the batch file
        cleaner: PubMedXMLCleaner instance
        
    Returns:
        Path to the cleaned file or empty string if failed
    """
    try:
        # Get directory and filename
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        cleaned_filename = f"cleaned_{filename}"
        cleaned_path = os.path.join(directory, cleaned_filename)
        
        # Check if cleaned file already exists
        if os.path.exists(cleaned_path):
            logger.info(f"Cleaned file already exists: {cleaned_path}")
            return cleaned_path
        
        # Load the batch file
        with open(file_path, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        # Extract XML data
        if 'data' not in batch_data:
            logger.warning(f"No 'data' field found in {file_path}")
            return ""
        
        xml_data = batch_data['data']
        
        # Parse the XML data
        articles = cleaner.parse_xml_string(xml_data)
        
        # Add source information
        for article in articles:
            article['source_file'] = filename
            article['batch_id'] = batch_data.get('batch_id', '')
            article['query'] = batch_data.get('query', '')
        
        # Create cleaned data structure
        cleaned_data = {
            "original_metadata": {
                "query": batch_data.get('query', ''),
                "batch_id": batch_data.get('batch_id', ''),
                "timestamp": batch_data.get('timestamp', ''),
                "total_count": batch_data.get('total_count', 0),
                "retstart": batch_data.get('retstart', 0),
                "retmax": batch_data.get('retmax', 0)
            },
            "articles": articles,
            "article_count": len(articles)
        }
        
        # Save the cleaned data
        with open(cleaned_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processed {file_path} -> {cleaned_path} ({len(articles)} articles)")
        return cleaned_path
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}", exc_info=True)
        return ""


def combine_cleaned_files(cache_dir: str, file_paths: List[str]) -> str:
    """
    Combine all cleaned files into one large file.
    
    Args:
        cache_dir: Path to the cache directory
        file_paths: List of cleaned file paths
        
    Returns:
        Path to the combined file
    """
    try:
        all_articles = []
        pmid_set = set()  # To avoid duplicates
        
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'articles' in data:
                for article in data['articles']:
                    if 'pmid' in article and article['pmid'] not in pmid_set:
                        pmid_set.add(article['pmid'])
                        all_articles.append(article)
        
        # Save the combined data
        combined_path = os.path.join(cache_dir, "all_cleaned_articles.json")
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump({
                "articles": all_articles,
                "article_count": len(all_articles),
                "source_files": len(file_paths)
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Combined {len(all_articles)} unique articles from {len(file_paths)} files into {combined_path}")
        return combined_path
        
    except Exception as e:
        logger.error(f"Error combining files: {e}", exc_info=True)
        return ""
    
    
def S2_Cleaner_processor_main(data_dir, combine_all = os.getenv("COMBINE_ALL_AFTER_S2", "false").lower() == "true"):
    """Process all PubMed batch files in the cache directory."""
    
    if combine_all:
        logger.info(f"The COMBINE_ALL_AFTER_S2 set to True. Will combine all jsons into one, as well as saving the cleaned output in each json.")
    try:  
        # Create the directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Find all batch files
        batch_files = find_batch_files(data_dir)
        if not batch_files:
            logger.warning("No batch files found to process")
            return
        
        # Initialize the cleaner
        cleaner = PubMedXMLCleaner(data_dir=data_dir)
        
        # Process each batch file
        processed_files = []
        for file_path in batch_files:
            cleaned_path = process_batch_file(file_path, cleaner)
            if cleaned_path:
                processed_files.append(cleaned_path)
        
        logger.info(f"Successfully processed {len(processed_files)} of {len(batch_files)} batch files")
        
        # Optionally combine all cleaned data into one file
        if combine_all and processed_files:
            combine_cleaned_files(data_dir, processed_files)
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
