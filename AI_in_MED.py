"""
PubMed data retrieval script.

This script demonstrates how to use the PubMedRetriever class to download
data from PubMed in optimized batches.
"""

#%pip install requests pandas eutils python-dotenv

from Code import PubMedRetriever, create_env_template
import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Example usage of the PubMedRetriever class."""
    # Check for .env file and create template if needed
    if not os.path.exists('.env'):
        logger.warning("No .env file found")
        create_env_template()
        logger.info("Please fill .env.template with your information and rename to .env")
        return
        
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize the retriever
        retriever = PubMedRetriever()
        
        # Example query
        query = r'"Artificial Intelligence"[Mesh]'
        start_date = "2000/01/01"
        end_date = "2025/03/01"
        
        logger.info(f"Retrieving data for query: {query} from {start_date} to {end_date}")
        
        # Retrieve data using date-based batching
        results = retriever.retrieve_date_batched_query(
            base_query=query,
            start_date=start_date,
            end_date=end_date,
            batch_size=500
        )
        
        logger.info(f"Retrieved {results['total_records']} records in {results['total_batches']} batches")
        
        # Extract and save structured article data
        articles = retriever.extract_articles_from_files()
        retriever.save_articles_to_json(articles, "ai_articles.json")
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)

if __name__ == "__main__":
    main()