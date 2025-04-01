"""
PubMed API Data Retriever

This module provides functionality to query the PubMed API, retrieve data in batches,
and save the results as JSON files. It includes features for incremental updates and
efficient handling of large result sets.

Version 3.0: Complete refactoring with professional Python standards
"""

import json
import logging
import os
import re
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Set

import requests
import urllib.parse

from dotenv import load_dotenv
# Load environment variables from .env file
# load_dotenv()

# Determine the appropriate tqdm import based on the execution environment
from IPython import get_ipython
try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':  # Jupyter Notebook or qtconsole
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    # Standard Python interpreter
    from tqdm import tqdm
    
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("S1_DownloadPubmed_main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("S1_DownloadPubmed_main")

# Constants
BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
MAX_RESULTS_PER_REQUEST = 9900  # Slightly below the 10000 limit to be safe
DEFAULT_DELAY = float(os.getenv('API_DELAY', '0.5'))  # Default delay between API requests in seconds


class PubMedRetriever:
    """Class for retrieving data from PubMed via the NCBI E-utilities API."""

    def __init__(self, 
                 email: Optional[str] = None,
                 api_key: Optional[str] = None,
                 db: str = 'pubmed',
                 delay: Optional[float] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize the PubMed retriever.
        
        Args:
            email: Email for NCBI identification (required by their API)
            api_key: NCBI API key for higher rate limits
            db: The NCBI database to query
            delay: Delay between API requests in seconds
            cache_dir: Directory to store cache files
        """
        # Load from environment variables if not provided
        self.email = email or os.getenv('PUBMED_EMAIL')
        self.api_key = api_key or os.getenv('PUBMED_API_KEY')
        self.db = db
        self.delay = delay if delay is not None else DEFAULT_DELAY
        self.cache_dir = cache_dir or os.getenv('CACHE_DIRECTORY', 'pubmed_data')
        self.last_request_time = 0
        
        # Validate required parameters
        if not self.email:
            raise ValueError("Email is required. Provide it as a parameter or set PUBMED_EMAIL in .env file")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Initialized PubMedRetriever with email: {self.email}, API key: {'Set' if self.api_key else 'Not set'}")
        
        # Valid return modes and types for different databases
        self.valid_return_formats = {
            'pubmed': {
                ('', 'xml'): True,
                ('medline', 'text'): True,
                ('uilist', 'text'): True,
                ('abstract', 'text'): True,
                ('gb', 'text'): True,
                ('gb', 'xml'): True,
                ('gbc', 'xml'): True,
                ('ft', 'text'): True
            },
            # Additional database formats can be added here as needed
        }
    
    def _respect_rate_limit(self) -> None:
        """Ensure we don't exceed NCBI's rate limit."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()
    
    def format_query(self, query: str) -> str:
        """
        Format a query string for PubMed search.
        
        Args:
            query: The raw query string
            
        Returns:
            A properly formatted query string for the API
        """
        # Use urllib's quote_plus for proper URL encoding
        formatted_query = urllib.parse.quote_plus(query)
        logger.info(f"Formatted query: {formatted_query}")
        return formatted_query

    def validate_return_format(self, ret_mode: str, ret_type: str) -> Tuple[str, str]:
        """
        Validate return mode and type for the current database.
        
        Args:
            ret_mode: The return mode
            ret_type: The return type
            
        Returns:
            Tuple of validated return mode and type
        """
        ret_mode = ret_mode.lower()
        
        # Check if we have validation rules for this database
        if self.db in self.valid_return_formats:
            if (ret_type, ret_mode) in self.valid_return_formats[self.db]:
                return ret_mode, ret_type
            else:
                valid_formats = list(self.valid_return_formats[self.db].keys())
                logger.warning(f"Invalid format. Valid formats for {self.db}: {valid_formats}")
                # Use default format for this database
                default_format = next(iter(self.valid_return_formats[self.db].keys()))
                logger.info(f"Using default format: {default_format}")
                return default_format[1], default_format[0]
        else:
            logger.warning(f"No validation rules for database '{self.db}'. Using provided format.")
            return ret_mode, ret_type

    def search(self, query: str) -> Tuple[str, str, int]:
        """
        Perform a search using NCBI's esearch utility.
        
        Args:
            query: The search query
            
        Returns:
            Tuple containing WebEnv, QueryKey, and count of results
        """
        self._respect_rate_limit()
        
        formatted_query = self.format_query(query)
        url = f"{BASE_URL}esearch.fcgi?db={self.db}&term={formatted_query}&usehistory=y&email={self.email}"
        
        # Include API key if provided
        if self.api_key:
            url += f"&api_key={self.api_key}"
        
        logger.info(f"Searching with URL: {url}")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse XML response
            content = response.text
            web_env = re.search(r'<WebEnv>(\S+)</WebEnv>', content).group(1)
            query_key = re.search(r'<QueryKey>(\d+)</QueryKey>', content).group(1)
            count = int(re.search(r'<Count>(\d+)</Count>', content).group(1))
            
            logger.info(f"Search returned {count} results")
            return web_env, query_key, count
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Search request failed: {e}")
            raise
        except (AttributeError, ValueError) as e:
            logger.error(f"Failed to parse search response: {e}")
            raise ValueError("Invalid response from NCBI API") from e

    def fetch_batch(self, 
                   web_env: str, 
                   query_key: str, 
                   retstart: int, 
                   retmax: int,
                   download_mode: str = 'full',
                   ret_mode: str = 'xml',
                   ret_type: str = '') -> str:
        """
        Fetch a batch of results using NCBI's efetch or esummary utility.
        
        Args:
            web_env: WebEnv value from search
            query_key: QueryKey value from search
            retstart: Starting index for retrieval
            retmax: Maximum number of records to retrieve
            download_mode: 'full' for efetch or 'summary' for esummary
            ret_mode: Return mode (xml, text)
            ret_type: Return type (varies by database)
            
        Returns:
            The response text from the API
        """
        self._respect_rate_limit()
        
        # Validate parameters
        if download_mode not in ['full', 'summary']:
            raise ValueError("Invalid download_mode. Choose 'full' or 'summary'.")
        
        ret_mode, ret_type = self.validate_return_format(ret_mode, ret_type)
        
        # Build the URL
        if download_mode == 'summary':
            url = f"{BASE_URL}esummary.fcgi?db={self.db}&query_key={query_key}&WebEnv={web_env}&retstart={retstart}&retmax={retmax}&email={self.email}"
        else:  # full
            url = f"{BASE_URL}efetch.fcgi?db={self.db}&query_key={query_key}&WebEnv={web_env}&retstart={retstart}&retmax={retmax}&email={self.email}"
            if ret_mode:
                url += f"&retmode={ret_mode}"
            if ret_type:
                url += f"&rettype={ret_type}"
        
        # Include API key if provided
        if self.api_key:
            url += f"&api_key={self.api_key}"
        
        logger.info(f"Fetching batch from {retstart} to {retstart+retmax-1}")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Fetch request failed: {e}")
            raise

    def generate_monthly_date_ranges(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Generate monthly date ranges between start and end dates.
        
        Args:
            start_date: Start date in format "YYYY/MM/DD"
            end_date: End date in format "YYYY/MM/DD"
            
        Returns:
            List of dictionaries containing date range information
        """
        # Convert string dates to datetime objects
        start = datetime.strptime(start_date, "%Y/%m/%d")
        end = datetime.strptime(end_date, "%Y/%m/%d")

        monthly_ranges = []
        current = start

        while current <= end:
            year = current.year
            month = current.month
            
            # Calculate the last day of the current month
            if month == 12:
                next_month = datetime(year + 1, 1, 1)
            else:
                next_month = datetime(year, month + 1, 1)
            
            last_day = next_month - timedelta(days=1)
            
            # Format the search term and create range info
            first_day_str = f"{year}/{month:02d}/01"
            last_day_str = f"{year}/{month:02d}/{last_day.day:02d}"
            
            range_info = {
                "start_date": first_day_str,
                "end_date": last_day_str,
                "query_fragment": f'"{first_day_str}"[Date - Publication] : "{last_day_str}"[Date - Publication]',
                "month_year": f"{year}-{month:02d}"
            }
            
            monthly_ranges.append(range_info)
            
            # Move to the first day of the next month
            current = next_month

        return monthly_ranges

    def get_monthly_counts(self, 
                          query: str, 
                          date_ranges: List[Dict[str, Any]],
                          cache_file: Optional[str] = None) -> Dict[str, int]:
        """
        Get the count of results for each monthly date range.
        
        Args:
            query: The base query
            date_ranges: List of date range dictionaries
            cache_file: Path to cache file for counts
            
        Returns:
            Dictionary mapping date range query fragments to result counts
        """
        # Generate cache filename if not provided
        if cache_file is None:
            safe_query = re.sub(r'[^\w]', '_', query)[:30]  # Create safe filename
            cache_file = os.path.join(self.cache_dir, f"{safe_query}_counts.json")
        else:
            # Ensure the cache file is in the cache directory
            cache_file = os.path.join(self.cache_dir, os.path.basename(cache_file))
        
        # Try to load existing counts from cache
        monthly_counts = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    monthly_counts = json.load(f)
                logger.info(f"Loaded {len(monthly_counts)} cached monthly counts")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load counts cache: {e}")
        
        # Get counts for any missing months
        for date_range in date_ranges:
            query_fragment = date_range["query_fragment"]
            
            if query_fragment in monthly_counts:
                logger.info(f"Using cached count for {date_range['month_year']}: {monthly_counts[query_fragment]}")
                continue
                
            try:
                full_query = f"({query}) AND ({query_fragment})"
                _, _, count = self.search(full_query)
                monthly_counts[query_fragment] = count
                logger.info(f"Month {date_range['month_year']}: {count} results")
                
                # Save after each successful query to avoid losing progress
                with open(cache_file, 'w') as f:
                    json.dump(monthly_counts, f, indent=2)
                
            except Exception as e:
                logger.error(f"Failed to get count for {date_range['month_year']}: {e}")
                monthly_counts[query_fragment] = 0
        
        return monthly_counts

    def create_optimized_queries(self, 
                               base_query: str, 
                               date_ranges: List[Dict[str, Any]], 
                               monthly_counts: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        Create optimized queries by combining date ranges to stay under API limits.
        
        Args:
            base_query: The base query without date restrictions
            date_ranges: List of date range dictionaries
            monthly_counts: Dictionary of counts for each date range
            
        Returns:
            List of optimized query dictionaries
        """
        optimized_queries = []
        current_batch = []
        current_count = 0
        batch_start_idx = 0
        
        for i, date_range in enumerate(date_ranges):
            query_fragment = date_range["query_fragment"]
            count = monthly_counts.get(query_fragment, 0)
            
            # If adding this month would exceed the limit, finalize the current batch
            if current_count + count > MAX_RESULTS_PER_REQUEST and current_batch:
                # Create a date range spanning all months in the batch
                start_date = date_ranges[batch_start_idx]["start_date"]
                end_date = date_ranges[i-1]["end_date"]
                combined_date_query = f'"{start_date}"[Date - Publication] : "{end_date}"[Date - Publication]'
                
                # Create the optimized query
                optimized_query = {
                    "query": f"({base_query}) AND ({combined_date_query})",
                    "count": current_count,
                    "start_date": start_date,
                    "end_date": end_date,
                    "date_range": combined_date_query,
                    "batch_id": len(optimized_queries)
                }
                
                optimized_queries.append(optimized_query)
                
                # Reset for next batch
                current_batch = []
                current_count = 0
                batch_start_idx = i
            
            # Add current month to the batch
            current_batch.append(date_range)
            current_count += count
        
        # Handle the final batch if not empty
        if current_batch:
            start_date = date_ranges[batch_start_idx]["start_date"]
            end_date = date_ranges[-1]["end_date"]
            combined_date_query = f'"{start_date}"[Date - Publication] : "{end_date}"[Date - Publication]'
            
            optimized_query = {
                "query": f"({base_query}) AND ({combined_date_query})",
                "count": current_count,
                "start_date": start_date,
                "end_date": end_date,
                "date_range": combined_date_query,
                "batch_id": len(optimized_queries)
            }
            
            optimized_queries.append(optimized_query)
        
        return optimized_queries

    def get_safe_filename(self, query: str, batch_id: int, start: int, end: int) -> str:
        """
        Generate a safe filename for storing query results.
        
        Args:
            query: The query string
            batch_id: Batch identifier
            start: Start index
            end: End index
            
        Returns:
            A safe filename string
        """
        # Create a hash of the query to keep filenames manageable
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        
        # Create a safe filename
        return f"pubmed_batch_{batch_id}_{query_hash}_{start}_to_{end}.json"

    def retrieve_and_save_batch(self,
                              query: str,
                              batch_id: int,
                              retstart: int,
                              retmax: int,
                              download_mode: str = 'full',
                              ret_mode: str = 'xml',
                              ret_type: str = '') -> str:
        """
        Retrieve and save a batch of results.
        
        Args:
            query: The query string
            batch_id: Batch identifier
            retstart: Starting index
            retmax: Maximum records to retrieve
            download_mode: 'full' or 'summary'
            ret_mode: Return mode
            ret_type: Return type
            
        Returns:
            Path to the saved file
        """
        # Generate filename
        filename = self.get_safe_filename(query, batch_id, retstart, retstart + retmax - 1)
        filepath = os.path.join(self.cache_dir, filename)
        
        # Check if file already exists
        if os.path.exists(filepath):
            logger.info(f"Batch already exists: {filepath}")
            return filepath
        
        # Search and fetch
        web_env, query_key, total_count = self.search(query)
        
        # Adjust retmax if it exceeds the total count
        if retstart + retmax > total_count:
            retmax = max(0, total_count - retstart)
            
        if retmax <= 0:
            logger.warning(f"No results to fetch for batch {batch_id} (start: {retstart})")
            return None
            
        # Fetch the data
        data = self.fetch_batch(
            web_env=web_env,
            query_key=query_key,
            retstart=retstart,
            retmax=retmax,
            download_mode=download_mode,
            ret_mode=ret_mode,
            ret_type=ret_type
        )
        
        # Save the data
        result = {
            "query": query,
            "batch_id": batch_id,
            "retstart": retstart,
            "retmax": retmax,
            "total_count": total_count,
            "download_mode": download_mode,
            "ret_mode": ret_mode,
            "ret_type": ret_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved batch to {filepath}")
        return filepath

    def retrieve_all_batches(self,
                           query: str,
                           batch_size: int = 500,
                           download_mode: str = 'full',
                           ret_mode: str = 'xml',
                           ret_type: str = '') -> List[str]:
        """
        Retrieve all results for a query in batches.
        
        Args:
            query: The query string
            batch_size: Number of records per batch
            download_mode: 'full' or 'summary'
            ret_mode: Return mode
            ret_type: Return type
            
        Returns:
            List of paths to saved batch files
        """
        # First, get the total count
        web_env, query_key, total_count = self.search(query)
        
        if total_count == 0:
            logger.info(f"No results found for query: {query}")
            return []
            
        if total_count > MAX_RESULTS_PER_REQUEST:
            logger.warning(f"Query returned {total_count} results, which exceeds the maximum of {MAX_RESULTS_PER_REQUEST}")
            logger.warning("Consider using date-based batching with retrieve_date_batched_query()")
            
        # Calculate number of batches
        num_batches = (total_count + batch_size - 1) // batch_size
        
        saved_files = []
        
        for i in range(num_batches):
            retstart = i * batch_size
            retmax = min(batch_size, total_count - retstart)
            
            filepath = self.retrieve_and_save_batch(
                query=query,
                batch_id=i,
                retstart=retstart,
                retmax=retmax,
                download_mode=download_mode,
                ret_mode=ret_mode,
                ret_type=ret_type
            )
            
            if filepath:
                saved_files.append(filepath)
            
        return saved_files

    def retrieve_date_batched_query(self, 
                                  base_query: str,
                                  start_date: str,
                                  end_date: str,
                                  batch_size: int = 500,
                                  download_mode: str = 'full',
                                  ret_mode: str = 'xml',
                                  ret_type: str = '') -> Dict[str, Any]:
        """
        Retrieve results using date-based batching.
        
        Args:
            base_query: The base query without date restrictions
            start_date: Start date in format "YYYY/MM/DD"
            end_date: End date in format "YYYY/MM/DD"
            batch_size: Number of records per batch
            download_mode: 'full' or 'summary'
            ret_mode: Return mode
            ret_type: Return type
            
        Returns:
            Dictionary with results summary
        """
        # Generate monthly date ranges
        date_ranges = self.generate_monthly_date_ranges(start_date, end_date)
        
        # Get counts for each month
        monthly_counts = self.get_monthly_counts(base_query, date_ranges)
        
        # Create optimized queries
        optimized_queries = self.create_optimized_queries(base_query, date_ranges, monthly_counts)
        
        results = {
            "base_query": base_query,
            "start_date": start_date,
            "end_date": end_date,
            "total_batches": 0,
            "total_records": 0,
            "batch_details": []
        }
        
        # Process each optimized query
        for query_info in tqdm(optimized_queries, desc="Processingoptimized_queries"):
            
            query = query_info["query"]
            count = query_info["count"]
            batch_id = query_info["batch_id"]
            
            logger.info(f"Processing batch {batch_id}: {count} records")
            
            # Calculate number of sub-batches needed
            num_sub_batches = (count + batch_size - 1) // batch_size
            
            batch_files = []
            
            for i in range(num_sub_batches):
                retstart = i * batch_size
                retmax = min(batch_size, count - retstart)
                
                filepath = self.retrieve_and_save_batch(
                    query=query,
                    batch_id=f"{batch_id}_{i}",
                    retstart=retstart,
                    retmax=retmax,
                    download_mode=download_mode,
                    ret_mode=ret_mode,
                    ret_type=ret_type
                )
                
                if filepath:
                    batch_files.append(filepath)
            
            # Add batch details to results
            results["batch_details"].append({
                "batch_id": batch_id,
                "query": query,
                "count": count,
                "start_date": query_info["start_date"],
                "end_date": query_info["end_date"],
                "files": batch_files
            })
            
            results["total_batches"] += num_sub_batches
            results["total_records"] += count
        
        return results

    def get_existing_pmids(self) -> Set[str]:
        """
        Get PMIDs from existing JSON files in the cache directory.
        
        Returns:
            Set of PMIDs already downloaded
        """
        existing_pmids = set()
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if 'data' in data:
                            # Extract PMIDs from the data
                            pmid_matches = re.findall(r'<PMID[^>]*>(\d+)</PMID>', data['data'])
                            existing_pmids.update(pmid_matches)
                except Exception as e:
                    logger.warning(f"Error reading PMIDs from {file_path}: {e}")
        
        logger.info(f"Found {len(existing_pmids)} existing PMIDs in saved files")
        return existing_pmids

    def combine_all_files(self, output_file: str = "all_pubmed_articles.json") -> None:
        """
        Combine all JSON files into a single file.
        
        Args:
            output_file: Name of the output file
        """
        all_data = []
        pmid_set = set()  # To avoid duplicates
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json') and filename != output_file:
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if 'data' in data:
                            # Extract PMIDs and content
                            pmid_matches = re.findall(r'<PMID[^>]*>(\d+)</PMID>', data['data'])
                            for pmid in pmid_matches:
                                if pmid not in pmid_set:
                                    pmid_set.add(pmid)
                                    all_data.append({
                                        "PMID": pmid,
                                        "source_file": filename,
                                        "query": data.get("query", "")
                                    })
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error reading {file_path}: {e}")
        
        if all_data:
            output_path = os.path.join(self.cache_dir, output_file)
            with open(output_path, 'w') as f:
                json.dump(all_data, f, indent=2)
            
            logger.info(f"Combined {len(all_data)} unique articles into {output_path}")
        else:
            logger.warning("No articles found to combine")

    def parse_pubmed_xml(self, xml_data: str) -> List[Dict[str, Any]]:
        """
        Parse PubMed XML data into structured records.
        
        Args:
            xml_data: XML data string from PubMed
            
        Returns:
            List of dictionaries with structured article data
        """
        articles = []
        
        try:
            # Find all PubmedArticle elements
            article_matches = re.findall(r'<PubmedArticle>(.*?)</PubmedArticle>', xml_data, re.DOTALL)
            
            for article_xml in article_matches:
                article = {}
                
                # Extract PMID
                pmid_match = re.search(r'<PMID[^>]*>(\d+)</PMID>', article_xml)
                if pmid_match:
                    article['PMID'] = pmid_match.group(1)
                
                # Extract title
                title_match = re.search(r'<ArticleTitle>(.*?)</ArticleTitle>', article_xml)
                if title_match:
                    article['title'] = title_match.group(1)
                
                # Extract abstract
                abstract_match = re.search(r'<AbstractText>(.*?)</AbstractText>', article_xml)
                if abstract_match:
                    article['abstract'] = abstract_match.group(1)
                
                # Extract authors
                author_matches = re.findall(r'<Author[^>]*>.*?<LastName>(.*?)</LastName>.*?<ForeName>(.*?)</ForeName>.*?</Author>', article_xml, re.DOTALL)
                if author_matches:
                    article['authors'] = [f"{forename} {lastname}" for lastname, forename in author_matches]
                
                # Extract publication date
                year_match = re.search(r'<PubDate>.*?<Year>(\d+)</Year>.*?</PubDate>', article_xml, re.DOTALL)
                month_match = re.search(r'<PubDate>.*?<Month>(\w+)</Month>.*?</PubDate>', article_xml, re.DOTALL)
                
                if year_match:
                    pub_date = year_match.group(1)
                    if month_match:
                        pub_date = f"{month_match.group(1)} {pub_date}"
                    article['publication_date'] = pub_date
                
                # Extract journal
                journal_match = re.search(r'<Journal>.*?<Title>(.*?)</Title>.*?</Journal>', article_xml, re.DOTALL)
                if journal_match:
                    article['journal'] = journal_match.group(1)
                
                if article:  # Only add if we extracted something
                    articles.append(article)
                
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
        
        return articles

    def extract_articles_from_files(self) -> List[Dict[str, Any]]:
        """
        Extract and parse article data from all downloaded files.
        
        Returns:
            List of dictionaries with structured article data
        """
        all_articles = []
        pmid_set = set()  # To avoid duplicates
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if 'data' in data:
                            articles = self.parse_pubmed_xml(data['data'])
                            for article in articles:
                                if 'PMID' in article and article['PMID'] not in pmid_set:
                                    pmid_set.add(article['PMID'])
                                    article['source_file'] = filename
                                    all_articles.append(article)
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
        
        logger.info(f"Extracted {len(all_articles)} unique articles from files")
        return all_articles

    def save_articles_to_json(self, articles: List[Dict[str, Any]], output_file: str = "parsed_articles.json") -> str:
        """
        Save parsed articles to a JSON file.
        
        Args:
            articles: List of article dictionaries
            output_file: Name of the output file
            
        Returns:
            Path to the saved file
        """
        output_path = os.path.join(self.cache_dir, output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(articles)} articles to {output_path}")
        return output_path

    # For backward compatibility with your old code
    def format_pubmed_query(self, query: str) -> str:
        """Alias for format_query to maintain backward compatibility."""
        return self.format_query(query)
    
    def get_UID_from_search(self, formatted_query: str, db: str = None) -> Tuple[Any, str, str, int]:
        """
        Legacy method for backward compatibility.
        
        Args:
            formatted_query: Formatted query string
            db: Database to search (defaults to self.db if None)
            
        Returns:
            Tuple of (response, WebEnv, QueryKey, count)
        """
        if db is None:
            db = self.db
            
        url = f"{BASE_URL}esearch.fcgi?db={db}&term={formatted_query}&usehistory=y&email={self.email}"
        if self.api_key:
            url += f"&api_key={self.api_key}"
            
        self._respect_rate_limit()
        
        response = requests.get(url)
        response.raise_for_status()
        
        content = response.text
        web_env = re.search(r'<WebEnv>(\S+)</WebEnv>', content).group(1)
        query_key = re.search(r'<QueryKey>(\d+)</QueryKey>', content).group(1)
        count = int(re.search(r'<Count>(\d+)</Count>', content).group(1))
        
        logger.info(f"Search returned {count} results")
        return response, web_env, query_key, count
    
    def get_data_from_UID_chunk(self, web: str, key: str, 
                              download_mode: str = 'full',
                              db: str = None,
                              ret_mode: str = 'xml',
                              ret_type: str = '',
                              retstart: int = 0,
                              retmax: int = 500) -> str:
        """
        Legacy method for backward compatibility.
        
        Args:
            web: WebEnv value
            key: QueryKey value
            download_mode: 'full' or 'summary'
            db: Database to search (defaults to self.db if None)
            ret_mode: Return mode
            ret_type: Return type
            retstart: Starting index
            retmax: Maximum records to retrieve
            
        Returns:
            Response text from the API
        """
        if db is None:
            db = self.db
            
        download_mode = 'full' if download_mode not in ['full', 'summary'] else download_mode
        ret_mode, ret_type = self.validate_return_format(ret_mode, ret_type)
        
        self._respect_rate_limit()
        
        if download_mode == 'summary':
            url = f"{BASE_URL}esummary.fcgi?db={db}&query_key={key}&WebEnv={web}&retstart={retstart}&retmax={retmax}&email={self.email}"
        else:  # full
            url = f"{BASE_URL}efetch.fcgi?db={db}&query_key={key}&WebEnv={web}&retstart={retstart}&retmax={retmax}&email={self.email}"
            if ret_mode:
                url += f"&retmode={ret_mode}"
            if ret_type:
                url += f"&rettype={ret_type}"
                
        if self.api_key:
            url += f"&api_key={self.api_key}"
            
        logger.info(f"Fetching data from {retstart} to {retstart+retmax-1}")
        
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    
    def generate_monthlyterms(self, start_date: str, end_date: str) -> List[str]:
        """
        Legacy method for backward compatibility.
        
        Args:
            start_date: Start date in format "YYYY/MM/DD"
            end_date: End date in format "YYYY/MM/DD"
            
        Returns:
            List of monthly date range query fragments
        """
        date_ranges = self.generate_monthly_date_ranges(start_date, end_date)
        return [range_info["query_fragment"] for range_info in date_ranges]
    
    def get_monthlycount_using_esearch(self, 
                                     monthly_terms: List[str],
                                     original_query: str,
                                     cache_countfile_path: str,
                                     db: str = None,
                                     request_delay: int = None) -> Dict[str, int]:
        """
        Legacy method for backward compatibility.
        
        Args:
            monthly_terms: List of monthly date range query fragments
            original_query: The base query
            cache_countfile_path: Path to cache file
            db: Database to search (defaults to self.db if None)
            request_delay: Delay between requests (defaults to self.delay if None)
            
        Returns:
            Dictionary mapping date range query fragments to result counts
        """
        if db is None:
            db = self.db
        
        if request_delay is None:
            request_delay = self.delay
        
        # Convert to JSON if it's a pickle file
        if cache_countfile_path.endswith('.pkl'):
            cache_countfile_path = cache_countfile_path.replace('.pkl', '.json')
        
        # Ensure the cache file is in the cache directory
        cache_file = os.path.join(self.cache_dir, os.path.basename(cache_countfile_path))
        
        # Try to load existing counts from cache
        monthly_counts = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    monthly_counts = json.load(f)
                logger.info(f"Loaded {len(monthly_counts)} cached monthly counts")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load counts cache: {e}")
        
        # Get counts for any missing months
        for monthly_term in monthly_terms:
            if monthly_term in monthly_counts:
                logger.info(f"Using cached count for {monthly_term}: {monthly_counts[monthly_term]}")
                continue
                
            try:
                query = f"({original_query}) AND ({monthly_term})"
                formatted_query = self.format_query(query)
                _, _, _, count = self.get_UID_from_search(formatted_query, db)
                monthly_counts[monthly_term] = int(count)
                logger.info(f"{monthly_term}: {count} results")
                
                # Save after each successful query to avoid losing progress
                with open(cache_file, 'w') as f:
                    json.dump(monthly_counts, f, indent=2)
                
                time.sleep(request_delay)
                
            except Exception as e:
                logger.error(f"Failed to get count for {monthly_term}: {e}")
                monthly_counts[monthly_term] = 0
        
        return monthly_counts
    
    def get_Ready4search_queries(self, original_query: str, cache_countfile_path: str) -> Dict[int, Dict[str, Any]]:
        """
        Legacy method for backward compatibility.
        
        Args:
            original_query: The base query
            cache_countfile_path: Path to cache file
            
        Returns:
            Dictionary of optimized query information
        """
        # Convert to JSON if it's a pickle file
        if cache_countfile_path.endswith('.pkl'):
            cache_countfile_path = cache_countfile_path.replace('.pkl', '.json')
            
        # Ensure the cache file is in the cache directory
        cache_file = os.path.join(self.cache_dir, os.path.basename(cache_countfile_path))
        
        # Load monthly counts
        monthly_counts = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    monthly_counts = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load counts cache: {e}")
                return {}
        
        if not monthly_counts:
            logger.warning("No monthly counts available. Cannot create optimized queries.")
            return {}
            
        # Extract date ranges from the monthly terms
        date_ranges = []
        for monthly_term in monthly_counts.keys():
            match = re.search(r'"(\d{4}/\d{2}/\d{2})"[^:]*:\s*"(\d{4}/\d{2}/\d{2})"', monthly_term)
            if match:
                start_date, end_date = match.groups()
                date_ranges.append({
                    "start_date": start_date,
                    "end_date": end_date,
                    "query_fragment": monthly_term,
                    "month_year": start_date[:7].replace('/', '-')
                })
        
        # Create optimized queries
        optimized_queries = self.create_optimized_queries(original_query, date_ranges, monthly_counts)
        
        # Convert to the legacy format
        legacy_format = {}
        for i, query_info in enumerate(optimized_queries):
            legacy_format[i] = {
                'query': query_info['query'],
                'count': query_info['count'],
                'first_date': query_info['start_date'],
                'last_date': query_info['end_date'],
                'combined_range': query_info['date_range']
            }
            
        return legacy_format


def create_env_template():
    """Create a template .env file."""
    template_content = """
# This .env file was generated as a template due to invalid previous ".env" file. This is a template and the reson for haiving .env file is that the API and private info are safely accessible to the code.

# You should rename the file name to ".env"  You should enter your pubmed API (for PubmedRetriver) and OpenAI API for extraction (or openai compatible API  and base url (for example firework for using open-source models provided by Firework)).

# PubMed API Configuration
PUBMED_EMAIL=your.email@example.com
PUBMED_API_KEY=

# API Request Settings
API_DELAY=1.0

# Storage Settings
CACHE_DIRECTORY=pubmed_data

# OpenAI API
OPENAI_API_KEY= your-opneai-api

# OpenAI-Compaiblte API and Base-URL
OPENAI_COMPATIBLE_API_KEY =  your-firwork/otherplatforms-api
OPENAI_COMPATIBLE_BASE_URL = firwork/otherplatforms-baseurl

"""
    with open('.env.template', 'w') as f:
        f.write(template_content)
    logger.info("Created .env.template file")

def S1_DownloadPubmed_main(query: str, start_date="2000/01/01", end_date="2025/03/01"):
    """
    Retrieve data from PubMed using the PubMedRetriever class.
    args:
    query: str, the query to be used for retrieving data. For example: '"Artificial Intelligence"[Mesh]'
    start_date: str, the start date for the query. Default is "2000/01/01"
    end_date: str, the end date for. the query. Default is "2025/03/01"
    """

    # Check for .env file and create template if needed
    if not os.path.exists('.env'):
        logger.warning("No .env file found")
        create_env_template()
        logger.info("Please fill .env.template with your information and rename to .env")
        return
        
    try:
        # # Load environment variables
        # # We already load this at the very begining so we can change them manually, if needed. 
        # load_dotenv()
        
        # Initialize the retriever
        retriever = PubMedRetriever()
        
        # Example query
        # query = r'"Artificial Intelligence"[Mesh]'
        # start_date = "2000/01/01"
        # end_date = "2025/03/01"
        
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
        # We will keep this for S2
        # articles = retriever.extract_articles_from_files()
        # retriever.save_articles_to_json(articles, "ai_articles.json")
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)