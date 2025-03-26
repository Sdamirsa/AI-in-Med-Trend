"""
PubMed XML Parser and Cleaner

This module provides functionality to parse PubMed XML data from JSON files
and convert it to structured data with appropriate data types.

It extracts key information such as:
- Article metadata (PMID, DOI, title, abstract)
- Author information
- Journal details
- MeSH terms and keywords
- Publication dates
- Grant information
"""

import json
import os
import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime
import xml.etree.ElementTree as ET
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("S2_Pubmed_XML_Cleaner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PubMedXMLCleaner")


class PubMedXMLCleaner:
    """Class for parsing and cleaning PubMed XML data."""

    def __init__(self, data_dir: str = "pubmed_data"):
        """
        Initialize the PubMed XML cleaner.
        
        Args:
            data_dir: Directory containing JSON files with PubMed XML data
        """
        self.data_dir = data_dir
        
    def list_data_files(self) -> List[str]:
        """
        List all JSON files in the data directory.
        
        Returns:
            List of file paths
        """
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return []
            
        files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) 
                 if f.endswith('.json') and os.path.isfile(os.path.join(self.data_dir, f))]
        
        logger.info(f"Found {len(files)} JSON files in {self.data_dir}")
        return files
    
    def load_xml_from_json(self, file_path: str) -> Optional[str]:
        """
        Load XML data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            XML string or None if loading fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'data' in data:
                return data['data']
            else:
                logger.warning(f"No 'data' field found in {file_path}")
                return None
                
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def parse_xml_string(self, xml_string: str) -> List[Dict[str, Any]]:
        """
        Parse XML string into a list of article dictionaries.
        
        Args:
            xml_string: PubMed XML string
            
        Returns:
            List of dictionaries containing article data
        """
        try:
            # Clean up XML string to handle potential issues
            xml_string = self._clean_xml_string(xml_string)
            
            # Parse XML
            root = ET.fromstring(xml_string)
            
            # Extract articles
            articles = []
            for article_elem in root.findall('.//PubmedArticle'):
                article_data = self._parse_pubmed_article(article_elem)
                if article_data:
                    articles.append(article_data)
            
            logger.info(f"Parsed {len(articles)} articles from XML")
            return articles
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            return []
    
    def _clean_xml_string(self, xml_string: str) -> str:
        """
        Clean XML string to handle potential parsing issues.
        
        Args:
            xml_string: Raw XML string
            
        Returns:
            Cleaned XML string
        """
        # Remove any invalid XML characters
        xml_string = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', xml_string)
        
        # Ensure XML declaration is present and correct
        if not xml_string.strip().startswith('<?xml'):
            xml_string = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_string
        
        return xml_string
    
    def _parse_pubmed_article(self, article_elem: ET.Element) -> Dict[str, Any]:
        """
        Parse a PubmedArticle XML element.
        
        Args:
            article_elem: ET.Element for a PubmedArticle
            
        Returns:
            Dictionary containing article data
        """
        article = {}
        
        # Extract PMID
        pmid_elem = article_elem.find('.//PMID')
        if pmid_elem is not None and pmid_elem.text:
            article['pmid'] = pmid_elem.text
        else:
            # Skip articles without PMID
            return {}
        
        # Extract article metadata
        medline_citation = article_elem.find('.//MedlineCitation')
        if medline_citation is not None:
            # Article core data
            article_info = self._extract_article_info(medline_citation)
            article.update(article_info)
            
            # MeSH headings
            mesh_headings = self._extract_mesh_headings(medline_citation)
            if mesh_headings:
                article['mesh_headings'] = mesh_headings
                
            # Keywords
            keywords = self._extract_keywords(medline_citation)
            if keywords:
                article['keywords'] = keywords
                
            # Journal info
            journal_info = self._extract_journal_info(medline_citation)
            if journal_info:
                article['journal'] = journal_info
        
        # Extract PubMed data (including DOI)
        pubmed_data = article_elem.find('.//PubmedData')
        if pubmed_data is not None:
            pubmed_info = self._extract_pubmed_data(pubmed_data)
            article.update(pubmed_info)
        
        return article
    
    def _extract_article_info(self, medline_citation: ET.Element) -> Dict[str, Any]:
        """
        Extract article information from MedlineCitation.
        
        Args:
            medline_citation: MedlineCitation element
            
        Returns:
            Dictionary with article information
        """
        info = {}
        
        # Extract article element
        article_elem = medline_citation.find('./Article')
        if article_elem is None:
            return info
        
        # Title
        title_elem = article_elem.find('./ArticleTitle')
        if title_elem is not None and title_elem.text:
            info['title'] = title_elem.text
        
        # Abstract
        abstract = self._extract_abstract(article_elem)
        if abstract:
            info['abstract'] = abstract
        
        # Authors
        authors = self._extract_authors(article_elem)
        if authors:
            info['authors'] = authors
        
        # Publication types
        pub_types = self._extract_publication_types(article_elem)
        if pub_types:
            info['publication_types'] = pub_types
        
        # Publication date
        pub_date = self._extract_publication_date(article_elem)
        if pub_date:
            info['publication_date'] = pub_date
        
        # Language
        language_elem = article_elem.find('./Language')
        if language_elem is not None and language_elem.text:
            info['language'] = language_elem.text
        
        # Grant information
        grants = self._extract_grants(article_elem)
        if grants:
            info['grants'] = grants
        
        return info
    
    def _extract_abstract(self, article_elem: ET.Element) -> str:
        """
        Extract abstract from Article element.
        
        Args:
            article_elem: Article element
            
        Returns:
            Abstract text
        """
        abstract_elem = article_elem.find('./Abstract')
        if abstract_elem is None:
            return ""
        
        # Handle structured abstracts (with multiple AbstractText elements)
        abstract_parts = []
        for abstract_text in abstract_elem.findall('./AbstractText'):
            # Check for label attribute
            label = abstract_text.get('Label', '')
            text = abstract_text.text or ""
            
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        
        return " ".join(abstract_parts)
    
    def _extract_authors(self, article_elem: ET.Element) -> List[Dict[str, str]]:
        """
        Extract author information from Article element.
        
        Args:
            article_elem: Article element
            
        Returns:
            List of author dictionaries
        """
        authors = []
        author_list = article_elem.find('./AuthorList')
        
        if author_list is None:
            return authors
        
        for author_elem in author_list.findall('./Author'):
            author = {}
            
            # Last name
            last_name = author_elem.find('./LastName')
            if last_name is not None and last_name.text:
                author['last_name'] = last_name.text
            
            # First name
            fore_name = author_elem.find('./ForeName')
            if fore_name is not None and fore_name.text:
                author['first_name'] = fore_name.text
            
            # Initials
            initials = author_elem.find('./Initials')
            if initials is not None and initials.text:
                author['initials'] = initials.text
            
            # Affiliation
            affiliation = author_elem.find('./AffiliationInfo/Affiliation')
            if affiliation is not None and affiliation.text:
                author['affiliation'] = affiliation.text
            
            # Only add if we have at least a last name
            if 'last_name' in author:
                authors.append(author)
        
        return authors
    
    def _extract_publication_types(self, article_elem: ET.Element) -> List[Dict[str, str]]:
        """
        Extract publication types from Article element.
        
        Args:
            article_elem: Article element
            
        Returns:
            List of publication type dictionaries
        """
        pub_types = []
        pub_type_list = article_elem.find('./PublicationTypeList')
        
        if pub_type_list is None:
            return pub_types
        
        for pub_type_elem in pub_type_list.findall('./PublicationType'):
            if pub_type_elem.text:
                ui = pub_type_elem.get('UI', '')
                pub_types.append({
                    'type': pub_type_elem.text,
                    'ui': ui
                })
        
        return pub_types
    
    def _extract_publication_date(self, article_elem: ET.Element) -> Dict[str, Any]:
        """
        Extract publication date from Article element.
        
        Args:
            article_elem: Article element
            
        Returns:
            Dictionary with publication date information
        """
        pub_date = {}
        
        # Try to get date from PubDate in JournalIssue
        journal_issue = article_elem.find('./Journal/JournalIssue')
        if journal_issue is not None:
            pub_date_elem = journal_issue.find('./PubDate')
            if pub_date_elem is not None:
                pub_date = self._parse_date_element(pub_date_elem)
        
        # If no date found, try ArticleDate
        if not pub_date:
            article_date = article_elem.find('./ArticleDate')
            if article_date is not None:
                pub_date = self._parse_date_element(article_date)
        
        return pub_date
    
    def _parse_date_element(self, date_elem: ET.Element) -> Dict[str, Any]:
        """
        Parse a date element (PubDate or ArticleDate).
        
        Args:
            date_elem: Date element
            
        Returns:
            Dictionary with date components
        """
        date_info = {}
        
        # Extract year
        year_elem = date_elem.find('./Year')
        if year_elem is not None and year_elem.text:
            try:
                date_info['year'] = int(year_elem.text)
            except ValueError:
                date_info['year'] = year_elem.text
        
        # Extract month
        month_elem = date_elem.find('./Month')
        if month_elem is not None and month_elem.text:
            # Try to convert month name to number
            month_text = month_elem.text
            try:
                # Try to parse as number
                month_num = int(month_text)
                if 1 <= month_num <= 12:
                    date_info['month'] = month_num
            except ValueError:
                # Try to parse as month name
                try:
                    date_obj = datetime.strptime(month_text[:3], '%b')
                    date_info['month'] = date_obj.month
                except ValueError:
                    # Keep as text
                    date_info['month'] = month_text
        
        # Extract day
        day_elem = date_elem.find('./Day')
        if day_elem is not None and day_elem.text:
            try:
                date_info['day'] = int(day_elem.text)
            except ValueError:
                date_info['day'] = day_elem.text
        
        # Create ISO date string if possible
        if 'year' in date_info:
            iso_date = f"{date_info['year']}"
            if 'month' in date_info and isinstance(date_info['month'], int):
                iso_date += f"-{date_info['month']:02d}"
                if 'day' in date_info and isinstance(date_info['day'], int):
                    iso_date += f"-{date_info['day']:02d}"
            date_info['iso_date'] = iso_date
        
        return date_info
    
    def _extract_grants(self, article_elem: ET.Element) -> List[Dict[str, str]]:
        """
        Extract grant information from Article element.
        
        Args:
            article_elem: Article element
            
        Returns:
            List of grant dictionaries
        """
        grants = []
        grant_list = article_elem.find('./GrantList')
        
        if grant_list is None:
            return grants
        
        for grant_elem in grant_list.findall('./Grant'):
            grant = {}
            
            # Grant ID
            grant_id = grant_elem.find('./GrantID')
            if grant_id is not None and grant_id.text:
                grant['id'] = grant_id.text
            
            # Agency
            agency = grant_elem.find('./Agency')
            if agency is not None and agency.text:
                grant['agency'] = agency.text
            
            # Country
            country = grant_elem.find('./Country')
            if country is not None and country.text:
                grant['country'] = country.text
            
            # Acronym
            acronym = grant_elem.find('./Acronym')
            if acronym is not None and acronym.text:
                grant['acronym'] = acronym.text
            
            # Only add if we have at least an ID or agency
            if grant:
                grants.append(grant)
        
        return grants
    
    def _extract_mesh_headings(self, medline_citation: ET.Element) -> List[Dict[str, Any]]:
        """
        Extract MeSH headings from MedlineCitation.
        
        Args:
            medline_citation: MedlineCitation element
            
        Returns:
            List of MeSH heading dictionaries
        """
        mesh_headings = []
        mesh_list = medline_citation.find('./MeshHeadingList')
        
        if mesh_list is None:
            return mesh_headings
        
        for mesh_elem in mesh_list.findall('./MeshHeading'):
            heading = {}
            
            # Descriptor
            descriptor = mesh_elem.find('./DescriptorName')
            if descriptor is not None and descriptor.text:
                heading['descriptor'] = descriptor.text
                heading['descriptor_ui'] = descriptor.get('UI', '')
                heading['descriptor_major'] = descriptor.get('MajorTopicYN', 'N') == 'Y'
            
            # Qualifiers
            qualifiers = []
            for qualifier in mesh_elem.findall('./QualifierName'):
                if qualifier.text:
                    qualifiers.append({
                        'name': qualifier.text,
                        'ui': qualifier.get('UI', ''),
                        'major': qualifier.get('MajorTopicYN', 'N') == 'Y'
                    })
            
            if qualifiers:
                heading['qualifiers'] = qualifiers
            
            # Only add if we have a descriptor
            if 'descriptor' in heading:
                mesh_headings.append(heading)
        
        return mesh_headings
    
    def _extract_keywords(self, medline_citation: ET.Element) -> List[Dict[str, Any]]:
        """
        Extract keywords from MedlineCitation.
        
        Args:
            medline_citation: MedlineCitation element
            
        Returns:
            List of keyword dictionaries
        """
        all_keywords = []
        keyword_lists = medline_citation.findall('./KeywordList')
        
        for keyword_list in keyword_lists:
            owner = keyword_list.get('Owner', '')
            
            for keyword_elem in keyword_list.findall('./Keyword'):
                if keyword_elem.text:
                    all_keywords.append({
                        'keyword': keyword_elem.text,
                        'major': keyword_elem.get('MajorTopicYN', 'N') == 'Y',
                        'owner': owner
                    })
        
        return all_keywords
    
    def _extract_journal_info(self, medline_citation: ET.Element) -> Dict[str, Any]:
        """
        Extract journal information from MedlineCitation.
        
        Args:
            medline_citation: MedlineCitation element
            
        Returns:
            Dictionary with journal information
        """
        journal_info = {}
        
        # MedlineJournalInfo element
        medline_journal = medline_citation.find('./MedlineJournalInfo')
        if medline_journal is not None:
            # Journal title
            journal_title = medline_journal.find('./MedlineTA')
            if journal_title is not None and journal_title.text:
                journal_info['title_abbreviated'] = journal_title.text
            
            # NLM ID
            nlm_id = medline_journal.find('./NlmUniqueID')
            if nlm_id is not None and nlm_id.text:
                journal_info['nlm_id'] = nlm_id.text
            
            # ISSN linking
            issn_linking = medline_journal.find('./ISSNLinking')
            if issn_linking is not None and issn_linking.text:
                journal_info['issn_linking'] = issn_linking.text
            
            # Country
            country = medline_journal.find('./Country')
            if country is not None and country.text:
                journal_info['country'] = country.text
        
        # Article/Journal element
        article_elem = medline_citation.find('./Article')
        if article_elem is not None:
            journal_elem = article_elem.find('./Journal')
            if journal_elem is not None:
                # Full journal title
                title = journal_elem.find('./Title')
                if title is not None and title.text:
                    journal_info['title'] = title.text
                
                # ISO abbreviation
                iso_abbr = journal_elem.find('./ISOAbbreviation')
                if iso_abbr is not None and iso_abbr.text:
                    journal_info['iso_abbreviation'] = iso_abbr.text
                
                # ISSN
                issn = journal_elem.find('./ISSN')
                if issn is not None and issn.text:
                    journal_info['issn'] = issn.text
                    journal_info['issn_type'] = issn.get('IssnType', '')
                
                # Volume, issue, pagination
                journal_issue = journal_elem.find('./JournalIssue')
                if journal_issue is not None:
                    # Volume
                    volume = journal_issue.find('./Volume')
                    if volume is not None and volume.text:
                        journal_info['volume'] = volume.text
                    
                    # Issue
                    issue = journal_issue.find('./Issue')
                    if issue is not None and issue.text:
                        journal_info['issue'] = issue.text
                    
                    # Cited medium
                    cited_medium = journal_issue.get('CitedMedium', '')
                    if cited_medium:
                        journal_info['cited_medium'] = cited_medium
                
                # Pagination
                pagination = article_elem.find('./Pagination/MedlinePgn')
                if pagination is not None and pagination.text:
                    journal_info['pagination'] = pagination.text
        
        return journal_info
    
    def _extract_pubmed_data(self, pubmed_data: ET.Element) -> Dict[str, Any]:
        """
        Extract information from PubmedData element.
        
        Args:
            pubmed_data: PubmedData element
            
        Returns:
            Dictionary with PubMed data
        """
        info = {}
        
        # Publication status
        pub_status = pubmed_data.find('./PublicationStatus')
        if pub_status is not None and pub_status.text:
            info['publication_status'] = pub_status.text
        
        # Article IDs (including DOI)
        article_ids = []
        id_list = pubmed_data.find('./ArticleIdList')
        if id_list is not None:
            for id_elem in id_list.findall('./ArticleId'):
                if id_elem.text:
                    id_type = id_elem.get('IdType', '')
                    article_ids.append({
                        'id': id_elem.text,
                        'type': id_type
                    })
                    
                    # Add DOI as a separate field for convenience
                    if id_type == 'doi':
                        info['doi'] = id_elem.text
        
        if article_ids:
            info['article_ids'] = article_ids
        
        # History dates
        history_dates = []
        history = pubmed_data.find('./History')
        if history is not None:
            for date_elem in history.findall('./PubMedPubDate'):
                status = date_elem.get('PubStatus', '')
                date_info = self._parse_date_element(date_elem)
                
                if date_info and status:
                    date_info['status'] = status
                    history_dates.append(date_info)
        
        if history_dates:
            info['history'] = history_dates
        
        return info
    
    def process_all_files(self, output_file: str = "parsed_articles.json") -> str:
        """
        Process all JSON files and extract article data.
        
        Args:
            output_file: Name of the output file
            
        Returns:
            Path to the output file
        """
        all_articles = []
        pmid_set = set()  # To avoid duplicates
        
        # Get list of files
        files = self.list_data_files()
        
        # Process each file
        for file_path in files:
            logger.info(f"Processing {file_path}")
            
            # Load XML from JSON
            xml_string = self.load_xml_from_json(file_path)
            if not xml_string:
                continue
                
            # Parse XML
            articles = self.parse_xml_string(xml_string)
            
            # Add to collection, avoiding duplicates
            for article in articles:
                if 'pmid' in article and article['pmid'] not in pmid_set:
                    pmid_set.add(article['pmid'])
                    article['source_file'] = os.path.basename(file_path)
                    all_articles.append(article)
        
        # Save to output file
        output_path = os.path.join(self.data_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(all_articles)} articles to {output_path}")
        return output_path
    
    def process_single_file(self, file_path: str, output_file: Optional[str] = None) -> str:
        """
        Process a single JSON file and extract article data.
        
        Args:
            file_path: Path to the JSON file
            output_file: Name of the output file (if None, derived from input file)
            
        Returns:
            Path to the output file
        """
        # Derive output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_file = f"{base_name}_parsed.json"
        
        # Load XML from JSON
        xml_string = self.load_xml_from_json(file_path)
        if not xml_string:
            return ""
            
        # Parse XML
        articles = self.parse_xml_string(xml_string)
        
        # Add source file information
        for article in articles:
            article['source_file'] = os.path.basename(file_path)
        
        # Save to output file
        output_path = os.path.join(self.data_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(articles)} articles to {output_path}")
        return output_path


# Example usage
def main():
    """Example usage of the PubMedXMLCleaner class."""
    try:
        # Initialize the cleaner
        cleaner = PubMedXMLCleaner(data_dir="pubmed_data")
        
        # Process all files
        output_path = cleaner.process_all_files()
        print(f"Processed all files. Output saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)

