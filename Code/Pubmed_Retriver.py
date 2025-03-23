# v2: Added downloading papers in chunks so we can get big data (from couple of hundred thousand)
# v1: tansfroming normal query to formatted query Done >> Get search results (esearch) DONE >> get the data (efetch) DONE >> Store in database (epost) DONE 
import urllib.parse
from typing import Tuple, final
from math import nan
import requests
import xml.etree.ElementTree as ET
import pickle
import xml.etree.ElementTree as ET
import os
import time


def format_pubmed_query(query: str) -> str:
    """
    This function takes a query string and formats it to be suitable for searching in PubMed using eutils.
    It will 1- replace three charecters of space, qutation mark, and hashtag to equal charecter. 2- it will lower case all strings.
    
    Parameters:
    query (str): The string to be formatted.

    Returns:
    str: The formatted query string.

    Examples:
    >>> format_pubmed_query('cancer biology')
    'cancer+biology'
    >>> format_pubmed_query('human genetics')
    'human+genetics'
    """
    
    transform_dic={
        r' ': r'+',
        r'"': r'%22',
        r'#': r'%23'
    }
    transform_key=list(transform_dic.keys())
    
    formatted_query_list=[]
    for string in query:
        if string in  transform_dic:
            new_string=transform_dic[string]
            formatted_query_list.append(new_string)
        else:
            formatted_query_list.append(string.lower())
            
    formatted_query = "".join(formatted_query_list)
    
    print(f'The formatted query::: {formatted_query}')
    return formatted_query

def validate_download_mode(download_mode:str) -> str :
    """
    Validates the downloade_mode. We will do this to find out wether we should use esearch or efetch.
    
    Args:
    download_mode: the download mode defined by user
    
    Returns:
    download_mode: return the download mode if it is correctly defined. Otherwise, it will raise error.
    """
    
    if download_mode not in ['full', 'summary']:
        raise ValueError("Invalid download_mode. Please choose between 'full' or 'summary'.")
    else:
        return download_mode
def validate_retmode_rettype_basedon_db(ret_mode:str,ret_type:str, db:str) -> Tuple[str, str]: #TODO # I completed this for pubmed only. You should complete this by using this table: https://www.ncbi.nlm.nih.gov/books/NBK25499/table/chapter4.T._valid_values_of__retmode_and/?report=objectonly
    """
    Validates the return mode and return type based on the database provided.
    
    Args:
    ret_mode (str): The return mode to validate.
    ret_type (str): The return type to validate.
    db (str): The database to base the validation on.
    
    Returns:
    Tuple[str, str]: A tuple containing the validated return mode and return type, if correctly defined. Otherwise, it will raise error.
    """
    
    validation_dic={
        'pubmed': { # rettype #retm
            '': 'xml',
            'medline':'text',
            'uilist': 'text',
            'abstract': 'text'},
        }
    
    if db in list(validation_dic.keys()):
        if (ret_type, ret_mode.lower()) in list(validation_dic[db].items()):
            return ret_mode.lower(), ret_type
        else:
            raise ValueError(f"Invalid ret_type ('{ret_type}') and ret_mode ('{ret_mode}') pair for your database. Please choose the current pair based on your database (See here: https://www.ncbi.nlm.nih.gov/books/NBK25499/table/chapter4.T._valid_values_of__retmode_and/?report=objectonly).")
    else:
        print('WARNING: The correct validation for your database is not complete. Add this to the validation_dic or use it carefully.')
        return ret_mode.lower(), ret_type
    
def get_UID_from_search(formatted_query:str, db:str):
    base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
    
    #assemble the esearch URL
    url = base + "esearch.fcgi?db=" + db + "&term=" + formatted_query + "&usehistory=y"
    print(f"The esearch url is: {url}")
    #post the esearch URL
    response = requests.get(url)
    #parse WebEnv and QueryKey
    web = response.text.split('<WebEnv>')[1].split('</WebEnv>')[0]
    key = response.text.split('<QueryKey>')[1].split('</QueryKey>')[0]
    count=response.text.split('<Count>')[1].split('</Count>')[0]
    print(f'The count of retrieved objects using esearch::: {count}')
    return response, web, key, count


def get_data_from_UID_chunk(web, key,
                        download_mode:str='full', #full or summary
                        db:str = 'pubmed', 
                        ret_mode: str='XML',
                        ret_type:str ='',
                        retstart:int=0,
                        retmax:int=500):
    
    download_mode=validate_download_mode(download_mode)
    ret_mode, ret_type=validate_retmode_rettype_basedon_db(ret_mode, ret_type , db)
    base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
    
    if download_mode == 'summary':
        #assemble the esummary URL
        url = base + "esummary.fcgi?db=" + db + "&query_key=" + key + "&WebEnv=" + web +"&retstart=" +str(retstart)+"&retmax=" +str(retmax)
        print(f"The esummary url is: {url}")
        #post the esummary URL
        doc_sums = requests.get(url).text
        return doc_sums
        
    elif download_mode == 'full':
        #assemble the efetch URL #fore more detail see this page: https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.EFetch or this table: https://www.ncbi.nlm.nih.gov/books/NBK25499/table/chapter4.T._valid_values_of__retmode_and/?report=objectonly
        url = base + "efetch.fcgi?db=" + db + "&query_key=" + key + "&WebEnv=" + web +"&retstart=" +str(retstart)+"&retmax=" +str(retmax)
        print(f"The efetch url is: {url}")
        #post the efetch URL
        doc_full = requests.get(url).text
        return doc_full

def save_cache(key:str,data ,cache_file_path):
    cache={}
    cache[key]=data
    with open(cache_file_path, 'wb') as f:
        pickle.dump(cache, f)
        print (f'Updated cache saved at {cache_file_path}')

def load_cache(key:str, cache_file_path):
    try:
        with open(cache_file_path, 'rb') as f:
            print("Cache loaded.")
            all_data = pickle.load(f)
            if key=='return_first_element':
                data = next(iter(all_data.values()), None)
            else:
                data=all_data[key]
            
            if data:
                return data
            else:
                print('The data was loaded but the key in the data is not valid.')
    except FileNotFoundError:
        print("No previously saved cache was found. Initializing cache.")
        return None
    
def searchquery_cache(key:str, cache):
    # Check if the result is already cached
    if key in cache:
        print("Loading data from cache...")
        return cache[key]
    
    # If not cached, return 'new_query'
    print("Query not found in cache. Returning None.")
    return None

def check_count_lower9998(count: int):
    if count >= 9999:
        raise ValueError('The number of retrived articles are higher than 9998. Therefore, you should manually chunk your data using restrictions in the search query, such as date.')
    else:
        return True

def handler_query2docs_wCache(query:str,chunk_size:int, cache_file_path='AIinGI_cache.pkl', overwrite_cache:bool=False, 
                         download_mode:str ='full' , db:str = 'pubmed', ret_mode: str='xml',ret_type:str ='',delay_in_loop:int=0,
                         ):
    """ This function will search the query, and download it in chunks. 
    During the processing it will save each chunk in the cache to avoid re-doing things.
    """
    try:
        #all_chunks={}

        formatted_query = format_pubmed_query(query)
        response, web, key, count = get_UID_from_search(formatted_query, db)
        
        retstart , retmax= 0, chunk_size
        cache_root, pickel_extension = os.path.splitext(cache_file_path)
        
        continuechunks=True
        while continuechunks:
            cache_root_4chunk_path="".join([cache_root,"_",str(retstart),"_to_",str(retmax),pickel_extension])
            query_and_chunk_key="".join([query+"_",str(retstart),"_to_",str(retmax)])
            
            # checking and loading cache
            docs=load_cache(key=query_and_chunk_key,cache_file_path=cache_root_4chunk_path)
            if docs and not overwrite_cache:
                #all_chunks[query_and_chunk_key]=docs
                print(f'This chunk was previously saved at: {cache_root_4chunk_path}')
            else:
                cache_chunk={}
                docs = get_data_from_UID_chunk(web=web, key=key, download_mode=download_mode, db = db, ret_mode=ret_mode, ret_type=ret_type,
                                            retstart=retstart,retmax=retmax)
                #all_chunks[query_and_chunk_key]=docs
                #saving the docs to cache
                save_cache(key=query_and_chunk_key,data=docs, cache_file_path=cache_root_4chunk_path)
                time.sleep(delay_in_loop)
                
            # setting the retstart and retmax for the next loop 
            retstart= int(retmax) + 1 
            retmax = int(retmax) + int(chunk_size)
            if retstart > int(count):
                continuechunks=False
            if retmax > int(count):
                retmax=int(count)
            
            
            #return all_chunks
            print(f'All chunks are done.')
    except Exception as e:
        print(f'Error occurred: {e}')
    except InterruptedError: 
        print(f'Okey then ... ')

# Preapring chunks in three steps: 
# generate monthly terms (from the start to the end of the month)
# getting count of monthly search terms (along with original query)
# Creating most sufficient full search queries, with count lower than 9995

from datetime import datetime
from datetime import timedelta
import time
import re

def generate_monthlyterms(start_date, end_date):
    # Convert string dates to datetime objects
    start = datetime.strptime(start_date, "%Y/%m/%d")
    end = datetime.strptime(end_date, "%Y/%m/%d")

    monthly_terms = []
    current = start

    while current <= end:
        year = current.year
        month = current.month
        # Calculate the last day of the current month
        last_day = (current.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        # Format the search term for the current month
        term = f'"{year}/{month:02d}/01"[Date - Publication] : "{year}/{month:02d}/{last_day.day:02d}"[Date - Publication]'
        monthly_terms.append(term)
        # Move to the first day of the next month
        if month == 12:
            current = datetime(year + 1, 1, 1)
        else:
            current = datetime(year, month + 1, 1)

    return monthly_terms

def get_monthlycount_using_esearch(monthly_terms:list,original_query:str,cache_countfile_path:str, db:str='pubmed', request_delay:int=5) -> dict:
    
    monthlycount_dic=load_cache(key=original_query,cache_file_path=cache_countfile_path)
    if not monthlycount_dic:
        monthlycount_dic={}
        
    try:
        for monthly_term in monthly_terms:
            previous_count=monthlycount_dic.get(monthly_term, None)
            if previous_count:
                print(f'CACHE: {monthly_term}::: {previous_count}')
            else:
                try:
                    query=f"""({original_query}) AND ({monthly_term})"""
                    formatted_query = format_pubmed_query(query)
                    response, web, key, count = get_UID_from_search(formatted_query, db)
                    monthlycount_dic[monthly_term]=int(count)
                    print(f'{monthly_term}::: {count}')
                    time.sleep(request_delay)
                except Exception as e:
                    monthlycount_dic[monthly_term]=e
                    print(f'{monthly_term}::: {e}')
    except Exception as ee:
        print(f'Error occurred in get_monthlycount_using_esearch: {ee}')
    finally:
        save_cache(key=original_query,data=monthlycount_dic, cache_file_path=cache_countfile_path)
        return monthlycount_dic

def get_Ready4search_queries(original_query:str, cache_countfile_path:str) -> dict:
    monthlycount_dic=load_cache(key=original_query,cache_file_path=cache_countfile_path)
    Ready4searchs_dic={}
    temp_list=[]
    temp_list.append(original_query)
    total_count=0

    for monthly_term in list(monthlycount_dic.key()):
        if first_date is None:
            first_date=monthly_term
        count=monthlycount_dic[monthly_term]
        total_count=total_count+count
        
        if total_count+count >9995:
            Ready4search=" ".join(temp_list)
            Ready4searchs_dic[Ready4search]['count']=total_count
            Ready4searchs_dic[Ready4search]['first_date']=first_date
            Ready4searchs_dic[Ready4search]['last_date']=last_date
            
            total_count=0
            temp_list=[]
            temp_list.append(original_query)
            temp_list.append(f'OR ({monthly_term})')
            
        else:
            last_date=monthly_term
            total_count=total_count+count
            temp_list.append(f'OR ({monthly_term})')
            
    return Ready4searchs_dic

import re
from datetime import datetime

def combine_date_ranges(*date_ranges):
    # Initialize variables to store the earliest and latest dates
    earliest_date = None
    latest_date = None

    # Define a pattern to extract dates from the input strings
    date_pattern = re.compile(r"(\d{4}/\d{2}/\d{2})")

    for range_str in date_ranges:
        # Find all date occurrences in the string
        dates = date_pattern.findall(range_str)
        if dates:
            # Convert string dates to datetime objects
            start_date = datetime.strptime(dates[0], "%Y/%m/%d")
            end_date = datetime.strptime(dates[1], "%Y/%m/%d")

            # Update earliest and latest dates based on current range
            if earliest_date is None or start_date < earliest_date:
                earliest_date = start_date
            if latest_date is None or end_date > latest_date:
                latest_date = end_date

    # Format the earliest and latest dates back to string in the required format
    if earliest_date and latest_date:
        combined_range = f'{earliest_date.strftime("%Y/%m/%d")}"[Date - Publication] : "{latest_date.strftime("%Y/%m/%d")}"[Date - Publication]'
        return combined_range
    else:
        return None


def get_Ready4search_queries(original_query:str, cache_countfile_path:str) -> dict:
    monthlycount_dic=load_cache(key=original_query,cache_file_path=cache_countfile_path)
    last_monthly_term = list(monthlycount_dic.keys())[-1]

    Ready4searchs_dic={}
    total_count=0
    first_date=None    
    i=0

    
    for monthly_term in list(monthlycount_dic.keys()):
        if first_date is None:
            first_date=monthly_term
        count=monthlycount_dic[monthly_term]
        
        if total_count+count >9995 or monthly_term==last_monthly_term:
            #creating Ready4search query
            monthly_term_range = combine_date_ranges(first_date,last_date)
            monthly_term_range_query=f'AND ("{monthly_term_range})'
            Ready4search=" ".join([original_query, monthly_term_range_query])
            #creating the Ready4searchs_dic 
            Ready4searchs_dic[i]={
                'query': str(Ready4search),
                'count': int(total_count),
                'first_date': str(first_date),
                'last_date': str(last_date),
                'combined_range':str(monthly_term_range)
                }
            print(f'The {i} object added with total count of {total_count} from {first_date} to {last_date} \n      combined_range:{monthly_term_range} \n      query: {Ready4search}')
            #(re)set the iterators
            i+=1
            total_count=0
            first_date=monthly_term
        else:
            last_date=monthly_term
            total_count=total_count+count
            
    return Ready4searchs_dic


def clean_combinerange_string(input_string):
    # Remove anything within square brackets along with the brackets
    no_brackets = re.sub(r'\[.*?\]', '', input_string)
    # Replace all slashes '/' with nothing to concatenate date parts
    concatenated_dates = re.sub(r'/', '', no_brackets)
    # Remove spaces and colons, replace them with '_'
    sanitized = re.sub(r'[\s:]+', '_', concatenated_dates)
    # Remove any non-digit and non-underscore characters
    sanitized = re.sub(r'[^0-9_]', '', sanitized)
    # Replace multiple consecutive underscores with a single underscore
    clean_filename = re.sub(r'_+', '_', sanitized)
    # Remove trailing or leading underscores if present
    clean_filename = re.sub(r'^_|_$', '', clean_filename)
    return clean_filename