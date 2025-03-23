# Setting the environment and libraries
#[cmd]python -m venv pubmedreq
#[cmd] .\pubmedreq\Scripts\activate
#%pip install requests
#%pip install pandas
#%pip install eutils

from Code.Pubmed_Retriver import *

# Example usage-----
start_date="1999/01/01"
end_date="2024/01/01"
monthly_terms = generate_monthlyterms(start_date=start_date, end_date=end_date)

original_query=r'("1999/01/01"[Date - Publication] : "2024/01/01"[Date - Publication]) AND ("Artificial Intelligence"[Mesh])'
cache_countfile_path='AIinMed_MountlyCount.pkl'
Ready4searchs_dic = get_Ready4search_queries(original_query=original_query,cache_countfile_path=cache_countfile_path)

monthlycount_dic = get_monthlycount_using_esearch(original_query=original_query,monthly_terms=monthly_terms, cache_countfile_path=cache_countfile_path)