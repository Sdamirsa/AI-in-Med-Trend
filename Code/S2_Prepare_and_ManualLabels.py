import json
import os
import logging

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
    

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("S2_Prepare_and_ManualLabels.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("S2_prepare_and_label_main")

##############################################
# 1) Prepare LLM input for each article
##############################################
def prepare_llm_input(article_info: dict) -> str:
    """
    Prepares a clean input string for LLM extraction based on article information.
    Includes title, abstract, publication types, and keywords.
    """
    title = article_info.get("title", "")
    abstract = article_info.get("abstract", "")
    # publication_types = article_info.get("publication_types", [])

    # Publication types as a comma-separated string
    # publication_types_str = ", ".join(
    #     pub_type.get("type", "") for pub_type in publication_types
    # )

    # Create a formatted string for the LLM input
    llm_input = (
        f"# Title:\n{title}\n\n"
        f"# Abstract:\n{abstract}\n\n"
        # f"# Publication Types:\n{publication_types_str}\n\n"
    )

    return llm_input

##############################################
# 2) Process publication types (exclusions)
##############################################
def process_publication_types(publication_types: list, manual_metadata: dict) -> None:
    """
    Processes the publication types to update the 'manual_metadata' dictionary.
    Checks types like 'Review', 'Case Reports', 'Comment', 'Letter', etc.
    """
    # Initialize a sub-dict for publication_types if not present
    if "publication_types" not in manual_metadata:
        manual_metadata["publication_types"] = {
            "ArticleType_Exclude_Review": False,
            "ArticleType_Exclude_CaseReport": False,
            "ArticleType_Exclude_Letter": False,
            "exclude_bool": False,
            "exclude_reason": ""
        }

    # Check publication types for specific exclusions
    for pub_type in publication_types:
        type_name = pub_type.get("type", "")
        if type_name == "Review":
            manual_metadata["publication_types"]["ArticleType_Exclude_Review"] = True
            manual_metadata["publication_types"]["exclude_bool"] = True
            manual_metadata["publication_types"]["exclude_reason"] = "Review"
            break  # Exit after setting the first exclusion
        elif type_name == "Case Reports":
            manual_metadata["publication_types"]["ArticleType_Exclude_CaseReport"] = True
            manual_metadata["publication_types"]["exclude_bool"] = True
            manual_metadata["publication_types"]["exclude_reason"] = "Case Report"
            break
        elif type_name == "Comment" or type_name == "Letter":
            manual_metadata["publication_types"]["ArticleType_Exclude_Letter"] = True
            manual_metadata["publication_types"]["exclude_bool"] = True
            manual_metadata["publication_types"]["exclude_reason"] = "Letter or Comment"
            break

##############################################
# 3) Process language (English vs. non-English)
##############################################
def process_language(article_info: dict, manual_metadata: dict) -> None:
    """
    Processes the language of the article and checks if the abstract is in English.
    Updates the manual_metadata with language and relevant flags.
    """
    language = article_info.get("language", "eng")

    # Initialize a sub-dict for language if not present
    if "language" not in manual_metadata:
        manual_metadata["language"] = {}

    # Record the article's language
    manual_metadata["language"]["language"] = language

    # If not English, set flags for not_eng_paper and not_eng_abstract
    if language != "eng":
        manual_metadata["language"]["not_eng_paper"] = True

        # Check if there's an "English Abstract" in the publication_types
        publication_types = article_info.get("publication_types", [])
        english_abstract_found = any(
            pub_type.get("type") == "English Abstract" for pub_type in publication_types
        )
        manual_metadata["language"]["not_eng_abstract"] = not english_abstract_found
    else:
        manual_metadata["language"]["not_eng_paper"] = False
        manual_metadata["language"]["not_eng_abstract"] = False

##############################################
# 4) Process keywords & mesh headings
##############################################
def process_keywords_and_mesh(article_info: dict, manual_metadata: dict) -> None:
    """
    Gathers mesh heading descriptors and user-provided keywords
    into a single list stored in manual_metadata["keywords_and_meshHeading"].
    """
    mesh_headings = article_info.get("mesh_headings", [])
    keywords = article_info.get("keywords", [])

    # Extract 'descriptor' from mesh_headings
    mesh_list = [
        mh.get("descriptor", "") 
        for mh in mesh_headings 
        if mh.get("descriptor")
    ]
    # Extract 'keyword' from keywords
    keywords_list = [
        kw.get("keyword", "") 
        for kw in keywords 
        if kw.get("keyword")
    ]

    # Combine the two lists
    combined = mesh_list + keywords_list
    manual_metadata["keywords_and_meshHeading"] = combined


##############################################
# 5) Function to prepare & label a single JSON
##############################################
def prepare_and_label_pubmed(json_file_path: str) -> dict:
    """
    Reads the PubMed JSON file, processes each article, and adds:
      - clean LLM input
      - publication type exclusion flags
      - language flags
      - combined keywords & mesh headings
    The updated JSON is returned after the processing.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Iterate through each article in the "articles" list
    for article_info in data.get("articles", []):
        
        # Prepare LLM input
        llm_input = prepare_llm_input(article_info)
        article_info["llm_input"] = llm_input
        
        # Build a single manual_metadata dictionary for this article
        manual_metadata = {}
        
        # Process publication types
        publication_types = article_info.get("publication_types", [])
        process_publication_types(publication_types, manual_metadata)

        # Process language info
        process_language(article_info, manual_metadata)

        # Process mesh headings + keywords
        process_keywords_and_mesh(article_info, manual_metadata)

        # Attach final metadata to the article
        article_info["manual_metadata"] = manual_metadata
        
        # Add keywords to the end of llm_input
        article_info["llm_input"] += f"\n# Keywords:\n{'; '.join(article_info['manual_metadata']['keywords_and_meshHeading'])}\n\n"
    
    return data

##############################################
# 6) The new main function: S2_prepare_and_label_main
##############################################
def S2_prepare_and_label_main(
    folder_path: str,
    filter_startstring: str = "",
    add_string_at_beginning: str = ""
) -> None:
    """
    Iterates over JSON files in 'folder_path'. For each file whose
    name starts with 'filter_substring', calls 'prepare_and_label_pubmed'.
    Depending on 'add_string_at_beginning':
      - If it's empty, overwrite the existing file.
      - Otherwise, prepend add_string_at_beginning to the filename 
        and save the processed result under that new name.
    """
    # List all files in the specified folder
    for filename in tqdm(os.listdir(folder_path), desc="Processing files"): 
        # We only process .json files that match the filter_substring
        if filename.endswith(".json") and filename.startswith(filter_startstring):
            file_path = os.path.join(folder_path, filename)

            # Prepare and label the JSON
            updated_data = prepare_and_label_pubmed(file_path)

            # Decide whether to overwrite or rename
            if add_string_at_beginning:
                new_filename = add_string_at_beginning + filename
            else:
                new_filename = filename  # Overwrite the same file

            output_file_path = os.path.join(folder_path, new_filename)

            # Write out the updated data
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, ensure_ascii=False, indent=2)

            print(f"Processed: {filename} => {new_filename}")

##############################################
# 7) Example usage if run as a script
##############################################
# if __name__ == "__main__":
#     # Example usage:
#     #   We want to process all JSON files containing "pubmed_batch"
#     #   in the folder "C:/Users/LEGION/Documents/SomeFolder"
#     #   and we do NOT rename them (so we overwrite them).
#     test_folder = r"C:\Users\LEGION\Documents\GIT\AI-in-Med-Trend\pubmed_data_test\pubmed_data_S2"
#     substring_filter = "cleaned_pubmed"
#     add_prefix = "prepared"  # empty => overwrite

#     S2_prepare_and_label_main(
#         folder_path=test_folder,
#         filter_startstring=substring_filter,
#         add_string_at_beginning=add_prefix
#     )
