import os
import json
import time
import asyncio
from typing import Optional, Dict, Any
from .S3_EXCT_Function import EXCT_main
from .S3_EXCT_

# Make sure these imports/definitions exist somewhere in your code:
# from your_file import EXCT_main, get_json_sublist, build_exct_dictionary

def S3_EXCT_processor_main(
    folder_path: str,
    filter_startstring: str,
    add_string_at_beginning: str,
    EXCT_main_kwargs_dictionary: Dict[str, Any]
) -> None:
    """
    Iterates over JSON files in 'folder_path'. For each file whose name starts 
    with 'filter_startstring', calls the async EXCT_main function. Merges 
    EXCT_main_kwargs_dictionary with the necessary arguments (json_file_path 
    and output_file_path).
    
    If 'add_string_at_beginning' is empty, overwrites the original file. 
    Otherwise, prefixes the new filename with add_string_at_beginning.
    """

    # List all files in the specified folder
    for filename in os.listdir(folder_path):
        # We only process .json files that start with filter_startstring
        if filename.endswith(".json") and filename.startswith(filter_startstring):
            input_file_path = os.path.join(folder_path, filename)

            # Decide whether to overwrite or rename
            if add_string_at_beginning:
                new_filename = add_string_at_beginning + filename
            else:
                new_filename = filename  # Overwrite the same file

            output_file_path = os.path.join(folder_path, new_filename)

            # Prepare kwargs for EXCT_main by merging user-provided dictionary
            exct_kwargs = dict(EXCT_main_kwargs_dictionary)
            exct_kwargs["json_file_path"] = input_file_path
            exct_kwargs["output_file_path"] = output_file_path

            # Now call EXCT_main in a synchronous context using asyncio.run
            asyncio.run(EXCT_main(**exct_kwargs))

            # Print or log status
            print(f"Processed: {filename} => {new_filename}")

# ------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Suppose you have a dictionary of all other EXCT_main parameters:
#     exct_params = {
#         "text_key": "abstract", 
#         "Pydantic_Objects_List": [],  # your pydantic models
#         "path_to_list": None,
#         "model_engine": "OpenAI_Async",
#         "parser_error_handling": "llm_to_correct",
#         "model": "gpt-3.5-turbo",
#         "pre_prompt": "",
#         "temperature": 0,
#         "max_tokens": 2048,
#         "logprobs": False,
#         "seed": None,
#         "timeout": 60,
#         "max_retries": 2,
#         "openai_api_key": "YOUR_OPENAI_API_KEY",
#         "runpod_base_url": "",
#         "runpod_api": "",
#         "azure_api_key": "YOUR_AZURE_API_KEY",
#         "azure_endpoint": "YOUR_AZURE_ENDPOINT",
#         "azure_api_version": "YOUR_AZURE_API_VERSION",
#         "total_async_n": 5,
#         # Note that we don't pass json_file_path or output_file_path here
#     }

#     folder_to_process = r"C:\path\to\folder"
#     filter_str = "processed_"  # e.g., only process JSON files that start with "processed_"
#     prefix_str = "extracted_"

#     S3_EXCT_processor_main(
#         folder_path=folder_to_process,
#         filter_startstring=filter_str,
#         add_string_at_beginning=prefix_str,
#         EXCT_main_kwargs_dictionary=exct_params
#     )
