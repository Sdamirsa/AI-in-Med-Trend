# Dictionary of experimental configurations for testing
from .EXCT_pydantic import SOMETHING

experiments_dic = {
    'GPT4o_t03_promptSimple': {
        "model_engine": "AzureOpenAI_Async",
        "total_async_n": 3,  # the number of requests to be sent in parallel
        "pre_prompt": "You possess knowledge of medical terminologies, especially that of a cardiologist with experience reading and interpreting echocardiographic reports with knowledge of structural heart findings. Follow the Pydantic structure to extract structured data from the echo report. Provide the structured data in clean JSON format.",
        "temperature": 0.3,
        "model": 'gpt-4o-2024-11-20',
        "experiment_aRGB_cellcolor": "8f4e0d",  # used for highlighting 
        'max_tokens': 2048,
        "timeout": 60,
        'max_retries': 2,
        "seed": None,
        "logprobs": False,
        'open_at_end': False,
        'excel_file_path': r"Syngo 100 sample.xlsx",  # could also be a .csv
        "Pydantic_Objects_List": [SOMETHING],
        "parser_error_handling": "llm_to_correct",
        "openai_api_key": "",
        "runpod_base_url": "",
        "runpod_api": "",
        "azure_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "azure_api_version": "2024-08-01-preview",
        "text_column": "Report",  # either textPath_column OR text_column
        "textPath_column": None,
    },
}



#%pip install nest_asyncio
import nest_asyncio
nest_asyncio.apply()

import os
import asyncio
from typing import Dict, Any
import getpass


def set_azure_credentials(experiments_dic: Dict[str, Dict[str, Any]], save_to_env_file=False):
    """
    Function to set Azure credentials for experiments using getpass for secure input.

    Args:
        experiments_dic (Dict[str, Dict[str, Any]]): Dictionary containing experiment configurations.
        save_to_env_file (bool): Whether to save credentials to a .env file for persistence.
    """
    for experiment_label, experiment_detail in experiments_dic.items():
        if experiment_detail.get("model_engine") == "AzureOpenAI_Async":
            print(f"Setting Azure credentials for experiment: {experiment_label}")
            experiment_detail["azure_api_key"] = getpass.getpass(prompt="Enter Azure API Key: ")
            experiment_detail["azure_endpoint"] = getpass.getpass(prompt="Enter Azure Endpoint: ")

            # Set environment variables
            os.environ["AZURE_OPENAI_API_KEY"] = experiment_detail["azure_api_key"]
            os.environ["AZURE_OPENAI_ENDPOINT"] = experiment_detail["azure_endpoint"]
            os.environ["AZURE_API_VERSION"] = "2024-08-01-preview"
            
            # Optionally save to environment file
            if save_to_env_file:
                # Look for existing environment files
                env_files = [f for f in os.listdir(".") if f.endswith(".env") or f == ".env"]
                env_file_path = ".env"  # Default if no env file found
                
                if env_files:
                    env_file_path = env_files[0]
                    print(f"Found existing environment file: {env_file_path}")
                
                with open(env_file_path, "a") as env_file:
                    env_file.write(f"AZURE_OPENAI_API_KEY={experiment_detail['azure_api_key']}\n")
                    env_file.write(f"AZURE_OPENAI_ENDPOINT={experiment_detail['azure_endpoint']}\n")
                    env_file.write("AZURE_API_VERSION=2024-08-01-preview\n")
                print(f"Credentials saved to {env_file_path} in {os.getcwd()}")

# Set Azure credentials if needed
set_azure_credentials(experiments_dic, save_to_env_file=True)

#===========================================================
# Import your main_async_EXCT_MARIA_multiagent and helpers
# from wherever you have defined them, e.g.:
# from your_module_name import (
#    main_async_EXCT_MARIA_multiagent,
#    color_cells_in_excel,
#    open_excel_file
# )
#===========================================================

async def run_experiment(experiments_dic: Dict[str, Dict[str, Any]]):
    """
    Function to run multiple experiments asynchronously, save results,
    and apply coloring based on experiment labels.

    Args:
        experiments_dic (Dict[str, Dict[str, Any]]): Dictionary containing experiment configurations.
    """
    # Run the main function for each experiment
    for experiment_label, experiment_detail in experiments_dic.items():
        print(f"\nRunning experiment: {experiment_label}")

        # Main asynchronous run, which will produce a new file named:
        #    originalBaseName_output.xlsx
        await main_async_EXCT_MARIA_multiagent(
            excel_file_path=experiment_detail["excel_file_path"],
            experiment_label=experiment_label,
            Pydantic_Objects_List=experiment_detail["Pydantic_Objects_List"],
            model_engine=experiment_detail.get("model_engine", "OpenAI_Async"),
            parser_error_handling=experiment_detail.get("parser_error_handling", "llm_to_correct"),
            open_at_end=False,  # We'll open the file ourselves below
            model=experiment_detail.get("model", "gpt-3.5-turbo"),
            pre_prompt=experiment_detail.get("pre_prompt", ""),
            temperature=experiment_detail.get("temperature", 0),
            max_tokens=experiment_detail.get("max_tokens", 2048),
            logprobs=experiment_detail.get("logprobs", False),
            seed=experiment_detail.get("seed", None),
            timeout=experiment_detail.get("timeout", 60),
            max_retries=experiment_detail.get("max_retries", 2),
            total_async_n=experiment_detail.get("total_async_n", 10),
            runpod_base_url=experiment_detail.get("runpod_base_url", ""),
            runpod_api=experiment_detail.get("runpod_api", ""),
            azure_api_key=experiment_detail.get("azure_api_key", os.getenv("AZURE_OPENAI_API_KEY")),
            azure_endpoint=experiment_detail.get("azure_endpoint", os.getenv("AZURE_OPENAI_ENDPOINT")),
            azure_api_version=experiment_detail.get("azure_api_version", os.getenv("AZURE_API_VERSION")),
            text_column=experiment_detail.get("text_column", "All_Report_Text"),
            textPath_column=experiment_detail.get("textPath_column", None),
        )

    # After each experiment is completed, color the newly created output file
    # because main_async_EXCT_MARIA_multiagent saves to "baseName_output.xlsx"
    for experiment_label, experiment_detail in experiments_dic.items():
        original_path = experiment_detail['excel_file_path']
        base_name, ext = os.path.splitext(original_path)
        new_file_path = base_name + "_output.xlsx"

        color_cells_in_excel(
            new_file_path,
            experiment_detail["experiment_aRGB_cellcolor"],
            experiment_label
        )

    # Optionally open the new file from the FIRST experiment if 'open_at_end' is True
    first_experiment_label = next(iter(experiments_dic))
    if experiments_dic[first_experiment_label].get('open_at_end', True):
        original_path = experiments_dic[first_experiment_label]['excel_file_path']
        base_name, ext = os.path.splitext(original_path)
        new_file_path = base_name + "_output.xlsx"
        open_excel_file(new_file_path)

# Finally, run the experiments
asyncio.run(run_experiment(experiments_dic))