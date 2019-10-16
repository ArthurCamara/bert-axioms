"""This script implemets end-to-end dataset extraction and testing for axioms.
Expected inputs are:
-> Full MsMarco docs corpora.
-> Train and development queries

Parameters are expected on file config-defaults.yaml. wandb deals with them.
Multiple checkpoints will be saved through the code, to make thinks faster on future runs with similar params
"""
from indri import generate_index
import logging
from data_fetch import fetch_data
from tokenization import tokenize_queries, tokenize_docs
# from bunch import Bunch
# import yaml
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import wandb  # noqa


wandb.init(project="axiomatic-bert")
config = wandb.config

# config = yaml.load(open("params.yaml"), Loader=yaml.FullLoader)
# config = Bunch(config)


def main():
    # First step, fetch data
    fetch_data(config)
    # Tokenize queries
    tokenize_queries(config)
    # Tokenize docs
    tokenize_docs(config)
    # Index documents on Indri
    generate_index(config, full=True)
    # Run Indri QL-FULL
    # Report nDCGs
    # Index short Documents on Indri
    # Run Indri-QL Cut
    # Create top 100 triples files
    # Create negative sampling triples file
    # Fit DistilBERT

    # DONE PRE PROCESSING

    # Predict values for test file on BERT
    # Report nDCG values for BERT
    # Create datasets
    # Run dataset check scripts for each dataset

if __name__ == "__main__":
    # Set logger
    level = logging.getLevelName(config.logging_level)
    Log = logging.getLogger()
    Log.setLevel(level)
    main()
