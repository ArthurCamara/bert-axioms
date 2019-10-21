"""This script implemets end-to-end dataset extraction and testing for axioms.
Expected inputs are:
-> Full MsMarco docs corpora.
-> Train and development queries

Parameters are expected on file config-defaults.yaml. wandb deals with them.
Multiple checkpoints will be saved through the code, to make thinks faster on future runs with similar params
"""
import warnings
from indri import generate_index, run_queries
import logging
from data_fetch import fetch_data
from tokenization import tokenize_queries, tokenize_docs
from split import split
from cut_dataset import cut_docs
from feature_generation import generate_features
warnings.filterwarnings("ignore")
import wandb  # noqa


wandb.init(project="axiomatic-bert")
config = wandb.config

def main():
    fetch_data(config)
    tokenize_queries(config)
    tokenize_docs(config)
    generate_index(config, full=True)
    split(config)
    run_queries(config, "test", False)
    cut_docs(config)
    generate_index(config, full=False)
    run_queries(config, "test", True)
    run_queries(config, "dev", True)
    run_queries(config, "train", True)
    generate_features(config, "cut", "train")
    generate_features(config, "cut", "dev")
    generate_features(config, "cut", "test")
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
