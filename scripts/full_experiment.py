"""This script implemets end-to-end dataset extraction and testing for axioms.
Expected inputs are:
-> Full MsMarco docs corpora.
-> Train and development queries

Parameters are expected on file config-defaults.yaml. wandb deals with them.
Multiple checkpoints will be saved through the code, to make things faster on future runs with similar params
"""
import warnings
import logging
from indri import generate_index, run_queries
from data_fetch import fetch_data
from tokenization import tokenize_queries, tokenize_docs
from split import split
from cut_dataset import cut_docs
from feature_generation import generate_features
import os
from bert import generate_run_file, fit_bert
from Dataset_Extraction import extract_datasets
from axioms import check_axioms
warnings.filterwarnings("ignore")

os.environ['WANDB_MODE'] = 'dryrun'
import wandb  # noqa
wandb.init(project="axiomatic-bert")
config = wandb.config


def main():
    # Fetch data if not already there
    # fetch_data(config)

    # tokenize queries and documents
    # tokenize_queries(config)
    tokenize_docs(config)

    # Cut documents for BERT
    cut_docs(config)

    # Indexes
    # generate_index(config, full=True)
    generate_index(config, full=False)

    # split dev-test queries
    split(config)
    run_queries(config, "test", False)
    run_queries(config, "dev", False)
    run_queries(config, "train", False)

    # run_queries(config, "test", True)
    # run_queries(config, "dev", True)
    # run_queries(config, "train", True)

    # Generate features
    generate_features(config, "cut", "train")
    generate_features(config, "cut", "dev")
    generate_features(config, "cut", "test")
    # generate_features(config, "full", "test")
    # generate_features(config, "full", "dev")
    # generate_features(config, "full", "train")

    # Fit DistilBERT
    fit_bert(config, "cut")
    # fit_bert(config, "full")

    generate_run_file("test", "cut")
    # generate_run_file("test", "full")

    # Create datasets
    extract_datasets("cut")

    # Check axiomatic performance
    check_axioms("cut")


if __name__ == "__main__":
    # Set logger
    level = logging.getLevelName(config.logging_level)
    Log = logging.getLogger()
    Log.setLevel(level)
    main()
