from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import sys
import random
from tqdm.autonotebook import tqdm, trange
import argparse

import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

import pickle

from pytorch_transformers import BertForNextSentencePrediction, BertTokenizer

logger = logging.getLogger(__name__)

from run_classifier_dataset_utils import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskname", default="msmarco", type="str", 
                                         help="Task name")
    parser.add_argument("--do_train", action="store_true",
                                         help="should we extract features from train?" )
    parser.add_argument("--do_eval", action="store_true", help="should we extract features from eval?" )



if __name__ == "__main__":
    main()