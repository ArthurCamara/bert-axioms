"""This script implements an end-to-end feature extraction and training for MSMarco dataset."""

import logging
import subprocess
import os
from msmarco_dataset import MsMarcoDataset
from trecrun_to_bert import TRECrun_to_BERT
from args_parser import getArgs
from sentence_level_classifier import fine_tune


def get_path(home_dir, x):
    return os.path.join(home_dir, x)


def run_retrieval_step(data_dir, k, anserini_path, overwrite=False):
    """Runs a retrieval step using Anserini"""
    index_path = get_path(data_dir, "lucene-index.msmarco")
    assert os.path.isdir(
        index_path), "Index named {} not found!".format(index_path)

    # Run retrieval step for trainning dataset
    train_rank_path = "msmarco-train-top_{}.run".format(k)
    if not os.path.isfile(train_rank_path) or overwrite:
        topics_file = get_path(data_dir, "msmarco-doctrain-queries.tsv")
        assert os.path.isfile(
            topics_file), "could not find topics file {}".format(topics_file)
        output_path = get_path(
            data_dir, "msmarco_train_top-{}_bm25.run".format(k))

        cmd = """java -cp {} io.anserini.search.SearchCollection -topicreader Tsv\
             -index {} -topics {} -output {} -bm25 -k1=3.44 -b=0.87""".format(
            anserini_path, index_path, topics_file, output_path)
        subprocess.run(cmd)

    # count number of lines in generated file above
    with open(output_path) as f:
        for i, _ in enumerate(f):
            pass
    line_counter_train = i

    logging.info("Run file for train has %d lines", line_counter_train)
    # Generated BERT-Formatted triples
    TRECrun_to_BERT(output_path, data_dir, 'train', line_counter_train, k)


def main():
    # args = getArgs(sys.argv[1:])
    data_dir = "/ssd2/arthur/insy/msmarco/data"
    argv = [
        "--data_dir", data_dir,
        "--train_file", data_dir + "/train-triples.0",
        "--dev_file", data_dir + "/dev-triples.0",
        "--bert_model", "bert-base-uncased"
    ]
    args = getArgs(argv)
    logging.basicConfig(level=logging.getLevelName(args.log_level))

    if args.run_retrieval:
        run_retrieval_step(args.data_dir, args.k,
                           args.anserini_path, args.overwrite)

    # Dataset loader
    train_dataset = MsMarcoDataset(args.train_file, args.data_dir)
    dev_dataset = MsMarcoDataset(args.dev_file, args.data_dir)

    # Fine tune
    fine_tune(train_dataset, dev_dataset, args.data_dir, n_workers=0)


if __name__ == "__main__":
    main()
