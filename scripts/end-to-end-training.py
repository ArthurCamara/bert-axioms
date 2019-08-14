"""This script implements an end-to-end feature extraction and training for MSMarco dataset."""

import argparse
import logging
import subprocess
import os
from TREC_run_to_BERT import trec_run_to_bert
from msmarco_dataset import MsMarcoDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        required=True, help="Data home")
    parser.add_argument("--k", type=int, default=100,
                        help="Top-k for reranking")
    parser.add_argument("--log_level", type=str, default="INFO",
                        help="Logging level. Defaults to INFO")
    parser.add_argument("--anserini_path", type=str,
                        default="/ssd2/arthur/Anserini/anserini.jar", help="anserini.jar path")
    parser.add_argument("--overwrite", action="store_true",
                        help="Force overwrite of everything")
    args = parser.parse_args()

    logging.basicConfig(level=logging.getLevelName(args.log_level))
    # today = str(datetime.date.today())

    def get_path(x):
        return os.path.join(args.data_dir, x)

    # First, check if index already exists. If not, ask for one.
    index_path = get_path("lucene-index.msmarco")
    assert os.path.isdir(
        index_path), "Index named {} not found!".format(index_path)

    # Run retrieval step for trainning dataset
    train_rank_path = "msmarco-train-top_{}.run".format(args.k)
    if not os.path.isfile(train_rank_path) or args.overwrite:
        topics_file = get_path("msmarco-doctrain-queries.tsv")
        assert os.path.isfile(topics_file), "could not find topics file {}".format(topics_file)
        output_path = get_path("msmarco_train_top-{}_bm25.run".format(args.k))

        cmd = """java -cp {} io.anserini.search.SearchCollection -topicreader Tsv\
             -index {} -topics {} -output {} -bm25 -k1=3.44 -b=0.87""".format(
            args.anserini_path, index_path, topics_file, output_path)
        subprocess.run(cmd)

    # count number of lines in generated file above
    with open(output_path) as f:
        for i, _ in enumerate(f):
            pass
    line_counter_train = i

    logging.info("Run file for train has %d lines", line_counter_train)
    # Generated BERT-Formatted triples
    trec_run_to_bert(output_path, args.data_home, 'train', line_counter_train, args.k)

    # Run feature extraction for training. This may take a while.
    # train_dataloader = load_data


if __name__ == "__main__":
    main()
