import multiprocessing as mp
from multiprocessing import current_process

import os
import sys
import argparse
import csv
import subprocess
import pickle
import gzip
from tqdm.auto import tqdm
from pytorch_transformers import BertTokenizer
from collections import Counter
mp.set_start_method('spawn', True)


def getcontent(docid, file_name):
    """getcontent(docid, f) will get content for a given docid (a string) from filehandle f.
    The content has four tab-separated strings: docid, url, title, body.
    """
    with open(file_name, encoding='utf-8') as f:
        f.seek(docoffset[docid])
        line = f.readline()
        assert line.startswith(docid + "\t"), \
            f"Looking for {docid}, found {line}"
    return line.rstrip()


def process_chunk(chunk_no, block_offset, inf, no_lines, args, model=None):
    current = current_process()
    lines = []
    with open(inf, 'r') as f:
        f.seek(block_offset[chunk_no])
        for i in range(no_lines):
            lines.append(f.readline().strip())

    tokenizer = BertTokenizer.from_pretrained(model)
    if current.name == "MainProcess":
        position = 1
    else:
        position = current._identity[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_file", type=str, required=True)
    parser.add_argument("--data_home", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--total_lines", type=int, default=3213835)
    parser.add_argument("--n_threads", type=int, default=-1)
    parser.add_argument("--single_thread", action="store_true")

    if len(sys.argv) > 2:
        args = parser.parse_args(sys.argv)
    else:
        argv = [
            "--data_home", "/ssd2/arthur/TREC2019/data",
            "--docs_file", "/ssd2/arthur/TREC2019/data/docs/msmarco-docs.tsv",
            "--output_dir", "/ssd2/arthur/TREC2019/data/docs/IDFS/"
        ]
        args = parser.parse_args(argv)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    if args.n_threads > 0:
        cpus = min(mp.cpu_count(), args.n_threads)
    else:
        cpus = mp.cpu_count()
    if args.single_thread:
        cpus = 1

    print("running with {} cpus".format(cpus))
    excess_lines = args.total_lines % cpus
    number_of_chunks = cpus
    if excess_lines > 0:
        number_of_chunks = cpus - 1
        excess_lines = args.total_lines % number_of_chunks
    block_offset = dict()
    lines_per_chunk = args.total_lines // number_of_chunks
    print("{}  lines per chunk".format(lines_per_chunk))
    print("{}  lines for last chunk".format(excess_lines))
    assert number_of_chunks * lines_per_chunk + excess_lines == args.total_lines

    lookup_file = os.path.join(args.data_home, "docs", "msmarco-docs-lookup.tsv.gz")
    if not os.path.isfile(lookup_file):
        lookup_file = lookup_file.replace(".gz", "")
    docoffset = {}
    if lookup_file.endswith(".gz"):
        with gzip.open(lookup_file, 'rt', encoding='utf-8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [docid, _, offset] in tsvreader:
                docoffset[docid] = int(offset)
    else:
        with open(lookup_file, 'r', encoding='utf-8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [docid, _, offset] in tsvreader:
                docoffset[docid] = int(offset)

    block_offset = {}
    start = 0
    if cpus < 2:
        block_offset[0] = 0
    else:
        with open(args.docs_file) as f:
            current_chunk = 0
            counter = 0
            line = True
            while(line):
                if counter % lines_per_chunk == 0:
                    block_offset[current_chunk] = f.tell()
                    current_chunk += 1
                line = f.readline()
                counter += 1
    pbar = tqdm(total=cpus)
    model = 'bert-base-uncased'

    def update(*a):
        pbar.update()
    if args.single_thread:
        process_chunk(0, block_offset, args.docs_file, args, model)
        sys.exit(0)

