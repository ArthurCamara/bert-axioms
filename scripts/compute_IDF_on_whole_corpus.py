import multiprocessing as mp
import os
import sys
import argparse
import pickle
from tqdm.auto import tqdm
from pytorch_transformers import BertTokenizer
from collections import Counter
mp.set_start_method('spawn', True)


def process_chunk(chunk_no, block_offset, inf, no_lines, args, model=None):
    DFS = Counter()
    tokenizer = BertTokenizer.from_pretrained(model)
    position = chunk_no + 1

    pbar = tqdm(total=no_lines, desc="Running for chunk {}".format(
        str(chunk_no).zfill(2)), position=position)
    with open(inf, 'r') as f:
        f.seek(block_offset[chunk_no])
        for i in range(no_lines):
            doc = " ".join(f.readline().split("\t")[1:])
            tokens = set(tokenizer.tokenize(doc)[:512])
            for w in tokens:
                DFS[w] += 1
            pbar.update()
    pbar.close()
    pickle.dump(DFS, open(os.path.join(
        args.output_dir, "IDFS-{}".format(chunk_no)), 'wb'))


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
            "--docs_file", "/ssd2/arthur/TREC2019/data/docs/tokenized-msmarco-docs.tsv",
            "--output_dir", "/ssd2/arthur/TREC2019/data/docs/IDFS/",
            "--n_threads", "50"
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
    pbar = tqdm(total=cpus, position=0)
    model = 'bert-base-uncased'

    def update(*a):
        pbar.update()
    if args.single_thread:
        process_chunk(0, block_offset, args.docs_file,
                      lines_per_chunk, args, model)
        sys.exit(0)

    pool = mp.Pool(cpus)
    jobs = []
    for i in range(len(block_offset)):
        jobs.append(pool.apply_async(process_chunk, args=(
            i, block_offset, args.docs_file, lines_per_chunk, args, model), callback=update))
    for job in jobs:
        job.get()
    pool.close()
    pbar.close()

    # join results
    full_IDFS = Counter()
    for i in range(len(block_offset)):
        _idf = pickle.load(open(os.path.join(args.output_dir, "IDFS-{}".format(i)), 'rb'))
        for k in _idf:
            full_IDFS[k] += _idf[k]
    pickle.dump(full_IDFS, open(os.path.join(args.output_dir, "IDFS-FULL"), 'wb'))