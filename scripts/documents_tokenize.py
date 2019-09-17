import multiprocessing as mp
from multiprocessing import current_process

import os
import sys
import csv
import argparse
from tqdm.auto import tqdm
from pytorch_transformers import BertTokenizer
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


def text_to_tokens(document, tokenizer):
    tokens = tokenizer.tokenize(document)
    return tokens


def process_chunk(chunk_no, block_offset, inf, no_lines, args, model=None):
    lines = []
    with open(inf, 'r') as f:
        f.seek(block_offset[chunk_no])
        for i in range(no_lines):
            lines.append(f.readline().strip())
    tokenizer = BertTokenizer.from_pretrained(os.path.join(args.data_home, "models"))
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(model)
    output_line_format = "{}\t{}\n"
    position = chunk_no + 1
    with open("{}/tokenized-docs.triples.{}".format(args.data_home, chunk_no), 'w', encoding='utf-8') as outf:
        for counter, line in tqdm(enumerate(lines),
                                  desc="running for {}".format(str(chunk_no).zfill(2)),
                                  total=len(lines),
                                  position=position):
            try:
                doc_id = line.split("\t")[0]
                document = " ".join(line.split("\t")[1:])
            except:
                continue
            tokenized = text_to_tokens(document, tokenizer)
            outf.write(output_line_format.format(doc_id, tokenized))


if __name__ == "__main__":
    mp.set_start_method('spawn', True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_home", type=str,
                        default="/ssd2/arthur/TREC2019/data/docs/")
    parser.add_argument("--single_thread", action="store_true"),
    parser.add_argument("--n_threads", type=int, default=1)
    if len(sys.argv) > 3:
        argv = sys.argv[1:]
    else:
        argv = [
            "--single_thread",
        ]
    args = parser.parse_args(argv)
    data_home = args.data_home

    lookup_file = os.path.join(data_home, "msmarco-docs-lookup.tsv")
    docs_file = os.path.join(data_home, "msmarco-docs.tsv")
    number_of_lines_to_process = 3213835

    docoffset = {}
    with open(lookup_file, 'r', encoding='utf-8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [docid, _, offset] in tsvreader:
            docoffset[docid] = int(offset)

    if args.n_threads > 0:
        cpus = min(mp.cpu_count(), args.n_threads)
    else:
        cpus = mp.cpu_count()

    if args.single_thread:
        cpus = 1

    print("running with {} cpus".format(cpus))
    excess_lines = number_of_lines_to_process % cpus
    number_of_chunks = cpus
    if excess_lines > 0:
        number_of_chunks = cpus - 1
    block_offset = dict()
    lines_per_chunk = number_of_lines_to_process // number_of_chunks
    print("{}  lines per chunk".format(lines_per_chunk))
    start = 0
    if cpus < 2:
        block_offset[0] = 0
    else:
        with open(docs_file) as f:
            current_chunk = 0
            counter = 0
            line = True
            while(line):
                if (counter) % lines_per_chunk == 0:
                    block_offset[current_chunk] = f.tell()
                    current_chunk += 1
                line = f.readline()

                counter += 1
    pbar = tqdm(total=cpus)
    model = "bert-base-uncased"

    def update(*a):
        pbar.update()
    if args.single_thread:
        process_chunk(0, block_offset, docs_file, lines_per_chunk, args, model)
        sys.exit(0)
    pool = mp.Pool(cpus)
    jobs = []

    for i in tqdm(range(len(block_offset))):
        jobs.append(pool.apply_async(process_chunk, args=(
            i, block_offset, docs_file, lines_per_chunk, args, model), callback=update))
    for job in jobs:
        job.get()
    pool.close()
