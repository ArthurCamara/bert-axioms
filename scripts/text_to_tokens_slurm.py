import multiprocessing as mp
from multiprocessing import current_process
import os
import sys
import csv
import argparse
import gzip
import subprocess
import logging
from tqdm.auto import tqdm
from pytorch_transformers import BertTokenizer, XLNetTokenizer


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


def _truncate_seq_pair(tokens_a, tokens_b, max_length=509):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def text_to_tokens(query, document, tokenizer, xlnet=False):
    tokens_a = tokenizer.tokenize(query)
    tokens_b = tokenizer.tokenize(document)
    _truncate_seq_pair(tokens_a, tokens_b)
    if xlnet:
        tokens = tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"] + ["[CLS]"]
    else:
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
    return tokens


def process_chunk(chunk_no, block_offset, inf, no_lines, args, model=None):
    current = current_process()
    lines = []
    with open(inf, 'r') as f:
        f.seek(block_offset[chunk_no])
        for i in range(no_lines):
            lines.append(f.readline().strip())
    if args.XLNet:
        tokenizer = XLNetTokenizer.from_pretrained(os.path.join(args.data_home, "models"))
        if tokenizer is None:
            logging.info("Loading model from %s", model)
            tokenizer = XLNetTokenizer.from_pretrained(model)
            if tokenizer is None:
                print("FUUUUCK")
                sys.exit(0)
    else:
        tokenizer = BertTokenizer.from_pretrained(
            os.path.join(args.data_home, "models"))
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained(model)
    output_line_format = "{}-{}\t{}\t{}\n"
    with open("{}/{}-triples.{}".format(args.data_home, args.split, chunk_no), 'w', encoding='utf-8') as outf:
        if current.name == "MainProcess":
            position = 1
        else:
            position = current._identity[0]
        with tqdm(total=len(lines), position=position) as progress_bar:
            for counter, line in tqdm(enumerate(lines),
                                      desc="running for {}".format(str(chunk_no).zfill(2)),
                                      position=position):
                try:
                    [topic_id, _, doc_id, ranking, score, _] = line.split()
                except:
                    continue
                is_relevant = doc_id in qrel[topic_id]
                query = querystring[topic_id]
                document = getcontent(doc_id, docs_file)
                tokenized = text_to_tokens(query, document, tokenizer, args.XLNet)
                outf.write(output_line_format.format(
                    topic_id, doc_id, tokenized, int(is_relevant)))
                progress_bar.update(1)


if __name__ == "__main__":
    mp.set_start_method('fork', True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--data_home", type=str,
                        default="/ssd2/arthur/TREC2019/data/")
    parser.add_argument("--run_file", type=str, required=True),
    parser.add_argument("--single_thread", action="store_true"),
    parser.add_argument("--XLNet", action="store_true")
    logging.basicConfig(level=logging.getLevelName("INFO"))
    if len(sys.argv) > 3:
        args = parser.parse_args(sys.argv[1:])
    else:
        argv = [
            "--split", "train",
            "--run_file", "tiny-top100",
            "--XLNet"
        ]

        args = parser.parse_args(argv)
    data_home = args.data_home
    run_file = os.path.join(args.data_home, args.run_file)

    queries_file = os.path.join(
        data_home, f"msmarco-doc{args.split}-queries.tsv.gz")
    if not os.path.isfile(queries_file):
        queries_file = queries_file.replace(".gz", "")

    lookup_file = os.path.join(data_home, "msmarco-docs-lookup.tsv.gz")
    if not os.path.isfile(lookup_file):
        lookup_file = lookup_file.replace(".gz", "")

    qrels_file = os.path.join(
        data_home, f"msmarco-doc{args.split}-qrels.tsv.gz")
    if not os.path.isfile(qrels_file):
        qrels_file = qrels_file.replace(".gz", "")

    docs_file = os.path.join(data_home, "msmarco-docs.tsv")
    number_of_lines_to_process = int(subprocess.check_output(
        "wc -l {}".format(run_file).split()).decode("utf=8").split()[0])
    querystring = {}
    n_topics = 0

    if queries_file.endswith(".gz"):
        with gzip.open(queries_file, 'rt', encoding='utf-8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [topicid, querystring_of_topicid] in tsvreader:
                querystring[topicid] = querystring_of_topicid
                n_topics += 1
    else:
        with open(queries_file, 'r', encoding='utf-8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [topicid, querystring_of_topicid] in tsvreader:
                querystring[topicid] = querystring_of_topicid
                n_topics += 1
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
    qrel = {}

    if qrels_file.endswith(".gz"):
        with gzip.open(qrels_file, 'rt', encoding='utf8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [topicid, _, docid, rel] in tsvreader:
                assert rel == "1"
                if topicid in qrel:
                    qrel[topicid].append(docid)
                else:
                    qrel[topicid] = [docid]
    else:
        with open(qrels_file, 'r', encoding='utf8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [topicid, _, docid, rel] in tsvreader:
                assert rel == "1"
                if topicid in qrel:
                    qrel[topicid].append(docid)
                else:
                    qrel[topicid] = [docid]
    # pre-process positions for each chunk
    cpus = mp.cpu_count()

    if args.single_thread:
        cpus = 1
    print("running with {} cpus".format(cpus))
    number_of_chunks = cpus
    block_offset = dict()
    lines_per_chunk = number_of_lines_to_process // cpus
    print("{}  lines per chunk".format(lines_per_chunk))
    excess_lines = number_of_lines_to_process % cpus
    start = 0
    if cpus <2:
        block_offset [0]=[0]
    else:
        with open(run_file) as f:
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
    if args.XLNet:
        model = "xlnet-base-cased"
        args.split = args.split + "-XLNet"
    else:
        model = "bert-base-uncased"

    def update(*a):
        pbar.update()
    if args.single_thread:
        process_chunk(0, block_offset, run_file, lines_per_chunk, args, model)
        sys.exit(0)
    pool = mp.Pool(cpus)
    jobs = []

    for i in tqdm(range(len(block_offset))):
        jobs.append(pool.apply_async(process_chunk, args=(
            i, block_offset, run_file, lines_per_chunk, args, model), callback=update))
    for job in jobs:
        job.get()
    pool.close()
