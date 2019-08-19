import multiprocessing as mp
import os
import csv
import argparse
import gzip
import subprocess
from tqdm.auto import tqdm
from pytorch_transformers import BertTokenizer


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


def text_to_tokens(query, document, tokenizer):
    tokens_a = tokenizer.tokenize(query)
    tokens_b = tokenizer.tokenize(document)
    _truncate_seq_pair(tokens_a, tokens_b)
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
    return tokens


def process_chunk(chunk_no, block_offset, inf, no_lines):
    lines = []
    with open(inf, 'r') as f:
        f.seek(block_offset[chunk_no])
        for i in range(no_lines):
            lines.append(f.readline().strip())
    tokenizer = BertTokenizer.from_pretrained(
        os.path.join(data_home, "models"))
    output_line_format = "{}-{}\t{}\t{}\n"
    with open("{}/{}-triples.{}".format(args.data_home, args.split, chunk_no), 'w', encoding='utf-8') as outf:
        for line in tqdm(lines):
            [topic_id, _, doc_id, ranking, score, _] = line.split()
            is_relevant = doc_id in qrel[topic_id]
            query = querystring[topic_id]
            document = getcontent(doc_id, docs_file)
            tokenized = text_to_tokens(query, document, tokenizer)
            outf.write(output_line_format.format(
                topic_id, doc_id, tokenized, int(is_relevant)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--data_home", type=str,
                        default="/ssd2/arthur/TREC2019/data/")
    parser.add_argument("--run_file", type=str,
                        default="msmarco-doctrain-top100")

    args = parser.parse_args()

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
    print("running with {} cpus".format(cpus))
    number_of_chunks = cpus
    block_offset = dict()
    lines_per_chunk = number_of_lines_to_process // cpus
    print("{}  lines per chunk".format(lines_per_chunk))
    excess_lines = number_of_lines_to_process % cpus
    start = 0
    with open(run_file) as f:
        current_chunk = 0
        counter = 0
        line = f.readline()
        while(line):
            if (counter + 1) % lines_per_chunk == 0:
                block_offset[current_chunk] = f.tell()
                print(current_chunk, block_offset[current_chunk])
                current_chunk += 1
            line = f.readline()

            counter += 1
    pbar = tqdm(total=cpus)

    def update(*a):
        pbar.update()
    pool = mp.Pool(cpus)
    jobs = []
    for i in tqdm(range(cpus)):
        jobs.append(pool.apply_async(process_chunk, args=(
            i, block_offset, run_file, lines_per_chunk, ), callback=update))
    for job in jobs:
        job.get()
    pool.close()
