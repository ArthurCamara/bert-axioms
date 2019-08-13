import os
from tqdm.auto import tqdm
import csv
import gzip
import argparse


def load_querystring(queries_file):
    querystring = {}
    with gzip.open(queries_file, 'rt', encoding='utf-8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, querystring_of_topicid] in tsvreader:
            querystring[topicid] = querystring_of_topicid
    return querystring


def load_docoffset(lookup_file):
    docoffset = {}
    with gzip.open(lookup_file, 'rt', encoding='utf-8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [docid, _, offset] in tsvreader:
            docoffset[docid] = int(offset)
    return docoffset


def load_qrels(qrels_file):
    qrel = {}
    with open(qrels_file, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, _, docid, rel] in tsvreader:
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]
    return qrel


def getcontent(docid, f, docoffset):
    """getcontent(docid, f) will get content for a given docid (a string) from filehandle f.
    The content has four tab-separated strings: docid, url, title, body.
    """

    f.seek(docoffset[docid])
    line = f.readline()
    assert line.startswith(docid + "\t"), \
        f"Looking for {docid}, found {line}"
    return line.rstrip()


def trec_run_to_bert(run_file, data_home, split, run_file_len=None, top_k=100):
    queries_file = os.path.join(
        data_home, f"msmarco-doc{split}-queries.tsv.gz")
    lookup_file = os.path.join(data_home, "msmarco-docs-lookup.tsv.gz")
    qrels_file = os.path.join(data_home, f"msmarco-doc{split}-qrels.tsv")

    qrel = load_qrels(qrels_file)
    querystring = load_querystring(queries_file)
    docoffset = load_docoffset(lookup_file)

    docs_file = os.path.join(data_home, "msmarco-docs.tsv")
    output_file = os.path.join(
        data_home, "bm25_bert_{}_top-{}.tsv".format(split, top_k))
    # query_id, document_id, querystring, document text
    output_line_format = "{}-{}\t{}\t{}\t{}\n"
    with open(run_file, 'r') as inf,\
            open(docs_file, 'r', encoding='utf-8') as f, \
            open(output_file, 'w', encoding='utf-8') as outf:
        for line in tqdm(inf, total=run_file_len):
            [topic_id, _, doc_id, ranking, score, _] = line.split()
            is_relevant = doc_id in qrel[topic_id]
            query = querystring[topic_id]
            document = getcontent(doc_id, f, docoffset)
            text_to_keep = ' '.join(document.split("\t")[1:])
            outf.write(output_line_format.format(
                topic_id, doc_id, query, text_to_keep, int(is_relevant)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_file", required=True,
                        type=str, help="TREC-formatted run file")
    parser.add_argument("--data_home", required=True, type=str,
                        help="Data home, with docs, queries and lookup")
    parser.add_argument("--split", required=True, type=str,
                        help="Split to be used. Train or Dev.")
    parser.add_argument("--run_file_len", default=None,
                        help="Number of lines in the run file")
    parser.add_argument("--top_k", default=100, type=int,
                        help="TOP-k documents that were used.")
    args = parser.parse_args()
    trec_run_to_bert(args.run_file, args.data_home,
                     args.split, args.run_file_len, args.top_k)
