import os
import gzip
import csv
import random
from collections import defaultdict
from tqdm import tqdm_notebook as tqdm

import argparse

def getcontent(docid, f):
    """getcontent(docid, f) will get content for a given docid (a string) from filehandle f.
    The content has four tab-separated strings: docid, url, title, body.
    """

    f.seek(docoffset[docid])
    line = f.readline()
    assert line.startswith(docid + "\t"), \
        f"Looking for {docid}, found {line}"
    return line.rstrip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, type=str)
    parser.add_argument("--data_home", required=True, type=str)
    parser.add_argument("--n_neg_to_keep", type=int, default=4)
    args = parser.parse_args()
    data_home = args.data_home
    split = args.split

    queries_file = os.path.join(data_home, f"msmarco-doc{split}-queries.tsv.gz")
    assert os.path.isfile(queries)
    lookup_file = os.path.join(data_home, "msmarco-docs-lookup.tsv.gz")
    assert os.path.isfile(lookup_file)
    qrels_file = os.path.join(data_home, f"msmarco-doc{split}-qrels.tsv.gz")
    assert os.path.isfile(qrels_file)
    top_100_file = os.path.join(data_home, f"msmarco-doc{split}-top100.gz")
    assert os.path.isfile(top_100_file)
    docs_file = os.path.join(data_home, "msmarco-docs.tsv")
    assert os.path.isfile(docs_file)
    outfile = os.path.join(data_home, f'{split}-samples.tsv')
    assert os.path.isfile(outfile)

    querystring = {}
    with gzip.open(queries_file, 'rt', encoding='utf-8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, querystring_of_topicid] in tsvreader:
            querystring[topicid] = querystring_of_topicid

    docoffset = {}
    with gzip.open(lookup_file, 'rt', encoding='utf-8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [docid, _, offset] in tsvreader:
            docoffset[docid] = int(offset)

    qrel = {}
    with gzip.open(qrels_file, 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, _, docid, rel] in tsvreader:
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]

    stats = defaultdict(int)
    n_negatives_for_topic_id = 0
    lines_written = 0

    if split=="dev":
        total_lines = 519300
    elif split=="train":
        total_lines = 36701116
    unjudged_rank_to_keep = random.choices(range(1,100), k=args.n_neg_to_keep)
    positive_sample_done = set()
    with gzip.open(top_100_file, 'rt', encoding='utf-8') as top100f, open(docs_file, encoding='utf-8') as f, open(outfile, 'w', encoding='utf-8') as outf:
        for line in tqdm(top100f, total=total_lines):
            [topicid, _, unjudged_docid, rank, _, _ ] = line.split()
            #Either we have already processed this topic or this is an unjudged sample we don't want to keep
            if n_negatives_for_topic_id >= n_neg_to_keep or int(rank) not in unjudged_rank_to_keep:
                stats['skipped']+=1
                if n_negatives_for_topic_id >= n_neg_to_keep:
                    n_negatives_for_topic_id = 0
                continue

            unjudged_rank_to_keep = random.choices(range(1,100), k=n_neg_to_keep)
            n_negatives_for_topic_id +=1
            assert topicid in querystring
            assert topicid in qrel
            assert unjudged_docid in docoffset

            positive_docid = random.choice(qrel[topicid])
            assert positive_docid in docoffset

            #if this positive example is also in the top100
            if unjudged_docid in qrel[topicid]:
                stats['docid_colission'] +=1
                continue
            stats['kept'] +=1
            #print this negative sample

            #if this is a positive example, and we haven't added it to the file yet
            if topicid not in positive_sample_done:
                lines_written += 1
                positive_sample_done.add(topicid)
                positive_document = getcontent(positive_docid, f)
                text_to_keep = ' '.join(positive_document.split("\t")[1:])
                docid = positive_document.split("\t")[0]
                outf.write(f"{topicid}-{docid}\t{querystring[topicid]}\t{text_to_keep}\t0\n")

            #add this negative example
            lines_written +=1
            negative_document = getcontent(unjudged_docid, f)
            text_to_keep = ' '.join(negative_document.split("\t")[1:])
            docid = negative_document.split("\t")[0]
            outf.write(f"{topicid}-{docid}\t{querystring[topicid]}\t{text_to_keep}\t1\n")
            if lines_written % 1000 == 0:
                print(lines_written)

main()
