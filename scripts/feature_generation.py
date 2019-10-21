import os
import random
from collections import defaultdict
import pickle
import logging
from tqdm.auto import tqdm


def get_content(doc_id, doc_file, offset_dict):
    offset = offset_dict[doc_id]
    with open(doc_file) as f:
        f.seek(offset)
        doc = f.readline()
    return doc


def generate_docs_offset(doc_file, config):
    offset_path = doc_file + ".offset"
    if os.path.isfile(offset_path):
        return pickle.load(open(offset_path, 'rb'))
    offset_dict = dict()
    pbar = tqdm(total=config.corpus_size)
    with open(doc_file) as inf:
        location = 0
        line = True
        while line:
            line = inf.readline()
            doc_id, _ = line.split("\t")
            offset_dict[doc_id] = location
            location = inf.tell()
            pbar.update()
    return offset_dict


def generate_features(config, cut, split):
    relevants = defaultdict(lambda: set())
    qrel_file = os.path.join(config.data_home, "qrels/{}.tsv".format(split))
    for line in open(qrel_file):
        topic_id, _, doc_id, label = line.strip().split("\t")
        if label == '1':
            relevants[topic_id].add(doc_id)

    QL_run_file = os.path.join(config.data_home, "runs/QL_{}-{}.run".format(split, cut))
    negative_docs = defaultdict(lambda: set())
    for line in open(QL_run_file, 'r'):
        topic_id, _, doc_id,  _, _, _ = line.split()
        if doc_id not in relevants[topic_id]:
            negative_docs[topic_id].add(doc_id)
    logging.info("Read %i positive samples", len(relevants))

    assert len(negative_docs) == len(relevants)
    # generate negative sampling -> We are assuming only one positive per query.
    triples = []
    for topic_id in relevants:
        for relevant_doc in relevants[topic_id]:  # Probably just one
            triples.append((topic_id, relevant_doc, 1))
            negative_samples = random.sample(negative_docs[topic_id], k=config.negative_samples)
            for n in negative_samples:
                triples.append((topic_id, n, 0))
    logging.info("Final sample has %i samples", len(triples))
    assert len(triples) == (1 + config.negative_samples) * len(relevants)
    
    # load queries
    queries_path = os.path.join(config.data_home, "queries/{}.tokenized.tsv".format(split))
    queries = dict()
    for line in open(queries_path):
        topic_id, query = line.split("\t")
        queries[topic_id] = query
    assert len(queries) == len(relevants)
    
    # Prepare docs
    docs_file = os.path.join(config.data_home, "docs/msmarco-docs.tokenized.{}.tsv".format(cut))
    docs_offset = generate_docs_offset(docs_file, config)

    # Actually generate triples for training
    if not os.path.isdir(os.path.join(config.data_home, "triples")):
        os.mkdir(os.path.join(config.data_home, "triples"))

    output_file = os.path.join(config.data_home, "triples/{}-{}.tsv".format(split, cut))
    with open(output_file) as outf:
        for topic_id, doc_id, label in triples:
            query_text = queries[topic_id]
            doc_text = get_content(doc_id, docs_file, docs_offset)
            outf.write("{}\t{}\t{}\n".format(query_text, doc_text, label))
