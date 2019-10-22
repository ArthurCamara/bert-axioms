import os
import random
from collections import defaultdict
import pickle
import logging
from tqdm.auto import tqdm
import subprocess


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
    pbar = tqdm(total=config.corpus_size, desc="Generating doc offset dictionary")
    with open(doc_file) as inf:
        location = 0
        line = True
        while line:
            line = inf.readline()
            try:
                doc_id, _ = line.split("\t")
            except:
                continue
            offset_dict[doc_id] = location
            location = inf.tell()
            pbar.update()
    assert len(offset_dict) == config.corpus_size
    pickle.dump(offset_dict, open(offset_path, 'wb'))
    return offset_dict


def generate_features(config, cut, split):
    random.seed(config.seed)

    QL_run_file = os.path.join(config.data_home, "runs/QL_{}-{}.run".format(split, cut))
    triples_file = os.path.join(config.data_home, "triples/triples-{}-{}".format(split, cut))
    if os.path.isfile(triples_file) and "feature_generator" not in config.force_steps:
        triples = pickle.load(open(triples_file, 'rb'))
        logging.info("Loaded %i triples from %s", len(triples), triples_file)
        qrel_file = os.path.join(config.data_home, "qrels/{}.tsv".format(split))
        if split == "train":
            qrel_file = os.path.join(config.data_home, "qrels/msmarco-doctrain-qrels.tsv")
        expected_lines = int(subprocess.check_output(['wc', '-l', qrel_file]).decode("utf-8").split(" ")[0])
    else:
        relevants = defaultdict(lambda: set())
        qrel_file = os.path.join(config.data_home, "qrels/{}.tsv".format(split))
        if split == "train":
            qrel_file = os.path.join(config.data_home, "qrels/msmarco-doctrain-qrels.tsv")
        assert os.path.isfile(qrel_file), "QRELs file not found!"
        expected_lines = int(subprocess.check_output(['wc', '-l', qrel_file]).decode("utf-8").split(" ")[0])
        for line in tqdm(open(qrel_file), total=expected_lines, desc="loading qrels file for {}".format(split)):
            topic_id, _, doc_id, label = line.strip().split("\t")
            if label == '1':
                relevants[topic_id].add(doc_id)
        logging.info("Read %i positive samples", len(relevants))
        negative_docs = defaultdict(lambda: set())
        for line in tqdm(open(QL_run_file, 'r'), total=expected_lines * config.indri_top_k, desc="loading run file for {}".format(split)):
            topic_id, _, doc_id, _, _, _ = line.split()
            if doc_id not in relevants[topic_id]:
                negative_docs[topic_id].add(doc_id)

        assert len(negative_docs) == len(relevants)
        # generate negative sampling -> We are assuming only one positive per query.
        triples = []
        if split != "test":
            for topic_id in tqdm(relevants, desc="Generating triples"):
                for relevant_doc in relevants[topic_id]:  # Probably just one
                    triples.append((topic_id, relevant_doc, 1))
                    negative_samples = random.sample(negative_docs[topic_id], k=config.negative_samples)
                    for n in negative_samples:
                        triples.append((topic_id, n, 0))
            logging.info("Final sample has %i samples", len(triples))
            assert len(triples) == (1 + config.negative_samples) * len(relevants)
        # Test dataset. No negative sampling! Only the top100 needed.
        else:
            for line in open(QL_run_file, 'r'):
                topic_id, _, doc_id, _, _, _ = line.split()
                if doc_id in relevants[topic_id]:
                    label = 1
                else:
                    label = 0
                triples.append((topic_id, doc_id, label))
                logging.info("Final sample has %i samples", len(triples))
        if not os.path.isdir(os.path.join(config.data_home, "triples")):
            os.mkdir(os.path.join(config.data_home, "triples"))
        pickle.dump(triples, open(triples_file, 'wb'))

    # load queries in BERT format
    if split == "train":
        queries_path = os.path.join(config.data_home, "queries/msmarco-doctrain-queries.tsv.bert")
    else:
        queries_path = os.path.join(config.data_home, "queries/{}.tokenized.bert".format(split))
    
    assert os.path.isfile(queries_path), "Queries file not found at %s" % queries_path
    queries = dict()
    for line in open(queries_path):
        topic_id, query = line.split("\t")
        queries[topic_id] = query
    assert len(queries) == expected_lines
    
    # Prepare docs
    if cut == "cut":
        docs_file = os.path.join(config.data_home, "docs/msmarco-docs.tokenized.cut.bert")
    else:
        docs_file = os.path.join(config.data_home, "docs/msmarco-docs.tokenized.bert")
    docs_offset = generate_docs_offset(docs_file, config)

    # Actually generate triples for training
    if not os.path.isdir(os.path.join(config.data_home, "triples")):
        os.mkdir(os.path.join(config.data_home, "triples"))

    output_file = os.path.join(config.data_home, "triples/{}-{}.tsv".format(split, cut))
    if os.path.isfile(output_file) and "{}-{}-triples".format(split, cut) not in config.force_steps:
        logging.info("File %s already found. Skipping it" % output_file)
        return
    with open(output_file, 'w') as outf:
        for topic_id, doc_id, label in triples:
            query_text = queries[topic_id]
            doc_text = get_content(doc_id, docs_file, docs_offset)
            outf.write("{}\t{}\t{}\n".format(query_text, doc_text, label))
