import random
import logging
import os


def split(config):
    random.seed(config.seed)
    queries_file = os.path.join(config.data_home, "queries/msmarco-docdev-queries.tsv.tokenized")
    bert_queries_file = os.path.join(config.data_home, "queries/msmarco-docdev-queries.tsv.bert")
    qrels_file = os.path.join(config.data_home, "qrels/msmarco-docdev-qrels.tsv")
    assert os.path.isfile(queries_file)
    assert os.path.isfile(qrels_file)

    new_queries_dev_file = os.path.join(config.data_home, "queries/dev.tokenized.tsv")
    new_queries_test_file = os.path.join(config.data_home, "queries/test.tokenized.tsv")
    new_bert_queries_dev_file = os.path.join(config.data_home, "queries/dev.tokenized.bert.tsv")
    new_bert_queries_test_file = os.path.join(config.data_home, "queries/test.tokenized.bert.tsv")

    new_qrels_dev_file = os.path.join(config.data_home, "qrels/dev.tsv")
    new_qrels_test_file = os.path.join(config.data_home, "qrels/test.tsv")

    if os.path.isfile(new_queries_dev_file) and os.path.isfile(new_queries_test_file):
        logging.info("Already found dev and test split")
        return

    # Get list of queries first
    all_queries = set()
    for line in open(queries_file):
        query_id, query = line.split("\t")
        all_queries.add(query_id)
    selected_queries = random.sample(all_queries, int(config.split_percentage * len(all_queries)))
    selected_queries = set(selected_queries)
    with open(new_queries_test_file, 'w') as tfile, open(new_queries_dev_file, 'w') as dfile, open(new_bert_queries_test_file, 'w') as btfile, open(new_bert_queries_dev_file, 'w') as bdfile:
        for line, bline in zip(open(queries_file), open(bert_queries_file)):
            query_id, _ = line.split("\t")
            bquery_id, _ = bline.split("\t")
            assert query_id == bquery_id, "Mismatch between bert query id and query id at %s,%s" % (queries_file, bert_queries_file)
            if query_id in selected_queries:
                btfile.write(bline)
                tfile.write(line)
            else:
                bdfile.write(bline)
                dfile.write(line)
    with open(new_qrels_dev_file, 'w') as dfile, open(new_qrels_test_file, 'w') as tfile:
        for line in open(qrels_file):
            topic_id, _, doc_id, label = line.split("\t")
            if topic_id in selected_queries:
                tfile.write(line)
            else:
                dfile.write(line)
