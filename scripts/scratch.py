import os
from collections import defaultdict
import random
from tqdm.auto import tqdm
from pytorch_transformers import BertTokenizer
import pickle
import multiprocessing as mp
mp.set_start_method('spawn', True)

data_home = "/ssd2/arthur/TREC2019/data/"
offset_dict = pickle.load(
    open(os.path.join(data_home, "docs", "tokenized-msmarco-docs.tsv.offset"), 'rb'))
tokenized_docs = os.path.join(data_home, "docs",  "tokenized-msmarco-docs.tsv")


def getcontent(doc_id, docs_file):
    offset = offset_dict[doc_id]
    with open(docs_file) as f:
        f.seek(offset)
        doc = f.readline()
    return doc


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
split = "dev"
qrels_file = os.path.join(data_home, "qrels", "{}_qrels".format(split))

qrels = defaultdict(lambda: set())
for line in tqdm(open(qrels_file), desc="loading qrels"):
    query_id, _, doc_id, label = line.strip().split()
    tuple_id = "{}-{}".format(query_id, doc_id)
    if label == "1":
        qrels[query_id].add(doc_id)


def process_chunk(batch_no, n_negatives, chunk, split, input_path):
    batch = []
    for line_no, line in tqdm(enumerate(open(input_path, encoding="utf-8", errors="ignore")), desc="reading..."):
        if line_no < chunk[0]:
            continue
        if  line_no >= chunk[1]:
            break
        batch.append(line)
    last_topic = -1
    top_100 = []
    with open(os.path.join(data_home, "triples-tokenized", "10neg/{}-chunks.{}neg.{}".format(split, n_negatives, batch_no)), 'w') as outf:
        for line in tqdm(batch):
            guid, doc, label = line.strip().split("\t")
            topic_id, doc_id = guid.split("-")
            if topic_id != last_topic and last_topic != -1:
                for positive in qrels[topic_id]:
                    tokenized_positive = tokenizer.tokenize(getcontent(positive, tokenized_docs))
                    outf.write("{}-{}\t{}\t1\n".format(topic_id, positive, tokenized_positive))
                    random_neg = random.sample(top_100, n_negatives)
                    for neg in random_neg:
                        outf.write("{}\t{}\t0\n".format(neg[0], neg[1]))
                top_100 = []
            if label != "1":
                top_100.append((guid, eval(doc), label))
            last_topic = topic_id


def main():
    n_cpus = 1
    print("running with {} cpus".format(n_cpus))
    n_negatives = 10

    pool = mp.Pool(n_cpus)
    jobs = []

    pbar = tqdm(total=len(qrels), desc="submitting")
    input_path = os.path.join(data_home, "triples-tokenized", "cut-{}.top100".format(split))
    total_size = 363499
    chunk_size = 363499
    chunk = []
    chunks = [(i, i + chunk_size) for i in range(0, total_size, chunk_size)]
    
    for index, chunk in enumerate(chunks):
        process_chunk(index, n_negatives, chunk, split, input_path)
        # jobs.append(pool.apply_async(process_chunk, args=(index, n_negatives, chunk, split, input_path), callback=pbar.update()))
    for job in jobs:
        job.get()
    pool.close()


if __name__ == "__main__":
    main()
