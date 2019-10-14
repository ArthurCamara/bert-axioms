import multiprocessing as mp
from tqdm.auto import tqdm
import pickle
from pytorch_transformers import BertTokenizer
import os
mp.set_start_method('spawn', True)
data_home = "/ssd2/arthur/TREC2019/data/"


def truncate_seq_pair(tokens_a, tokens_b, max_length=509):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def getcontent(doc_id, docs_file):
    offset = offset_dict[doc_id]
    with open(docs_file) as f:
        f.seek(offset)
        doc = f.readline()
    return doc


def process_chunk(chunk_no, n_itens, max_lines, split):
    starting_index = chunk_no * n_itens
    end_index = min(((chunk_no + 1) * n_itens), max_lines)
    outfile = os.path.join(data_home, "triples-tokenized/full-{}.top100.{}".format(split, chunk_no))
    with open(outfile, 'w') as outf:
        for topic_id, doc_id in tqdm(all_docs[starting_index:end_index], position=chunk_no + 1):
            _id = "{}-{}".format(topic_id, doc_id)
            doc_content = tokenizer.tokenize(getcontent(doc_id, tokenized_docs))
            label = "1" if _id in qrels else "0"
            if "LNC" in _id:
                label = "1" if _id.replace("LNC2", "") in qrels else "0"
            query = queries[topic_id]
            truncate_seq_pair(query, doc_content)
            tokens = ["[CLS]"] + query + ["[SEP]"] + doc_content + ["[SEP]"]
            outf.write("{}\t{}\t{}\n".format(_id, tokens, label))


offset_dict = pickle.load(open(os.path.join(data_home, "docs/tokenized-msmarco-docs.tsv.offset"), 'rb'))
tokenized_docs = os.path.join(data_home, "docs/tokenized-msmarco-docs.tsv")
split = "train"

tokenized_queries = {}
run_file = os.path.join(data_home, "runs/indri_train_10_10.run")
qrels_file = os.path.join(data_home, "qrels/{}_qrels".format(split))
qrels = set()
for line in tqdm(open(qrels_file), desc="qrels loading"):
    query_id, _, doc_id, label = line.strip().split()
    tuple_id = "{}-{}".format(query_id, doc_id)
    if label == "1":
        qrels.add(tuple_id)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

queries = {}
for line in tqdm(open(os.path.join(data_home, "queries/{}_queries.tsv".format(split))), desc="loading queries"):
    query_id, query_text = line.strip().split("\t")
    queries[query_id] = tokenizer.tokenize(query_text)

all_docs = []
for line in tqdm(open(run_file), total=11 * len(queries), desc="loading run file"):
    topic_id, _, doc_id, _, _, _ = line.split()
    all_docs.append((topic_id, doc_id))

n_cpus = 12
excess_lines = len(all_docs) % n_cpus
if excess_lines > 0:
    number_of_chunks = n_cpus - 1

else:
    number_of_chunks = n_cpus
chunk_size = len(all_docs) // number_of_chunks

print(excess_lines)


def main():

    pool = mp.Pool(n_cpus)
    jobs = []

    pbar = tqdm(total=number_of_chunks)
    for i in range(n_cpus):
        # process_chunk(i, chunk_size, len(all_docs), split)
        jobs.append(pool.apply_async(process_chunk, args=(i, chunk_size, len(all_docs), split), callback=pbar.update))
    pbar.close()
    for job in jobs:
        job.get()
    pool.close()
    # Join end results
    with open(os.path.join(data_home, "triples-tokenized/full-{}.top100".format(split)), 'w') as outf:
        for i in range(n_cpus):
            with open(os.path.join(data_home, "triples-tokenized/full-{}.top100.{}".format(split, i))) as inf:
                for line in inf:
                    outf.write(line)


if __name__ == "__main__":
    main()
