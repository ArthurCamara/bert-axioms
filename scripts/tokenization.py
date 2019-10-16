from transformers import DistilBertTokenizer
import logging
from tqdm.auto import tqdm
import os
import multiprocessing as mp
from multiprocessing import current_process
import pickle
# mp.set_start_method('spawn', True)


def tokenize_queries(config):
    # Tokenize queries from a tsv file
    tokenizer = DistilBertTokenizer.from_pretrained(config.tokenizer_class)
    train_queries_path = os.path.join(config.data_home, "queries/msmarco-doctrain-queries.tsv")
    assert os.path.isfile(train_queries_path), "Train queries not found at {}".format(train_queries_path)
    if (not os.path.isfile(train_queries_path + ".tokenized")) or "train_query_tokenizer" in config.force_steps:
        logging.info("tokenizing train queries")
        with open(train_queries_path) as inf, open(train_queries_path + ".tokenized", 'w') as outf:
            for line in tqdm(inf, total=config.train_queries, desc="tokenizing train queries"):
                q_id, query = line.split("\t")
                tokenized_query = ' '.join([x for x in tokenizer.tokenize(query)]).replace("##", "")
                outf.write("{}\t{}".format(q_id, tokenized_query))
    else:
        logging.info("Already found tokenized train queries")

    # Tokenize dev queries
    dev_queries_path = os.path.join(config.data_home, "queries/msmarco-docdev-queries.tsv")
    assert os.path.isfile(dev_queries_path), "Dev queries not found at {}".format(dev_queries_path)
    if (not os.path.isfile(dev_queries_path + ".tokenized")) or "dev_query_tokenizer" in config.force_steps:
        logging.info("tokenizing dev queries")
        with open(dev_queries_path) as inf, open(dev_queries_path + ".tokenized", 'w') as outf:
            for line in tqdm(inf, total=config.full_dev_queries, desc="Tokenizing dev queries"):
                q_id, query = line.split("\t")
                tokenized_query = ' '.join([x for x in tokenizer.tokenize(query)]).replace("##", "")
                outf.write("{}\t{}".format(q_id, tokenized_query))
    else:
        logging.info("Already found tokenized dev queries")


def process_chunk(chunk_no, block_offset, no_lines, config):
    current = current_process()
    if current.name == "MainProcess":
        position = 2
    else:
        position = current._identity[0] + 2
    # Load lines
    lines = []
    docs_path = os.path.join(config.data_home, "docs/msmarco-docs.tsv")
    with open(docs_path, encoding="utf-8") as f:
        f.seek(block_offset[chunk_no])
        for i in tqdm(range(no_lines), desc="Loading block for {}".format(chunk_no), position=position):
            lines.append(f.readline())
    tokenizer = DistilBertTokenizer.from_pretrained(config.tokenizer_class)
    output_line_format = "{}\t{}\n"
    trec_format = "<DOC>\n<DOCNO>{}</DOCNO>\n<TEXT>{}</TEXT></DOC>\n"
    partial_doc_path = os.path.join(config.data_home, "tmp", "docs-{}".format(chunk_no))
    partial_trec_path = os.path.join(config.data_home, "tmp", "trec_docs-{}".format(chunk_no))
    with open(partial_doc_path, 'w', encoding="utf-8") as outf, open(partial_trec_path, 'w', encoding="utf-8") as outf_trec:
        for line in tqdm(lines, desc="Running for chunk {}".format(chunk_no), position=position):
            try:
                doc_id, url, title, text = line[:-1].split("\t")
            except:
                print(line)
                continue
            full_text = " ".join([url, title, text])
            tokenized_text = ' '.join([x for x in tokenizer.tokenize(full_text)]).replace("##", "")
            outf.write(output_line_format.format(doc_id, tokenized_text))
            outf_trec.write(trec_format.format(doc_id, tokenized_text))


def tokenize_docs(config):
    """Tokenize docs, both tsv and TREC formats. Also generates offset file. Can take a LONG time"""
    if (os.path.isfile(os.path.join(config.data_home, "docs/msmarco-docs.tokenized.tsv"))
                                and "doc_tokenizer" not in config.force_steps):  # noqa
        logging.info("tokenized tsv file found. skipping all document tokenization process.")
        return

    docs_path = os.path.join(config.data_home, "docs/msmarco-docs.tsv")
    assert os.path.isfile(docs_path), "Could not find documents file at {}".format(docs_path)
    # Load in memory, split blocks and run in paralel. Later
    excess_lines = config.corpus_size % config.number_of_cpus
    number_of_chunks = config.number_of_cpus
    if excess_lines > 0:
        number_of_chunks = config.number_of_cpus - 1
    block_offset = {}
    lines_per_chunk = config.corpus_size // number_of_chunks
    logging.info("Number of lines per CPU chunk: %i", lines_per_chunk)
    if not os.path.isdir(os.path.join(config.data_home, "tmp")):
        os.mkdir(os.path.join(config.data_home, "tmp"))
    if config.number_of_cpus < 2:
        block_offset[0] = 0
    elif not os.path.isfile(os.path.join(config.data_home, "block_offset_{}.pkl".format(config.number_of_cpus))):
        pbar = tqdm(total=config.corpus_size)
        with open(docs_path) as inf:
            current_chunk = 0
            counter = 0
            line = True
            while line:
                pbar.update()
                if counter % lines_per_chunk == 0:
                    block_offset[current_chunk] = inf.tell()
                    current_chunk += 1
                line = inf.readline()
                counter += 1
        pbar.close()
        pickle.dump(block_offset, open(os.path.join(config.data_home, "block_offset_{}.pkl".format(config.number_of_cpus)), 'wb'))  # noqa E501
    else:
        block_offset = pickle.load(open(os.path.join(config.data_home, "block_offset_{}.pkl".format(config.number_of_cpus)), 'rb'))  # noqa E501
    pbar = tqdm(total=config.number_of_cpus)
    assert len(block_offset) == config.number_of_cpus

    def update(*a):
        pbar.update()
    if config.number_of_cpus == 1:
        process_chunk(0, block_offset, lines_per_chunk, config)
        return
    pool = mp.Pool(config.number_of_cpus)
    jobs = []
    for i in range(len(block_offset)):
        jobs.append(pool.apply_async(process_chunk, args=(i, block_offset, lines_per_chunk, config), callback=update))
    for job in jobs:
        job.get()
    pool.close()

    with open(os.path.join(config.data_home, "docs/msmarco-docs.tokenized.tsv"), 'w') as outf:
        for i in tqdm(range(config.number_of_cpus), desc="Merging tsv file"):
            partial_path = os.path.join(config.data_home, "tmp", "docs-{}".format(i))
            for line in open(partial_path):
                outf.write(line)
            os.remove(partial_path)

    with open(os.path.join(config.data_home, "docs/msmarco-docs.tokenized.trec"), 'w') as outf:
        for i in tqdm(range(config.number_of_cpus), desc="Merging TREC file"):
            partial_trec_path = os.path.join(config.data_home, "tmp", "trec_docs-{}".format(i))
            for line in open(partial_trec_path):
                outf.write(line)
            os.remove(partial_trec_path)
