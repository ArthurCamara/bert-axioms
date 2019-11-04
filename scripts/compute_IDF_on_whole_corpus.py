import multiprocessing as mp
import os
import pickle
from tqdm.auto import tqdm
from collections import Counter
import wandb
import logging
mp.set_start_method('spawn', True)


def process_chunk(chunk_no, block_offset, inf, no_lines, output_folder):
    DFS = Counter()
    position = chunk_no + 1

    pbar = tqdm(total=no_lines, desc="Running for chunk {}".format(
        str(chunk_no).zfill(2)), position=position)
    with open(inf, 'r') as f:
        f.seek(block_offset[chunk_no])
        for i in range(no_lines):
            try:
                line = f.readline()
                tokens = line.split("\t")[1].split()
            except IndexError:
                continue
            for w in tokens:
                DFS[w] += 1
            pbar.update()
    pbar.close()
    pickle.dump(DFS, open(os.path.join(
        output_folder, "IDFS-{}".format(chunk_no)), 'wb'))


def compute_IDFS(output_folder, cut):
    config = wandb.config
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    cpus = max(1, config.number_of_cpus)
    logging.info("Computing IDF with %i cpus" % cpus)
    excess_lines = config.corpus_size % cpus
    number_of_chunks = cpus
    if excess_lines > 0:
        number_of_chunks = cpus - 1
        excess_lines = config.corpus_size % number_of_chunks
    lines_per_chunk = config.corpus_size // number_of_chunks
    logging.info("{}  lines per chunk".format(lines_per_chunk))
    logging.info("{}  lines for last chunk".format(excess_lines))
    assert (number_of_chunks * lines_per_chunk + excess_lines) == config.corpus_size

    if cut == 'cut':
        docs_path = os.path.join(config.data_home, "docs/msmarco-docs.tokenized.cut.tsv")
    else:
        docs_path = os.path.join(config.data_home, "docs/msmarco-docs.tokenized.tsv")

    block_offset = dict()
    if cpus < 2:
        block_offset[0] = 0
    else:  # Compute offset for documents for each chunk to be processed
        output_file = os.path.join(output_folder, "blocks_offset_{}-cpus".format(cpus))
        if not os.path.isfile(output_file):
            pbar = tqdm(total=config.corpus_size + 1, desc="Computing chunks for each processor")
            with open(docs_path) as f:
                current_chunk = 0
                counter = 0
                line = True
                while(line):
                    if counter % lines_per_chunk == 0:
                        block_offset[current_chunk] = f.tell()
                        current_chunk += 1
                    line = f.readline()
                    pbar.update()
                    counter += 1
            pbar.close()
            pickle.dump(block_offset, open(output_file, 'wb'))
        else:
            block_offset = pickle.load(open(output_file, 'rb'))

    if cpus < 2:  # Single CPU, compute directly.
        process_chunk(0, block_offset, docs_path, lines_per_chunk, output_folder)
    else:
        pbar = tqdm(total=cpus, position=0)

        def update(*a):  # Update progress bar
            pbar.update()
        pool = mp.Pool(cpus)
        jobs = []
        for i in range(len(block_offset)):
            jobs.append(pool.apply_async(process_chunk, args=(
                i, block_offset, docs_path, lines_per_chunk, output_folder), callback=update))
        for job in jobs:
            job.get()
        pool.close()
        pbar.close()
    full_IDFS = Counter()
    for i in range(len(block_offset)):
        _idf = pickle.load(open(os.path.join(output_folder, "IDFS-{}".format(i)), 'rb'))
        for k in _idf:
            full_IDFS[k] += _idf[k]
        os.remove(os.path.join(output_folder, "IDFS-{}".format(i)))
    pickle.dump(full_IDFS, open(os.path.join(output_folder, "IDFS-FULL-{}".format(cut)), 'wb'))
