import os
import pickle
from collections import Counter, defaultdict
from tqdm.auto import tqdm
from itertools import product
import multiprocessing as mp
import numpy as np
from scipy.spatial.distance import cosine
import wandb
import logging
from compute_IDF_on_whole_corpus import compute_IDFS
from functools import partial
from gensim.models import KeyedVectors
from bert import generate_run_file
import re
logging.getLogger("gensim").setLevel(logging.WARNING)


# mp.set_start_method("spawn", True)


def getcontent(doc_id, docs_file, offset_dict):
    offset = offset_dict[doc_id]
    with open(docs_file) as f:
        f.seek(offset)
        doc_id, doc = f.readline().split("\t")
    return doc.split(" ")


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


def TFC1(chunk_no, chunk, all_docs, tuples, docs_lens, args, scores):
    instances = []
    for sample in tqdm(chunk, desc="processor {}".format(chunk_no), position=chunk_no):
        topic_id, query_terms = sample
        query_terms = set(query_terms)
        docs_skipped = set()
        # return
        for di_id in tuples[topic_id]:
            if di_id in docs_skipped:
                continue
            di_text = [w for w in all_docs[di_id] if w in query_terms]
            if len(di_text) == 0:
                continue
            di_terms_counter = Counter(di_text)  # document TF
            sum_occurences_di = sum(di_terms_counter.values())  # Sum of TFs
            for dj_id in tuples[topic_id]:
                if di_id == dj_id or dj_id in docs_skipped:
                    continue
                if abs(docs_lens[di_id] - docs_lens[dj_id]) > args["delta"]:
                    continue
                dj_text = [w for w in all_docs[dj_id] if w in query_terms]
                if len(dj_text) == 0:
                    docs_skipped.add(dj_id)
                    continue
                dj_terms_counter = Counter(dj_text)  # document TF
                # Axiom premisses
                if sum_occurences_di <= sum(dj_terms_counter.values()):
                    continue
                if sum([di_terms_counter[w] < dj_terms_counter[w] for w in query_terms]) > 0:
                    continue
                instances.append((topic_id, di_id, dj_id))
    return instances


def TFC2(chunk_no, chunk, all_docs, tuples, docs_lens, args, scores):
    instances = []
    for sample in tqdm(chunk, desc="processor {}".format(chunk_no), position=chunk_no):
        topic_id, query_terms = sample
        if topic_id is None:
            continue
        query_terms = set(query_terms)
        fullfils = 0
        for di_id in tuples[topic_id]:
            di_text = [w for w in all_docs[di_id] if w in query_terms]
            if len(di_text) == 0:
                continue
            di_terms_counter = Counter(di_text)
            sum_di_terms = sum(di_terms_counter.values())
            for dj_id in tuples[topic_id]:
                if dj_id == di_id:
                    continue
                if abs(docs_lens[di_id] - docs_lens[dj_id]) > args["delta"]:
                    continue
                dj_text = [w for w in all_docs[dj_id] if w in query_terms]
                if len(dj_text) == 0:
                    continue
                dj_terms_counter = Counter(dj_text)
                sum_dj_terms = sum(dj_terms_counter.values())
                for dk_id in tuples[topic_id]:
                    if dk_id == di_id or dk_id == dj_id:
                        continue
                    if (abs(docs_lens[dk_id] - docs_lens[dj_id]) > args["delta"]
                       or abs(docs_lens[dk_id] - docs_lens[di_id]) > args["delta"]):  # noqa: W503
                        continue
                    dk_text = [w for w in all_docs[dk_id] if w in query_terms]
                    if len(dk_text) == 0:
                        continue
                    dk_terms_counter = Counter(dk_text)
                    sum_dk_terms = sum(dk_terms_counter.values())
                    if not (sum_dk_terms > sum_dj_terms and sum_dj_terms > sum_di_terms and sum_di_terms > 0):
                        continue
                    flag = False
                    for w in query_terms:
                        if (dj_terms_counter[w] - di_terms_counter[w]) != (dk_terms_counter[w] - dj_terms_counter[w]):
                            flag = True
                            break
                    if flag:
                        continue
                    instances.append((topic_id, di_id, dj_id, dk_id))
                    di_score = scores["{}-{}".format(topic_id, di_id)]
                    dj_score = scores["{}-{}".format(topic_id, dj_id)]
                    dk_score = scores["{}-{}".format(topic_id, dk_id)]
                    if dj_score - di_score > dk_score - dj_score:
                        fullfils += 1
    return instances


def MTDC(chunk_no, chunk, all_docs, tuples, docs_lens, args, scores):
    instances = []
    IDFS = pickle.load(open(args["IDF_file"], 'rb'))
    for sample in tqdm(chunk, desc="processor {}".format(chunk_no), position=chunk_no):
        topic_id, query = sample
        if topic_id is None:
            continue
        query_terms = set(query)
        query_terms_counter = Counter(query)
        for di_id in tuples[topic_id]:
            di_text = [w for w in all_docs[di_id] if w in query_terms]
            if len(di_text) == 0:
                continue
            di_terms_counter = Counter(di_text)
            di_terms_sum = sum(di_terms_counter.values())
            for dj_id in tuples[topic_id]:
                # same lenght
                if di_id == dj_id or abs(docs_lens[di_id] - docs_lens[dj_id]) > args["delta"]:
                    continue
                dj_text = [w for w in all_docs[dj_id] if w in query_terms]
                dj_terms_counter = Counter(dj_text)
                dj_terms_sum = sum(dj_terms_counter.values())

                # same sum of query terms, but not equal counts
                if di_terms_sum != dj_terms_sum or di_terms_counter == dj_terms_counter:
                    continue

                valid_query_terms = [w for w in query_terms if di_terms_counter[w] != dj_terms_counter[w]]
                # for each query term pair, it must be valid. It's valid if:
                # 1. idf(wa) >= idf(wb)
                # 2. c(wa, d1) = c(wb, d2) and c(wb, d1) = c(wa, d2)
                # 3 c(wa,d1) > c(wa, d2)
                # 4 c(wa, q) >= c(wb, q)
                query_terms_used = set()
                for wa, wb in product(valid_query_terms, valid_query_terms):
                    if wa == wb:
                        continue
                    # smoothing
                    if IDFS[wa] == 0:
                        IDFS[wa] = 1
                    if IDFS[wb] == 0:
                        IDFS[wb] = 1
                    if ((1 / IDFS[wa]) < (1 / IDFS[wb])):  # 1
                        continue
                    if not(di_terms_counter[wa] == dj_terms_counter[wb]
                            and di_terms_counter[wb] == dj_terms_counter[wa]):
                        continue
                    if di_terms_counter[wa] <= dj_terms_counter[wa]:  # 3
                        continue
                    if query_terms_counter[wa] < query_terms_counter[wb]:  # 4
                        continue
                    query_terms_used.add(wa)
                    query_terms_used.add(wb)
                if query_terms_used == set(valid_query_terms):
                    instances.append((topic_id, di_id, dj_id))
    return instances


def LNC1(chunk_no, chunk, all_docs, tuples, docs_lens, args, scores):
    instances = []
    for sample in tqdm(chunk, desc="processor {}".format(chunk_no), position=chunk_no):
        topic_id, query = sample
        if topic_id is None:
            continue
        query_terms = set(query)
        for di_id in tqdm(tuples[topic_id]):
            di_text = all_docs[di_id]
            di_terms_counter = Counter(di_text)
            w_prime_di = set(di_text).difference(query_terms)
            for dj_id in tuples[topic_id]:
                if di_id == dj_id:
                    continue
                dj_text = all_docs[dj_id]
                dj_terms_counter = Counter(dj_text)
                # for any query term w, c(w, d2) = c(w, d1)
                equality = [di_terms_counter[w] == dj_terms_counter[w] for w in query_terms]
                if sum(equality) < 0:
                    continue
                # If for some word w′ ∈/ q, c(w′, d2) = c(w′, d1) + 1
                flag = False
                for w_prime in w_prime_di:
                    if dj_terms_counter[w_prime] == di_terms_counter[w_prime] + 1:
                        flag = True
                        break
                if not flag:
                    continue
                instances.append((topic_id, di_id, dj_id))
        return instances


def LNC2(chunk_no, chunk, all_docs, tuples, docs_lens, args, scores):
    """Prepares param file for c++ code and prepares test file for bert model
    THIS SHOULD ONLY BE RUN IN A SINGLE PROCESS!
    """

    # first, new indri param
    queries_path = os.path.join(args["data_home"], "queries/test.tokenized.tsv")
    new_param_file = os.path.join(args["data_home"], "indri_params/LNC2.param")
    param_format = "<query>\n<number>{}</number>\n<text>{}</text>\n</query>\n"
    run_path = os.path.join(args["data_home"], "runs/QL_test-cut.run")
    pattern = re.compile('([^\s\w]|_)+')  # noqa W605
    with open(new_param_file, 'w') as outf:
        index_path = os.path.join(args["data_home"], "indexes/cut-tokenized")
        outf.write(f"<parameters>\n<rankedDocsFile>{run_path}</rankedDocsFile>\n<index>{index_path}</index>\n")
        for line in open(queries_path):
            q_id, query = line.strip().split("\t")
            query = pattern.sub('', query)
            outf.write(param_format.format(q_id, query))
        outf.write("</parameters>\n")

    # Generate a list of documents that were ranked that should be copied. copy and generate a file for BERT input.
    # Load queries
    queries_file = os.path.join(args["data_home"], "queries/test.tokenized.bert")
    queries = {}
    for line in open(queries_file):
        topic_id, query = line.strip().split("\t")
        query = eval(query)
        queries[topic_id] = query
    # Load docs, bert version.
    docs_file = os.path.join(args["data_home"], "docs/msmarco-docs.tokenized.bert")
    offset_dict = pickle.load(open(docs_file + ".offset", 'rb'))
    instances = []
    new_triples_file = os.path.join(args["data_home"], "triples/LNC-cut-triples.tsv")
    if not os.path.isfile(new_triples_file):
        with open(new_triples_file, 'w') as outf:
            for line in tqdm(open(run_path), total=100 * args["test_set_size"]):
                topic_id, _, doc_id, _, score, _ = line.split()
                # Check doc length
                if docs_lens[doc_id] > 256:
                    continue
                doc = getcontent(doc_id, docs_file, offset_dict)
                tokens = ["[CLS]"] + queries[topic_id] + ["[SEP]"] + doc[:512] + ["[SEP]"]
                # Dummy relevance label.
                outf.write("{}-{}\t{}\t1\n".format(topic_id, doc_id, tokens))
                instances.append((topic_id, doc_id, doc_id + "-LNC2"))
                k = 512 // len(doc)
                doc = doc * k
                # Create triple with proper tokens
                query = queries[topic_id]
                truncate_seq_pair(query, doc)
                tokens = ["[CLS]"] + query + ["[SEP]"] + doc + ["[SEP]"]
                outf.write("{}-{}-LNC2\t{}\t0\n".format(topic_id, doc_id, tokens))
        generate_run_file("test", "cut", new_triples_file)
    else:
        for line in tqdm(open(run_path), total=100 * args["test_set_size"]):
            topic_id, _, doc_id, _, score, _ = line.split()
            # Check doc length
            if docs_lens[doc_id] > 256:
                continue
            instances.append((topic_id, doc_id, doc_id + "-LNC2"))
    # Generate run file for this
    return instances


def TPC(chunk_no, chunk, all_docs, tuples, docs_lens, args, scores):
    instances = []
    for sample in tqdm(chunk, desc="processor {}".format(chunk_no), position=chunk_no):
        topic_id, query = sample
        if topic_id is None:
            continue
        query_terms = set(query)
        query_pairs = product(query_terms, query_terms)
        query_pairs = [x for x in query_pairs if x[0] != x[1]]
        for di_id in tuples[topic_id]:
            min_indexes_di = []
            di_text = all_docs[di_id]
            di_terms = set(di_text)
            for q1, q2 in query_pairs:
                if q1 not in di_terms or q2 not in di_terms:
                    continue
                all_indexes_q1 = [i for i, x in enumerate(di_text) if x == q1]
                all_indexes_q2 = [i for i, x in enumerate(di_text) if x == q2]
                index_pairs = list(product(all_indexes_q1, all_indexes_q2))
                min_indexes_di.append(min(map(lambda x: abs(x[0] - x[1]), index_pairs)))
            if len(min_indexes_di) == 0:
                continue
            min_index_di = min(min_indexes_di)
            for dj_id in tuples[topic_id]:
                if di_id == dj_id:
                    continue
                min_indexes_dj = []
                dj_text = all_docs[dj_id]
                dj_terms = set(dj_text)
                for q1, q2 in query_pairs:
                    if q1 not in dj_terms or q2 not in dj_terms:
                        continue
                    all_indexes_q1 = [i for i, x in enumerate(dj_text) if x == q1]
                    all_indexes_q2 = [i for i, x in enumerate(dj_text) if x == q2]
                    index_pairs = list(product(all_indexes_q1, all_indexes_q2))
                    min_indexes_dj.append(min(map(lambda x: abs(x[0] - x[1]), index_pairs)))
                if len(min_indexes_dj) == 0:
                    continue
                min_index_dj = min(min_indexes_dj)
                if min_index_dj < min_index_di:
                    instances.append((topic_id, di_id, dj_id))
    return instances


def STMC1(chunk_no, chunk, all_docs, tuples, docs_lens, args, scores):
    '''
    Let Q = {q} be a query with only one term q. -> Relaxation - Multiple terms
    Let D1 = {d1} and D2 = {d2} be two single-term documents, where q != d1 and q != d2.
        - > Relaxation: At least one query term is present in di but not in dj
    If s(q, d1) > s(q, d2), then -> if average term embedding differ
    S(Q, D1) > S(Q, D2).

    di and dj has the same number of query terms. Remove all query terms. Check similarity with query.
        should correspond to score.
    '''

    instances = []
    vectors = args["vectors"]
    for sample in tqdm(chunk, desc="processor {}".format(chunk_no), position=chunk_no):
        topic_id, query = sample
        if topic_id is None:
            continue
        query_terms = set(query)
        _docs = list(tuples[topic_id])
        query_avg = np.mean(vectors[[x for x in query if x in vectors.vocab]], axis=0)
        for i, di_id in enumerate(_docs):
            di_text = all_docs[di_id]
            # remove query terms
            di_clean = [w for w in di_text if w not in query_terms]
            di_query_terms = [w for w in di_text if w in query_terms]
            di_avg = np.mean(vectors[[x for x in di_clean if x in vectors.vocab]], axis=0)
            di_similarity = cosine(di_avg, query_avg)
            for dj_id in _docs[i:]:
                if di_id == dj_id:
                    continue
                dj_text = all_docs[dj_id]
                dj_query_terms = [w for w in dj_text if w in query_terms]
                if len(di_query_terms) != len(dj_query_terms):
                    continue
                dj_clean_text = [w for w in dj_text if w not in query_terms]
                dj_avg = np.mean(vectors[[x for x in dj_clean_text if x in vectors.vocab]], axis=0)
                # di_id and dj_id must have same number of query terms
                # at least one query term occur in di but not in dj
                dj_similarity = cosine(dj_avg, query_avg)
                if di_similarity > dj_similarity:
                    instances.append((topic_id, di_id, dj_id))
                elif dj_similarity > di_similarity:
                    instances.append((topic_id, dj_id, di_id))
    return instances


def STMC2(chunk_no, chunk, all_docs, tuples, docs_lens, args, scores):
    '''
    Let Q = {q} be a single term query -> Multiple terms
    and d be a non-query term such that s(q, d) > 0.
    If D1 and D2 are  two documents such that |D1| = 1, c(q, D1) = 1, -> At least one term from the query is present
        in the document
    |D2| = k and c(d, D2) = k (k ≥ 1), where c(q, D1) and c(d, D2) are the counts of q and d in D1 and D2 respectively,
        then S(Q, D1) ≥ S(Q, D2).

    d1 w/out query terms approx.equal to d2 w/out query terms
    avg(d1_terms) > avg(d2_terms).
    S(d1) > S(d2)
    '''

    instances = []
    vectors = args["vectors"]
    for sample in tqdm(chunk, desc="processor {}".format(chunk_no), position=chunk_no):
        topic_id, query = sample
        if topic_id is None:
            continue
        query_terms = set(query)
        _docs = list(tuples[topic_id])
        for i, di_id in enumerate(_docs):
            # remove query terms
            di_clean = [w for w in all_docs[di_id] if w not in query_terms]
            di_query_terms = [w for w in all_docs[di_id] if w in query_terms]
            di_avg = np.mean(vectors[[x for x in di_clean if x in vectors.vocab]], axis=0)
            for dj_id in _docs[i:]:
                if di_id == dj_id:
                    continue
                dj_clean = [w for w in all_docs[dj_id] if w not in query_terms]
                # di and dj must be similar
                dj_avg = np.mean(vectors[[x for x in dj_clean if x in vectors.vocab]], axis=0)
                di_dj_distance = cosine(di_avg, dj_avg)
                if di_dj_distance > args["stmc_sim"]:
                    continue
                dj_query_terms = [w for w in all_docs[dj_id] if w in query_terms]
                # di must have more query terms than dj
                if len(dj_query_terms) > len(di_query_terms):
                    instances.append((topic_id, dj_id, di_id))
                elif len(dj_query_terms) < len(di_query_terms):
                    instances.append((topic_id, di_id, dj_id))
    return instances


def STMC3(chunk_no, chunk, all_docs, tuples, docs_lens, args, scores):
    '''
    conditions: D1 and D2 covers the same number of query terms
    D1 and D2 are approx. the same size
    D1 has more query terms than D2.
    D2 without query terms is more similar to q than D1.
    '''
    instances = []
    vectors = args["vectors"]
    for sample in tqdm(chunk, desc="processor {}".format(chunk_no), position=chunk_no):
        topic_id, query = sample
        if topic_id is None:
            continue
        query_terms = set(query)
        query_avg = np.mean(vectors[[x for x in query if x in vectors.vocab]], axis=0)
        _docs = list(tuples[topic_id])
        for i, di_id in enumerate(_docs):
            di_clean = [w for w in all_docs[di_id] if w not in query_terms]  # Di without query terms
            di_query_terms = [w for w in all_docs[di_id] if w in query_terms]
            di_query_set = set(di_query_terms)
            di_avg = np.mean(vectors[[x for x in di_clean if x in vectors.vocab]], axis=0)
            di_query_sim = cosine(di_avg, query_avg)
            for dj_id in _docs[i:]:
                # similarity
                if di_id == dj_id or abs(docs_lens[di_id] - docs_lens[dj_id]) >= args["delta"]:
                    continue
                dj_query_terms = [w for w in all_docs[dj_id] if w in query_terms]
                dj_query_set = set(dj_query_terms)
                # both must cover same number of terms
                if len(dj_query_set) != len(di_query_set):
                    continue
                dj_clean = [w for w in all_docs[dj_id] if w not in query_terms]
                dj_avg = np.mean(vectors[[x for x in dj_clean if x in vectors.vocab]], axis=0)
                dj_query_sim = cosine(dj_avg, query_avg)
                if len(di_clean) > len(dj_clean) and dj_query_sim > di_query_sim:
                    instances.append((topic_id, di_id, dj_id))
                elif len(dj_clean) > len(di_clean) and di_query_sim > dj_query_sim:
                    instances.append((topic_id, dj_id, di_id))
    return instances


def extract_datasets(cut):
    config = wandb.config
    axioms = config.axioms
    if cut == 'cut':
        docs_path = os.path.join(config.data_home, "docs/msmarco-docs.tokenized.cut.tsv")
    else:
        docs_path = os.path.join(config.data_home, "docs/msmarco-docs.tokenized.tsv")
    offset_path = docs_path + ".offset"
    if os.path.isfile(offset_path):
        offset_dict = pickle.load(open(offset_path, 'rb'))
    else:
        offset_dict = {}
        pbar = tqdm(total=config.corpus_size + 1, desc="loading offset dict for file {}".format(docs_path))
        with open(docs_path, 'r', encoding="utf-8") as f:
            location = f.tell()
            line = f.readline().encode("utf-8")
            pbar.update()
            while(line):
                if len(line) < 2:
                    line = f.readline().encode("utf-8")
                    pbar.update()
                    location = f.tell()
                    continue
                did = line.decode().split("\t")[0]
                offset_dict[did] = location
                location = f.tell()
                line = f.readline().encode("utf-8")
                pbar.update()
        pbar.close()
        pickle.dump(offset_dict, open(offset_path, 'wb'))

    top_100_path = os.path.join(config.data_home, "runs/QL_test-{}.run".format(cut))
    assert os.path.isfile(top_100_path), "could not find run file at %s" % top_100_path
    assert os.path.isfile(docs_path), "could not find docs file at %s" % docs_path

    tuples = defaultdict(lambda: set())
    all_docs = dict()
    docs_lens = dict()
    size = 100 * config.test_set_size
    all_docs_p = os.path.join(config.data_home, "tmp", "all_docs.pkl")
    docs_lens_p = os.path.join(config.data_home, "tmp", "docs_lens.pkl")
    tuples_p = os.path.join(config.data_home, "tmp", "tuples.pkl")
    if os.path.isfile(all_docs_p) and os.path.isfile(docs_lens_p) and os.path.isfile(tuples_p):
        all_docs = pickle.load(open(all_docs_p, 'rb'))
        docs_lens = pickle.load(open(docs_lens_p, 'rb'))
        tuples = pickle.load(open(tuples_p, 'rb'))
    else:
        for i in tqdm(open(top_100_path), total=size, desc="loading unique docs"):
            topic_id, _, doc_id, _, score, _ = i.split()
            tuples[topic_id].add(doc_id)
            if doc_id in all_docs:
                continue
            all_docs[doc_id] = getcontent(doc_id, docs_path, offset_dict)
            docs_lens[doc_id] = len(all_docs[doc_id])
        tuples = dict(tuples)
        pickle.dump(all_docs, open(all_docs_p, 'wb'))
        pickle.dump(docs_lens, open(docs_lens_p, 'wb'))
        pickle.dump(tuples, open(tuples_p, 'wb'))
    queries_file = os.path.join(config.data_home, "queries/test.tokenized.tsv")
    assert os.path.isfile(queries_file), "could not find queries file at %s" % queries_file
    diagnostics_path = os.path.join(config.data_home, "diagnostics")
    if not os.path.isdir(diagnostics_path):
        os.mkdir(diagnostics_path)
    all_lines = [(x.split("\t")[0], x.split("\t")[1].strip().split(" ")) for x in open(queries_file).readlines()]
    scores = {}
    for line in open(top_100_path):
        topic_id, _, doc_id, _, score, _ = line.split(" ")
        scores["{}-{}".format(topic_id, doc_id)] = float(score)
    for axiom in axioms:
        cpus = config.number_of_cpus
        vectors = None
        if axiom == "MTDC":  # We need IDFs!
            IDF_folder = os.path.join(config.data_home, "docs/IDFS/IDFS-FULL-{}".format(cut))
            config.IDF_file = IDF_folder
            if not os.path.isfile(IDF_folder):
                # Generate IDFs
                compute_IDFS(IDF_folder.replace("/IDFS-FULL", ""), cut)
        elif "STMC" in axiom:  # We need the word embeddings!
            embeddings_path = os.path.join(config.data_home, "GloVe/w2v.txt")
            assert os.path.isfile(embeddings_path), ("Embeddings not found at %s. They need to be manually computed!"
                                                     % embeddings_path)
            if vectors is None:
                logging.info("Loading embeddings from %s" % embeddings_path)
                vectors = KeyedVectors.load_word2vec_format(embeddings_path)
                vectors.init_sims(replace=True)
                # After generating the vectors using GLoVe, they also need to be transformed into W2V format.
                # Check https://radimrehurek.com/gensim/scripts/glove2word2vec.html on how to do so.
        elif axiom == "LNC2":
            cpus = 0
        logging.info("Running axiom %s with %i cpus and %i lines" % (axiom, max(1, cpus), len(all_lines)))
        args = dict(config)
        args["vectors"] = vectors
        if cpus > 1:
            if len(all_lines) % config.number_of_cpus != 0:
                logging.info("padding topics with %i empty topics" % (len(all_lines) % config.number_of_cpus))
            while len(all_lines) % config.number_of_cpus != 0:
                all_lines.append((None, None))
            chunk_size = len(all_lines) // config.number_of_cpus
            logging.info("Each process will read %i topics" % chunk_size)
            chunks = [all_lines[i * chunk_size: (i + 1) * chunk_size] for i in range(config.number_of_cpus)]
            all_docs_per_chunk = []
            for chunk in chunks:
                chunk_dict = {}
                for topic, _ in chunk:
                    if topic is None:
                        continue
                    for doc in tuples[topic]:
                        chunk_dict[doc] = all_docs[doc]
                all_docs_per_chunk.append(chunk_dict)
            f = partial(globals()[axiom],
                        tuples=tuples,
                        docs_lens=docs_lens,
                        args=args,
                        scores=scores)
            all_ids = list(range(config.number_of_cpus))
            with mp.Pool(config.number_of_cpus) as pool:
                instances = list(pool.starmap(f, zip(all_ids, chunks, all_docs_per_chunk)))
            instances_flat = [item for sublist in instances for item in sublist]
            pickle.dump(instances_flat, open(os.path.join(diagnostics_path, "{}-instances".format(axiom)), 'wb'))
            logging.info("Created dataset for axiom %s with %i instances" % (axiom, len(instances_flat)))
        else:
            instances = []
            chunk = all_lines
            f = globals()[axiom]
            instances = f(0, chunk, all_docs, tuples, docs_lens, args, scores)
            pickle.dump(instances, open(os.path.join(diagnostics_path, "{}-instances".format(axiom)), 'wb'))
            logging.info("Created dataset for axiom %s with %i instances" % (axiom, len(instances)))
