import argparse
import os
import sys
import pickle
from pytorch_transformers import BertTokenizer
from collections import Counter, defaultdict
from tqdm.auto import tqdm
from itertools import product
from gensim.models import KeyedVectors
import multiprocessing as mp
import numpy as np
from scipy.spatial.distance import cosine
from multiprocessing import Manager, current_process
import wandb
import logging
from itertools import repeat
from compute_IDF_on_whole_corpus import compute_IDFS

# mp.set_start_method("spawn", True)


def getcontent(doc_id, docs_file, offset_dict):
    offset = offset_dict[doc_id]
    with open(docs_file) as f:
        f.seek(offset)
        doc_id, doc = f.readline().split("\t")
    return eval(doc)


def TFC1(sample, all_docs, tuples, docs_lens, args, scores):
    topic_id, query_terms = sample
    query_terms = set(query_terms)
    instances = []
    docs_skipped = set()
    current = current_process()
    try:
        position = current._identity[0] - 1
    except:
        position = None
    # return
    for di_id in tqdm(tuples[topic_id], position=position, total=100):
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
    # output_folder = os.path.join(os.path.join(args["data_home"], "tmp/TFC1_{}".format(topic_id)))
    # pickle.dump(instances, open(output_folder, 'wb'))
    return instances


def TFC2(sample, all_docs, tuples, docs_lens, args, qrels):
    topic_id, query_terms = sample
    query_terms = set(query_terms)
    instances = []
    fullfils = 0
    for di_id in tqdm(tuples[topic_id]):
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
                di_score = qrels["{}-{}".format(topic_id, di_id)]
                dj_score = qrels["{}-{}".format(topic_id, dj_id)]
                dk_score = qrels["{}-{}".format(topic_id, dk_id)]
                if dj_score - di_score > dk_score - dj_score:
                    fullfils += 1
    if len(instances) == 0:
        print("No instances. WTF?")
    if len(instances) != 0:
        print(len(instances), fullfils / len(instances))
    return instances


def MTDC(sample, all_docs, tuples, docs_lens, args, qrels):
    IDFS = pickle.load(open(os.path.join(args.data_home, args.idf_file), 'rb'))
    query = [x.replace("##", "") for x in tokenized_query]
    query_terms = set(query)
    instances = []
    query_terms_counter = Counter(query)
    for di_id in tqdm(tuples[topic_id]):
        di_text = [w for w in all_docs[di_id] if w in query_terms]
        if len(di_text) == 0:
            continue
        di_terms_counter = Counter(di_text)
        di_terms_sum = sum(di_terms_counter.values())
        for dj_id in tuples[topic_id]:
            # same lenght
            if di_id == dj_id or abs(docs_lens[di_id] - docs_lens[dj_id]) < args.delta:
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
                if not(di_terms_counter[wa] == dj_terms_counter[wb] and di_terms_counter[wb] == dj_terms_counter[wa]):
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


def LNC1(topic_id, tokenized_query, all_docs, tuples, docs_lens, args, scores):
    query = ' '.join([x for x in tokenized_query]).replace("##", "")
    query = [x.replace("##", "") for x in tokenized_query]
    query_terms = set(query)
    instances = []
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


def TPC(topic_id, tokenized_query, all_docs, tuples, docs_lens, args, scores):
    query = ' '.join([x for x in tokenized_query]).replace("##", "")
    query = [x.replace("##", "") for x in tokenized_query]
    query_terms = set(query)
    instances = []
    query_pairs = product(query_terms, query_terms)
    query_pairs = [x for x in query_pairs if x[0] != x[1]]
    for di_id in tqdm(tuples[topic_id]):
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


def STMC1(topic_id, tokenized_query, all_docs, tuples, docs_lens, args, scores):
    '''
    Let Q = {q} be a query with only one term q. -> Relaxation - Multiple terms
    Let D1 = {d1} and D2 = {d2} be two single-term documents, where q != d1 and q != d2.
        - > Relaxation: At least one query term is present in di but not in dj
    If s(q, d1) > s(q, d2), then -> if average term embedding differ
    S(Q, D1) > S(Q, D2).

    di and dj has the same number of query terms. Remove all query terms. Check similarity with query.
        should correspond to score.
    '''

    query = [x.replace("##", "") for x in tokenized_query]
    query_terms = set(query)
    instances = []
    _docs = list(tuples[topic_id])
    query_avg = np.mean(args.vectors[[x for x in query if x in args.vectors.vocab]], axis=0)
    for i, di_id in tqdm(enumerate(_docs), total=100):
        di_text = all_docs[di_id]
        # remove query terms
        di_clean = [w for w in di_text if w not in query_terms]
        di_query_terms = [w for w in di_text if w in query_terms]
        di_avg = np.mean(args.vectors[[x for x in di_clean if x in args.vectors.vocab]], axis=0)
        di_similarity = cosine(di_avg, query_avg)
        for dj_id in _docs[i:]:
            if di_id == dj_id:
                continue
            dj_text = all_docs[dj_id]
            dj_query_terms = [w for w in dj_text if w in query_terms]
            dj_clean_text = [w for w in dj_text if w not in query_terms]
            dj_avg = np.mean(args.vectors[[x for x in dj_clean_text if x in args.vectors.vocab]], axis=0)
            # di_id and dj_id must have same number of query terms
            if len(di_query_terms) != len(dj_query_terms):
                continue
            # at least one query term occur in di but not in dj
            dj_similarity = cosine(dj_avg, query_avg)
            if di_similarity > dj_similarity:
                instances.append((topic_id, di_id, dj_id))
            elif dj_similarity > di_similarity:
                instances.append((topic_id, dj_id, di_id))
    return instances


def STMC2(topic_id, tokenized_query, all_docs, tuples, docs_lens, args, scores):
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
    query = [x.replace("##", "") for x in tokenized_query]
    query_terms = set(query)
    instances = []
    pbar = tqdm(total=4950)
    _docs = list(tuples[topic_id])
    for i, di_id in enumerate(_docs):
        # remove query terms
        di_clean = [w for w in all_docs[di_id] if w not in query_terms]
        di_query_terms = [w for w in all_docs[di_id] if w in query_terms]
        di_avg = np.mean(args.vectors[[x for x in di_clean if x in args.vectors.vocab]], axis=0)
        for dj_id in _docs[i:]:
            pbar.update()
            if di_id == dj_id:
                continue
            dj_clean = [w for w in all_docs[dj_id] if w not in query_terms]
            # di and dj must be similar
            dj_avg = np.mean(args.vectors[[x for x in dj_clean if x in args.vectors.vocab]], axis=0)
            di_dj_distance = cosine(di_avg, dj_avg)
            if di_dj_distance > args.stmc_sim:
                continue
            dj_query_terms = [w for w in all_docs[dj_id] if w in query_terms]
            # di must have more query terms than dj
            if len(dj_query_terms) > len(di_query_terms):
                instances.append((topic_id, dj_id, di_id))
            elif len(dj_query_terms) < len(di_query_terms):
                instances.append((topic_id, di_id, dj_id))
    return instances


def STMC3(topic_id, tokenized_query, all_docs, tuples, docs_lens, args, scores):
    '''
    conditions: D1 and D2 covers the same number of query terms
    D1 and D2 are approx. the same size
    D1 has more query terms than D2.
    D2 without query terms is more similar to q than D1.
    '''
    query = [x.replace("##", "") for x in tokenized_query]
    query_terms = set(query)
    instances = []
    query_avg = np.mean(args.vectors[[x for x in query if x in args.vectors.vocab]], axis=0)
    _docs = list(tuples[topic_id])
    for i, di_id in tqdm(enumerate(_docs), total=100):
        di_clean = [w for w in all_docs[di_id] if w not in query_terms]  # Di without query terms
        di_query_terms = [w for w in all_docs[di_id] if w in query_terms]
        di_query_set = set(di_query_terms)
        di_avg = np.mean(args.vectors[[x for x in di_clean if x in args.vectors.vocab]], axis=0)
        di_query_sim = cosine(di_avg, query_avg)
        for dj_id in _docs[i:]:
            # similarity
            if di_id == dj_id or abs(docs_lens[di_id] - docs_lens[dj_id]) >= args.delta:
                continue
            dj_query_terms = [w for w in all_docs[dj_id] if w in query_terms]
            dj_query_set = set(dj_query_terms)

            # both must cover same number of terms
            if len(dj_query_set) != len(di_query_set):
                continue
            dj_clean = [w for w in all_docs[dj_id] if w not in query_terms]
            dj_avg = np.mean(args.vectors[[x for x in dj_clean if x in args.vectors.vocab]], axis=0)
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
        docs_path = os.path.join(config.data_home, "docs/msmarco-docs.tokenized.cut.bert")
    else:
        docs_path = os.path.join(config.data_home, "docs/msmarco-docs.tokenized.bert")
    offset_dict = pickle.load(open(docs_path + ".offset", 'rb'))
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
    queries_file = os.path.join(config.data_home, "queries/test.tokenized.bert")
    assert os.path.isfile(queries_file), "could not find queries file at %s" % queries_file
    diagnostics_path = os.path.join(config.data_home, "diagnostics")
    if not os.path.isdir(diagnostics_path):
        os.mkdir(diagnostics_path)
    all_lines = [(x.split("\t")[0], eval(x.split("\t")[1])) for x in open(queries_file).readlines()]
    scores = {}
    for line in open(top_100_path):
        topic_id, _, doc_id, _, score, _ = line.split(" ")
        scores["{}-{}".format(topic_id, doc_id)] = float(score)
    for axiom in axioms:
        if axiom == "MTDC":  # We need IDFs!
            IDF_folder = os.path.join(config.data_home, "docs/IDFS/IDFS-FULL")
            if not os.path.isfile(IDF_folder):
                # Generate IDFs
                compute_IDFS(IDF_folder.replace("/IDFS-FULL", ""), cut)
            return
        if axiom in ["TFC1", "TFC2"]:  # These axioms are too fast to be ran in paralel. Just do it in serial. Faster.
            cpus = 0
        else:
            cpus = config.number_of_cpus
        logging.info("Running axiom %s with %i cpus and %i lines" % (axiom, config.number_of_cpus, len(all_lines)))
        if cpus > 1:
            pool = mp.Pool(config.number_of_cpus)
            jobs = []
            pbar = tqdm(total=len(all_lines))
            # manager = Manager()
            # all_docs = manager.dict(all_docs)
            # docs_lens = manager.dict(docs_lens)
            # scores = manager.dict(scores)
            # args = manager.dict(dict(config))
            args = dict(config)
            instances = []

            def update(*a):
                pbar.update()
            f = globals()[axiom]
            jobs = []
            for i in all_lines:
                jobs.append(pool.apply_async(f, args=(i, all_docs, tuples, docs_lens, args, scores), callback=update))
            for job in jobs:
                job.get()
            pool.close()
            pbar.close()
        else:
            instances = []
            for sample in tqdm(all_lines, desc="Processing {}".format(axiom)):
                instances.append(globals()[axiom](sample, all_docs, tuples, docs_lens, dict(config), scores))
            print(len(instances))
        pickle.dump(instances, open(os.path.join(diagnostics_path, "{}-instances".format(axiom)), 'wb'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_home", type=str, default="/ssd2/arthur/TREC2019/data")
    parser.add_argument("--docs_file", type=str, default="docs/tokenized-msmarco-docs.tsv")
    parser.add_argument("--queries_file", type=str, default="queries/test_queries.tsv")
    parser.add_argument("--axioms", type=str, default="TFC1,TFC2,MTDC,LNC1,LNC2,LB1,LB2,STMC1,STMC2,STMC3,TP")
    parser.add_argument("--top100", type=str, default="runs/indri_test_10_10.run")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--total_docs", type=int, default=3213835)
    parser.add_argument("--delta", type=int, default=10)
    parser.add_argument("--idf_file", type=str, default="docs/IDFS/IDFS-FULL")
    parser.add_argument("--embeddings_path", type=str, default="GloVe/w2v.txt")
    parser.add_argument("--stmc_sim", type=float, default=0.2)

    if len(sys.argv) < 2:
        argv = ["--cpus", "1",
                "--axioms", "TPC"]
        args = parser.parse_args(argv)
    else:
        args = parser.parse_args(sys.argv[1:])
    print(args)

    docs_path = os.path.join(args.data_home, args.docs_file)
    offset_dict = pickle.load(open(docs_path + ".offset", 'rb'))
    top_100_path = os.path.join(args.data_home, args.top100)
    assert os.path.isfile(top_100_path)
    assert os.path.isfile(docs_path)

    tuples = defaultdict(lambda: set())
    all_docs = dict()
    docs_lens = dict()
    for i in tqdm(open(top_100_path), total=155800, desc="loading docs and queries"):
        topic_id, _, doc_id, _, score, _ = i.split()
        tuples[topic_id].add(doc_id)
        if doc_id in all_docs:
            continue
        all_docs[doc_id] = getcontent(doc_id, docs_path, offset_dict).split()
        docs_lens[doc_id] = len(all_docs[doc_id])
    tuples = dict(tuples)

    queries_file = os.path.join(args.data_home, args.queries_file)
    assert os.path.isfile(queries_file)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    diagnostics_path = os.path.join(args.data_home, "diagnostics")
    if not os.path.isdir(diagnostics_path):
        os.mkdir(diagnostics_path)

    all_lines = [(x.split("\t")[0], tokenizer.tokenize(x.split("\t")[1])) for x in open(queries_file).readlines()]
    # qrels_path = os.path.join(args.data_home, 'qrels', "test_qrels")
    qrels = {}
    for line in open("/ssd2/arthur/TREC2019/data/runs/indri_test_10_10.run"):
        topic_id, _, doc_id, _, score, _ = line.split(" ")
        qrels["{}-{}".format(topic_id, doc_id)] = float(score)
        # if topic_id in qrels:
        #     qrels[topic_id].append(doc_id)
        # else:
        #     qrels[topic_id] = [doc_id]
    args.qrels = qrels

    for axiom in args.axioms.split(","):
        instances = []
        if "STMC" in axiom:
            # load GloVe embeddings
            vectors = KeyedVectors.load_word2vec_format(os.path.join(args.data_home, args.embeddings_path))
            vectors.init_sims(replace=True)
            args.vectors = vectors
            pbar = tqdm(total=len(all_lines))

        if args.cpus > 1:
            pool = mp.Pool(args.cpus)
            jobs = []

            manager = Manager()
            all_docs = manager.dict(all_docs)
            docs_lens = manager.dict(docs_lens)

            def update(instance):
                pbar.update()
                instances.append(instance)

            for q_id, tokenized_query in all_lines:
                jobs.append(pool.apply_async(globals()[axiom], args=(q_id, tokenized_query, all_docs, tuples, docs_lens, args), callback=update)) # noqa E301
            for p in jobs:
                p.get()
            pbar.close()
            pool.close()
        else:
            for q_id, tokenized_query in tqdm(all_lines):
                instances += globals()[axiom](q_id, tokenized_query, all_docs, tuples, docs_lens, args)

        pickle.dump(instances, open(os.path.join(diagnostics_path, "{}-instances".format(axiom)), 'wb'))


if __name__ == "__main__":
    main()
