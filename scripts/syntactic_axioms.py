import argparse
import sys
import os
import pickle
import multiprocessing as mp
mp.set_start_method('spawn', True)


def TFC1(scores, args):
    dataset_path = os.path.join(args.data_home, args.datasets_path, "TFC1-instances")
    dataset = pickle.load(open(dataset_path, 'rb'))
    agreements = 0
    for topic_id, di_id, dj_id in dataset:
        guid1 = f"{topic_id}-{di_id}"
        guid2 = f"{topic_id}-{dj_id}"
        if scores[guid1] > scores[guid2]:
            agreements += 1
    return len(dataset), agreements, agreements / len(dataset)


def TFC2(scores, args):
    dataset_path = os.path.join(args.data_home, args.datasets_path, "TFC2-instances")
    dataset = pickle.load(open(dataset_path, 'rb'))
    agreements = 0
    for topic_id, di_id, dj_id, dk_id in dataset:
        guidi = f"{topic_id}-{di_id}"
        guidj = f"{topic_id}-{dj_id}"
        guidk = f"{topic_id}-{dk_id}"
        if (scores[guidj] - scores[guidi]) > (scores[guidk] - scores[guidj]):
            agreements += 1
    return len(dataset), agreements, agreements / len(dataset)


def MTDC(scores, args):
    dataset_path = os.path.join(args.data_home, args.datasets_path, "MTDC-instances")
    dataset = pickle.load(open(dataset_path, 'rb'))
    agreements = 0
    for topic_id, di_id, dj_id in dataset:
        guid1 = f"{topic_id}-{di_id}"
        guid2 = f"{topic_id}-{dj_id}"
        if scores[guid1] > scores[guid2]:
            agreements += 1
    return len(dataset), agreements, agreements / len(dataset)


def LNC1(scores, args):
    dataset_path = os.path.join(args.data_home, args.datasets_path, "LNC1-instances")
    dataset = pickle.load(open(dataset_path, 'rb'))
    agreements = 0
    for topic_id, di_id, dj_id in dataset:
        guid1 = f"{topic_id}-{di_id}"
        guid2 = f"{topic_id}-{dj_id}"
        if scores[guid1] >= scores[guid2]:
            agreements += 1
    return len(dataset), agreements, agreements / len(dataset)


def TP(scores, args):
    dataset_path = os.path.join(args.data_home, args.datasets_path, "TPC-instances")
    dataset = pickle.load(open(dataset_path, 'rb'))
    agreements = 0
    for topic_id, di_id, dj_id in dataset:
        guid1 = f"{topic_id}-{di_id}"
        guid2 = f"{topic_id}-{dj_id}"
        if scores[guid1] > scores[guid2]:
            agreements += 1
    return len(dataset), agreements, agreements / len(dataset)


def STMC1(scores, args):
    '''
    Let Q = {q} be a query with only one term q. -> Relaxation - Multiple terms
    Let D1 = {d1} and D2 = {d2} be two single-term documents, where q != d1 and q != d2. - > Relaxation: At least one query term is present in di but not in dj
    If s(q, d1) > s(q, d2), then -> if average term embedding differ
    S(Q, D1) > S(Q, D2).

    di and dj has the same number of query terms. Remove all query terms. Check similarity with query. should correspond to score.
    '''
    dataset_path = os.path.join(args.data_home, args.datasets_path, "STMC1-instances")
    dataset = pickle.load(open(dataset_path, 'rb'))
    agreements = 0
    if len(dataset) == 1558:
        dataset = [item for sublist in dataset for item in sublist]
    for topic_id, di_id, dj_id in dataset:
        guid1 = f"{topic_id}-{di_id}"
        guid2 = f"{topic_id}-{dj_id}"
        if scores[guid1] > scores[guid2]:
            agreements += 1
    return len(dataset), agreements, agreements / len(dataset)


def STMC2(scores, args):
    '''
    Let Q = {q} be a single term query -> Multiple terms
    and d be a non-query term such that s(q, d) > 0. 
    If D1 and D2 are  two documents such that |D1| = 1, c(q, D1) = 1, -> At least one term from the query is present in the document
    |D2| = k and c(d, D2) = k (k ≥ 1), where c(q, D1) and c(d, D2) are the counts of q and d in D1 and D2 respectively, then S(Q, D1) ≥ S(Q, D2).

    d1 w/out query terms approx.equal to d2 w/out query terms
    sum(d1_terms) > sum(d2_terms).
    S(d1) > S(d2)
    '''
    dataset_path = os.path.join(args.data_home, args.datasets_path, "STMC2-instances")
    dataset = pickle.load(open(dataset_path, 'rb'))
    agreements = 0

    if len(dataset) == 1558:
        dataset = [item for sublist in dataset for item in sublist]
    for topic_id, di_id, dj_id in dataset:
        guid1 = f"{topic_id}-{di_id}"
        guid2 = f"{topic_id}-{dj_id}"
        if scores[guid1] >= scores[guid2]:
            agreements += 1
    return len(dataset), agreements, agreements / len(dataset)


def STMC3(scores, args):\
    # conditions: D1 and D2 covers the same number of query terms
    # D1 and D2 are approx. the same size
    # D1 has more query terms than D2.
    # D2 without query terms is more similar to q than D1.


    dataset_path = os.path.join(args.data_home, args.datasets_path, "STMC3-instances")
    dataset = pickle.load(open(dataset_path, 'rb'))
    if len(dataset) == 1558:
        dataset = [item for sublist in dataset for item in sublist]
    agreements = 0
    for topic_id, di_id, dj_id in dataset:
        guid1 = f"{topic_id}-{di_id}"
        guid2 = f"{topic_id}-{dj_id}"
        if scores[guid1] <= scores[guid2]:
            agreements += 1
    return len(dataset), agreements, agreements / len(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_home", type=str, default="/ssd2/arthur/TREC2019/data/")
    parser.add_argument("--axioms", type=str)
    parser.add_argument("--datasets_path", default="diagnostics")

    docoffset = {}

    if len(sys.argv) > 2:
        args = parser.parse_args(sys.argv[1:])
    else:
        argv = [
            "--axioms", "TFC1"]
        args = parser.parse_args(argv)

    methods = {
        "QL": os.path.join(args.data_home, 'runs', "test_distilBert-0.0.run"),
        "BERT": os.path.join(args.data_home, 'runs', "test_distilBert-1.0.run"),
        "QL+BERT": os.path.join(args.data_home, 'runs', "test_distilBert-0.85.run")
    }

    for axiom in args.axioms.split(","):
        print(axiom)
        for method in methods:
            scores = dict()
            run_file = methods[method]

            with open(run_file) as inf:
                for line in inf:
                    topic_id, _, doc_id, _, score, _ = line.split()
                    pair_id = "{}-{}".format(topic_id, doc_id)
                    scores[pair_id] = float(score)
            print("\t", method, globals()[axiom](scores, args))
