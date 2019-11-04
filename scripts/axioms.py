import os
import pickle
import multiprocessing as mp
import wandb
import logging
import subprocess
import re
mp.set_start_method('spawn', True)


def TFC1(scores, args):
    dataset_path = os.path.join(args.data_home, "diagnostics/TFC1-instances")
    dataset = pickle.load(open(dataset_path, 'rb'))
    wandb.log({"TFC1 size": len(dataset)})
    logging.info("TFC has size %i" % (len(dataset)))
    agreements = 0
    for topic_id, di_id, dj_id in dataset:
        guid1 = f"{topic_id}-{di_id}"
        guid2 = f"{topic_id}-{dj_id}"
        if scores[guid1] > scores[guid2]:
            agreements += 1
    return len(dataset), agreements, agreements / len(dataset)


def TFC2(scores, args):
    dataset_path = os.path.join(args.data_home, "diagnostics/TFC2-instances")
    dataset = pickle.load(open(dataset_path, 'rb'))
    wandb.log({"TFC2 size": len(dataset)})
    logging.info("TFC2 has size %i" % (len(dataset)))
    agreements = 0
    for topic_id, di_id, dj_id, dk_id in dataset:
        guidi = f"{topic_id}-{di_id}"
        guidj = f"{topic_id}-{dj_id}"
        guidk = f"{topic_id}-{dk_id}"
        if (scores[guidj] - scores[guidi]) > (scores[guidk] - scores[guidj]):
            agreements += 1
    return len(dataset), agreements, agreements / len(dataset)


def MTDC(scores, args):
    dataset_path = os.path.join(args.data_home, "diagnostics/MTDC-instances")
    dataset = pickle.load(open(dataset_path, 'rb'))
    wandb.log({"MTDC size": len(dataset)})
    logging.info("MTDC has size %i" % (len(dataset)))
    agreements = 0
    for topic_id, di_id, dj_id in dataset:
        guid1 = f"{topic_id}-{di_id}"
        guid2 = f"{topic_id}-{dj_id}"
        if scores[guid1] > scores[guid2]:
            agreements += 1
    return len(dataset), agreements, agreements / len(dataset)


def LNC1(scores, args):
    dataset_path = os.path.join(args.data_home, "diagnostics/LNC1-instances")
    dataset = pickle.load(open(dataset_path, 'rb'))
    wandb.log({"LNC1 size": len(dataset)})
    logging.info("LNC1 has size %i" % (len(dataset)))
    agreements = 0
    for topic_id, di_id, dj_id in dataset:
        guid1 = f"{topic_id}-{di_id}"
        guid2 = f"{topic_id}-{dj_id}"
        if scores[guid1] >= scores[guid2]:
            agreements += 1
    return len(dataset), agreements, agreements / len(dataset)


def LNC2(scores, args):
    """Ignores every parameter. Load own run files, call C++ code and check BERT run file"""
    dataset_path = os.path.join(args.data_home, "diagnostics/LNC2-instances")
    dataset = pickle.load(open(dataset_path, 'rb'))
    wandb.log({"LNC2 size": len(dataset)})
    logging.info("LNC2 has size %i" % (len(dataset)))
    agreements = 0
    scores = {}
    run_file = os.path.join(args["data_home"], "runs/LNC-BERT-cut.run")
    for line in open(run_file):
        topic_id, _, doc_id, _, score, _ = line.split()
        pair_id = "{}-{}".format(topic_id, doc_id)
        scores[pair_id] = float(score)

    for topic_id, di_id, dj_id in dataset:
        guid1 = f"{topic_id}-{di_id}"
        guid2 = f"{topic_id}-{dj_id}"
        if scores[guid1] < scores[guid2]:
            agreements += 1
    logging.info("agreement for axiom LNC2 and method BERT: %f" % (agreements / len(dataset)))
    # Do the same for the C++ code
    code_path = args["LNC2_path"]
    param_path = os.path.join(args["data_home"], "indri_params/LNC2.param")
    cmd = f"{code_path} {param_path} 1"
    s = ("Running LNC2 with C++ code. May take a few minutes. DO NOT kill the process."
            "Otherwise, a new index will need to be created")  # noqa E221
    logging.info(s)
    output = subprocess.check_output(cmd.split())
    pattern = re.compile(r"\:([^,]*\'*)")
    instances, correct = map(int, pattern.findall(output.decode().split("\n")[-2]))
    logging.info("agreement for axiom LNC2 and method QL: %f" % (correct / instances))    
    return len(dataset), agreements, agreements / len(dataset)


def TPC(scores, args):
    dataset_path = os.path.join(args.data_home, "diagnostics/TPC-instances")
    dataset = pickle.load(open(dataset_path, 'rb'))
    wandb.log({"TPC size": len(dataset)})
    logging.info("TPC has size %i" % (len(dataset)))
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
    Let D1 = {d1} and D2 = {d2} be two single-term documents,
        where q != d1 and q != d2. - > Relaxation: At least one query term is present in di but not in dj
    If s(q, d1) > s(q, d2), then -> if average term embedding differ
    S(Q, D1) > S(Q, D2).

    di and dj has the same number of query terms.
    Remove all query terms.
    Check similarity with query. should correspond to score.
    '''
    dataset_path = os.path.join(args.data_home, "diagnostics/STMC1-instances")
    dataset = pickle.load(open(dataset_path, 'rb'))
    wandb.log({"STMC1 size": len(dataset)})
    logging.info("STMC1 has size %i" % (len(dataset)))
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
    If D1 and D2 are  two documents such that |D1| = 1, c(q, D1) = 1, ->
        At least one term from the query is present in the document
    |D2| = k and c(d, D2) = k (k ≥ 1), where c(q, D1) and c(d, D2) are the counts of q and d in D1 and D2 respectively,
        then S(Q, D1) ≥ S(Q, D2).

    d1 w/out query terms approx.equal to d2 w/out query terms
    sum(d1_terms) > sum(d2_terms).
    S(d1) > S(d2)
    '''
    dataset_path = os.path.join(args.data_home, "diagnostics/STMC2-instances")
    dataset = pickle.load(open(dataset_path, 'rb'))
    wandb.log({"STMC2 size": len(dataset)})
    logging.info("STMC2 has size %i" % (len(dataset)))

    agreements = 0

    if len(dataset) == 1558:
        dataset = [item for sublist in dataset for item in sublist]
    for topic_id, di_id, dj_id in dataset:
        guid1 = f"{topic_id}-{di_id}"
        guid2 = f"{topic_id}-{dj_id}"
        if scores[guid1] >= scores[guid2]:
            agreements += 1
    return len(dataset), agreements, agreements / len(dataset)


def STMC3(scores, args):
    # conditions: D1 and D2 covers the same number of query terms
    # D1 and D2 are approx. the same size
    # D1 has more query terms than D2.
    # D2 without query terms is more similar to q than D1.

    dataset_path = os.path.join(args.data_home, "diagnostics/STMC3-instances")
    dataset = pickle.load(open(dataset_path, 'rb'))
    wandb.log({"STMC3 size": len(dataset)})
    logging.info("STMC3 has size %i" % (len(dataset)))

    if len(dataset) == 1558:
        dataset = [item for sublist in dataset for item in sublist]
    agreements = 0
    for topic_id, di_id, dj_id in dataset:
        guid1 = f"{topic_id}-{di_id}"
        guid2 = f"{topic_id}-{dj_id}"
        if scores[guid1] <= scores[guid2]:
            agreements += 1
    return len(dataset), agreements, agreements / len(dataset)


def check_axioms(cut):
    config = wandb.config
    axioms = ["TFC1", "TFC2", "MTDC", "LNC1", "LNC2", "TPC", "STMC1", "STMC2", "STMC3"]  # TODO change for parameters
    methods = {
        "QL": os.path.join(config.data_home, "runs/{}-test-alpha_0.0.run".format(cut)),
        "BERT": os.path.join(config.data_home, "runs/{}-test-alpha_1.0.run".format(cut))
    }
    for axiom in axioms:
        logging.info("Checking axiom %s" % axiom)
        f = globals()[axiom]
        for method, run_file in methods.items():
            scores = dict()
            with open(run_file) as inf:
                for line in inf:
                    topic_id, _, doc_id, _, score, _ = line.split()
                    pair_id = "{}-{}".format(topic_id, doc_id)
                    scores[pair_id] = float(score)
            result = f(scores, config)
            if axiom != "LNC2":
                logging.info("agreement for axiom %s and method %s: %f" % (axiom, method, result[2]))
            else:
                break

