import argparse
import sys
import os
from collections import Counter
from itertools import product
from tqdm.auto import tqdm


def TFC1(docs, query, last_topic_id, delta, scores):
    number_of_candidates = 0
    query_terms = set(query)
    satisfies = 0
    for di_id in docs.keys():
        di = docs[di_id]
        di_terms_counter = Counter(di)
        di_pair_id = "{}-{}".format(last_topic_id, di_id)
        sum_occurences_di = sum([di_terms_counter[term]
                                 for term in query_terms])
        for dj_id in docs.keys():
            if di_id == dj_id:
                continue
            dj = docs[dj_id]
            if abs(len(di) - len(dj)) > delta:
                continue

            dj_terms_counter = Counter(dj)
            dj_pair_id = "{}-{}".format(last_topic_id, dj_id)
            sum_occurences_dj = sum([dj_terms_counter[term]
                                     for term in query_terms])

            # Axiom premisses
            for w in query_terms:
                if di_terms_counter[w] < dj_terms_counter[w]:
                    continue
            if sum_occurences_di <= sum_occurences_dj:
                continue
            number_of_candidates += 1

            # Satisfies premisses. Check Scores.
            if scores[di_pair_id] > scores[dj_pair_id]:
                satisfies += 1
    return number_of_candidates, satisfies


def TFC2(docs, query, last_topic_id, delta, scores):
    number_of_candidates = 0
    query_terms = set(query)
    satisfies = 0
    delta = 4
    for d1_id, d2_id, d3_id in tqdm(product(docs.keys(), docs.keys(), docs.keys()), total=len(docs.keys())**3):
        if d1_id == d2_id or d2_id == d3_id or d1_id == d3_id:
            continue
        d1 = docs[d1_id]
        TF_d1 = Counter(d1)

        d2 = docs[d2_id]
        TF_d2 = Counter(d2)

        d3 = docs[d3_id]
        TF_d3 = Counter(d3)
        # Every document must contain at least one query term
        if (len(query_terms.intersection(TF_d1)) < 1
                or len(query_terms.intersection(TF_d2)) < 1
                or len(query_terms.intersection(TF_d3)) < 1):
            continue
        # Doc lenghts should be within an acceptable range
        if (abs(len(d3) - len(d1)) > delta
                or abs(len(d2) - len(d1)) > delta
                or abs(len(d3) - len(d2)) > delta):
            continue

        sum_TF_d1 = sum([TF_d1[w] for w in query_terms])
        sum_TF_d2 = sum([TF_d2[w] for w in query_terms])
        sum_TF_d3 = sum([TF_d3[w] for w in query_terms])

        if sum_TF_d3 <= sum_TF_d2 or sum_TF_d2 <= sum_TF_d1:
            continue
        for w in query_terms:
            if (TF_d2[w] - TF_d1[w]) == (TF_d3[w] - TF_d2[w]):
                continue
        number_of_candidates += 1
        d1_pair_id = "{}-{}".format(last_topic_id, d1_id)
        d2_pair_id = "{}-{}".format(last_topic_id, d2_id)
        d3_pair_id = "{}-{}".format(last_topic_id, d3_id)
        if (scores[d2_pair_id] - scores[d1_pair_id]) > (scores[d3_pair_id] - scores[d2_pair_id]):
            satisfies += 1
    return number_of_candidates, satisfies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=int, default=3, help="delta for TFC-1")
    parser.add_argument("--data_home", type=str,
                        default="/ssd2/arthur/TREC2019/data/", help="Home diretory for data")

    axioms = [TFC2]
    datasets = ["test"]

    if len(sys.argv) > 2:
        args = parser.parse_args(sys.argv[1:])
    else:
        argv = []
        args = parser.parse_args(argv)

    methods = {
        "QL": os.path.join(args.data_home, 'runs', "{}_distilBert-0.0.run"),
        "BERT": os.path.join(args.data_home, 'runs', "{}_distilBert-1.0.run"),
        "QL+BERT": os.path.join(args.data_home, 'runs', "{}_distilBert-0.85.run")
    }

    for method in methods.keys():

        for dataset in datasets:
            last_topic_id = -1
            docs = {}
            scores = {}

            all_candidates = 0
            satisfies = 0
            query = None
            run_file = methods[method].format(dataset)

            with open(run_file) as inf:
                for line in inf:
                    topic_id, _, doc_id, _, score, _ = line.split()
                    pair_id = "{}-{}".format(topic_id, doc_id)
                    scores[pair_id] = float(score)
            triples_file = os.path.join(
                args.data_home, 'triples-tokenized', '{}-triples.top100'.format(dataset))
            pbar = tqdm(total=1558, desc="topics computed")
            with open(triples_file) as inf:
                for index, line in enumerate(inf):
                    topic_id, doc_id = line.split("\t")[0].split("-")
                    if topic_id != last_topic_id and last_topic_id != -1:
                        topic_candidates, topic_satisfies = TFC2(
                            docs, query, last_topic_id, args.delta, scores)
                        pbar.update()
                        all_candidates += topic_candidates
                        satisfies += topic_satisfies
                        docs = {}

                    last_topic_id = topic_id
                    doc = eval(line.split("\t")[1])
                    sep_id = doc.index("[SEP]")
                    document = doc[sep_id + 1:-1]
                    docs[doc_id] = document
                    query = doc[1:sep_id]
            print(method, dataset, all_candidates,
                  satisfies, satisfies / all_candidates)
