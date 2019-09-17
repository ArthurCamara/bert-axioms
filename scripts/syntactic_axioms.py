import argparse
import sys
import os
from collections import Counter
from itertools import product
from tqdm.auto import tqdm
from pytorch_transformers import BertTokenizer
import pickle
import csv
import multiprocessing as mp
mp.set_start_method('spawn', True)

def getcontent(docid, file_name):
    """getcontent(docid, f) will get content for a given docid (a string) from filehandle f.
    The content has four tab-separated strings: docid, url, title, body.
    """
    with open(file_name, encoding='utf-8') as f:
        f.seek(docoffset[docid])
        line = f.readline()
        assert line.startswith(docid + "\t"), \
            f"Looking for {docid}, found {line}"
    return line.rstrip()

def TFC1(docs, query, last_topic_id, delta, scores, tokenizer):
    number_of_candidates = 0
    query_terms = set(tokenizer.tokenize(" ".join(query)))
    satisfies = 0
    for di_id in docs.keys():
        di = docs[di_id]  # document text
        di_terms_counter = Counter(di)  # document TF
        di_pair_id = "{}-{}".format(last_topic_id, di_id)
        sum_occurences_di = sum([di_terms_counter[term]
                                 for term in query_terms])  # Sum of TFs
        for dj_id in docs.keys():
            if di_id == dj_id:  # If it's the same document, skip
                continue
            dj = docs[dj_id]
            if abs(len(di) - len(dj)) > delta:  # if they are too diferent, skip
                continue

            dj_terms_counter = Counter(dj)
            dj_pair_id = "{}-{}".format(last_topic_id, dj_id)
            sum_occurences_dj = sum([dj_terms_counter[term]
                                     for term in query_terms])

            # Axiom premisses
            flag = True
            for w in query_terms:
                if di_terms_counter[w] < dj_terms_counter[w]:  # w must occur more on di
                    flag = False
                    break
            if flag and sum_occurences_di > sum_occurences_dj:
                number_of_candidates += 1
            else:
                continue
            # Satisfies premisses. Check Scores.
            if scores[di_pair_id] > scores[dj_pair_id]:  # Score for di is larger than dj?
                satisfies += 1
    if number_of_candidates == 0:
        return (0, 0)
    return number_of_candidates, satisfies


def TFC2(docs, query, last_topic_id, delta, scores, tokenizer):
    number_of_candidates = 0
    query_terms = set(tokenizer.tokenize(" ".join(query)))
    satisfies = 0
    for d1_id, d2_id, d3_id in product(docs.keys(), docs.keys(), docs.keys()):
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
        flag = False
        for w in query_terms:
            if (TF_d2[w] - TF_d1[w]) == (TF_d3[w] - TF_d2[w]):
                flag = True
                break
        if flag:
            continue
        number_of_candidates += 1
        d1_pair_id = "{}-{}".format(last_topic_id, d1_id)
        d2_pair_id = "{}-{}".format(last_topic_id, d2_id)
        d3_pair_id = "{}-{}".format(last_topic_id, d3_id)
        if (scores[d2_pair_id] 
        - scores[d1_pair_id]) > (scores[d3_pair_id] - scores[d2_pair_id]):
            satisfies += 1
    return number_of_candidates, satisfies


def MTDC(docs, query, last_topic_id, delta, scores, tokenizer):
    IDFS = pickle.load(
        open("/ssd2/arthur/TREC2019/data/docs/IDFS/IDFS-FULL", 'rb'))

    number_of_candidates = 0
    satisfies = 0
    query = tokenizer.tokenize(" ".join(query))
    query_terms = set(query)
    query_terms_counter = Counter(query)
    for d1_id in docs.keys():
        d1 = docs[d1_id]
        # d1_terms_counter = Counter(d1)
        d1_terms_counter = Counter([w for w in d1 if w in query_terms])
        d1_tf_sum = sum([d1_terms_counter[term]
                        for term in query_terms])
        for d2_id in docs.keys():
            if d1_id == d2_id:
                continue
            d2 = docs[d2_id]
            if abs(len(d1) - len(d2)) > delta:
                continue
            # d2_terms_counter = Counter(d2)
            d2_terms_counter = Counter([w for w in d2 if w in query_terms])
            d2_tf_sum = sum([d2_terms_counter[term]
                             for term in query_terms])
            # Candidate query-terms are the ones that c(w, d1) != c(w, d2)
            valid_query_terms = [
                w for w in query_terms if d1_terms_counter[w] != d2_terms_counter[w]]

            # document pair must differ in at least one query term
            if d1_terms_counter == d2_terms_counter:
                continue

            # document pair must have same tf sum
            if d1_tf_sum != d2_tf_sum:
                continue

            # for each query term pair, it must be valid. It's valid if:
            # 1. idf(wa) >= idf(wb)
            # 2. c(wa, d1) = c(wb, d2) and c(wb, d1) = c(wa, d2)
            # 3 c(wa,d1) > c(wa, d2)
            # 4 c(wa, q) >= c(wb, q)
            query_terms_used = set()
            for wa, wb in product(valid_query_terms, valid_query_terms):
                if (wa == wb):
                    continue
                if ((1/IDFS[wa]) < (1/IDFS[wb])):  # 1
                    continue
                if (d1_terms_counter[wa] != d2_terms_counter[wb]) or (d1_terms_counter[wb] != d2_terms_counter[wa]):
                    continue
                if d1_terms_counter[wa] <= d2_terms_counter[wa]:  # 3
                    continue
                if query_terms_counter[wa] < query_terms_counter[wb]:  # 4
                    continue
                query_terms_used.add(wa)
                query_terms_used.add(wb)
            if query_terms_used == set(valid_query_terms):
                # now, this triple is valid. check for scores
                number_of_candidates += 1
            else:
                continue
            # check score
            d1_pair_id = "{}-{}".format(last_topic_id, d1_id)
            d2_pair_id = "{}-{}".format(last_topic_id, d2_id)
            if scores[d1_pair_id] > scores[d2_pair_id]:
                satisfies += 1
    if number_of_candidates == 0:
        return (0,0)
    print(number_of_candidates, satisfies, satisfies / number_of_candidates)
    return number_of_candidates, satisfies


# def LNC2(docs, query, last_topic_id, delta, scores):


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=int, default=300, help="delta for TFC-1")
    parser.add_argument("--data_home", type=str,
                        default="/ssd2/arthur/TREC2019/data/", help="Home diretory for data")
    parser.add_argument("--axiom", type=str)
    parser.add_argument("--cpus", type=int, default=32)

    datasets = ["test"]

    docoffset = {}

    if len(sys.argv) > 2:
        args = parser.parse_args(sys.argv[1:])
    else:
        argv = [
            "--cpus", "1",
            "--axiom", "MTDC"]
        args = parser.parse_args(argv)
    if not args.axiom:
        axiom = MTDC
    else:
        if args.axiom == "MTDC":
            axiom = MTDC
        elif args.axiom == "TFC1":
            axiom = TFC1
        elif args.axiom == "TFC2":
            axiom = TFC2
        elif args.axiom == "LNC2":
            raise NotImplementedError
        else:
            raise NotImplementedError

    methods = {
        "QL": os.path.join(args.data_home, 'runs', "{}_distilBert-0.0.run"),
        "BERT": os.path.join(args.data_home, 'runs', "{}_distilBert-1.0.run"),
        "QL+BERT": os.path.join(args.data_home, 'runs', "{}_distilBert-0.85.run")
    }
    tokenizer = BertTokenizer.from_pretrained(
        "/ssd2/arthur/TREC2019/data/models/")

    docs_file = "/ssd2/arthur/TREC2019/data/docs/msmarco-docs.tsv"
    lookup_file = os.path.join(args.data_home, "docs", "msmarco-docs-lookup.tsv")
    with open(lookup_file, 'r', encoding='utf-8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [docid, _, offset] in tsvreader:
            docoffset[docid] = int(offset)

    for method, dataset in product(methods.keys(), datasets):
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

        def update(*a):
            pbar.update()

        cpus = args.cpus
        pool = mp.Pool(cpus)
        jobs = []
        with open(triples_file) as inf:
            for index, line in enumerate(inf):
                topic_id, doc_id = line.split("\t")[0].split("-")
                if topic_id != last_topic_id and last_topic_id != -1:
                    if cpus > 1:
                        jobs.append(pool.apply_async(axiom, args=(
                            docs, query, last_topic_id, args.delta, scores, tokenizer), callback=update))
                    else:
                        jobs.append(axiom(docs, query, last_topic_id,
                                          args.delta, scores, tokenizer))
                        pbar.update()
                    docs = {}
                last_topic_id = topic_id
                doc = eval(line.split("\t")[1])
                sep_id = doc.index("[SEP]")
                # document = doc[sep_id + 1:-1]
                document = getcontent(doc_id, docs_file)
                docs[doc_id] = tokenizer.tokenize(document)
                query = doc[1:sep_id]
            pbar.close()
        if cpus > 1:
            results = [p.get() for p in jobs]
        else:
            results = jobs
        pool.close()
        all_candidates = sum(x[0] for x in results)
        satisfies = sum(x[1] for x in results)
        print(method, dataset, all_candidates,
              satisfies, satisfies / all_candidates)
