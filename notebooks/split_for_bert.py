import gzip
from tqdm import tqdm
import spacy
import multiprocessing
from joblib import Parallel, delayed

max_cpus = 20
n_splits = 1000
inputs = tqdm(range(n_splits))

def run_one_file(n):
    with open(f"../data/splitted-tsv/msmarco-docs.tsv.{n:03}", encoding='utf-8') as f, open(f"../data/one-per-line/one-sentence-per-line.{n:03}", 'w', encoding='utf-8') as outf:
        for line in f:
            try:
                document = nlp(line.decode("utf-8").split("\t")[3])
                title = line.decode("utf-8").split("\t")[2]
            except:
                try:
                    document = nlp(line.split("\t")[3])
                    title = line.split("\t")[2]
                except:
                    continue
                
            lines_to_write = [title] + [str(sent) for sent in document.sents]
            for l in lines_to_write:
                outf.write(l+"\n")
            outf.write("\n")

nlp = spacy.load("en_core_web_sm")
Parallel(n_jobs=max_cpus)(delayed(run_one_file)(i) for i in inputs)

