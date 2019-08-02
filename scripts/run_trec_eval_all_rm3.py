import subprocess
import os
from tqdm import tqdm

data_path = "/ssd2/arthur/insy/msmarco/data/results/dev"
trec_path = "/ssd2/arthur/trec_eval/trec_eval"
qrel_path = "/ssd2/arthur/TREC2019/data/msmarco-docdev-qrels.tsv"
cmd = "{} -q -c {} {}"

all_files = [os.path.join(data_path, x) for x in os.listdir(data_path)]


best_map = 0.0
for file in tqdm(all_files):
    print(file)
    results = subprocess.check_output(cmd.format(trec_path, qrel_path, file).split()).decode('utf-8')
    map = float(results.split("\n")[-26].split("\t")[-1])
    if map > best_map:
        best_map = map
        print("map of {} found for file {}".format(map, file))
        best_file = file    
