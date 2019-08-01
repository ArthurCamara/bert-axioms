#!/usr/bin/python3
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm


def dump_buffers(buffers, n_splits, output_dir):
    for i in range(n_splits):
        with open(output_dir/f"split_{i}.txt", "a+") as f:
            f.writelines(buffers[i])
            buffers[i] = []


def main():
    parser = ArgumentParser()
    parser.add_argument("--n_splits", required=True, type=int)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--input_file", required=True, type=Path)
    parser.add_argument("--buffer_size", type=int, default=1000)
    
    args = parser.parse_args()
    
    buffers = [[] for _ in range(args.n_splits)]

    with args.input_file.open() as f:
        doc_lines = []
        docs_processed = 0
        for l in tqdm(f, desc="Loading Dataset", 
                unit=" lines", total=181500860):
            line = l.strip()
            if line=="</DOC>":
                doc_lines.append(line)
                full_doc = "\n".join(doc_lines)
                target_split = docs_processed % args.n_splits
                buffers[target_split].append(full_doc)
                docs_processed +=1
                if len(buffers[-1]) == args.buffer_size:
                    dump_buffers(buffers, args.n_splits, args.output_dir)
                doc_lines = []
            else:
                doc_lines.append(line)



if __name__=="__main__":
    main()
