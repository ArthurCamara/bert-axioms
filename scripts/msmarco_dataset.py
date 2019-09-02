import os
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer, DistilBertTokenizer
from tqdm.auto import tqdm
import logging
import pickle


class MsMarcoDataset(Dataset):
    """MsMarco preprocessing Dataset"""

    def __init__(
            self,
            tsv_file: str,
            data_dir: str,
            max_seq_len=512,
            size=None,
            transform=None,
            invert_label=True,
            xlnet=False,
            distil=False):
        """
        Args:
            tsv_file (string): TSV file with triples, generated by text_to_tokens_slurm.py, formatted like
                "qid-did    bert-formatted-tokens   label"
            data_dir (string): Directory with the rest of the data, where we can write to
            size (int, optional): Number of lines to process. If None, will run wc -l first.
            transform (callable, optional): Transformations to be performed on the data
            invert_label: Must be True if you are planning to use BertForNextSentencePrediction
            XLNet: If you plan on using XLNet insted of BERT, the [CLS] token must be changed.
                Set this to True if you plan to. (NOT YET IMPLEMENTED)
            Distil: Use distlBERT?
        """
        logging.info("Loading dataset from %s", tsv_file)
        self.tsv_path = tsv_file
        self.data_dir = data_dir
        self.transform = transform
        self.max_seq_len = max_seq_len
        self.xlnet = xlnet
        self.distil = distil
        if tsv_file.endswith("train-triples.top100"):
            size = 36127662
        elif tsv_file.endswith("dev-triples.top100"):
            size = 519296
        print(size)
        if size is None:
            with open(tsv_file) as f:
                for i, _ in tqdm(enumerate(f), desc="Counting lines on file..."):
                    pass
            self.size = i + 1
        else:
            self.size = size

        self.offset_dict, self.index_dict = self.load_offset_dict()
        assert os.path.isdir(os.path.join(self.data_dir, "models"))
        if distil:
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        else:
            self.tokenizer = BertTokenizer.from_pretrained(
                os.path.join(self.data_dir, "models"))
        if invert_label:
            self.label_map = {label: i for i, label in enumerate(["1", "0"])}
        else:
            self.label_map = {label: i for i, label in enumerate(["0", "1"])}

    def load_offset_dict(self):
        offset_dict = {}
        index_dict = {}
        cache_file = self.tsv_path + ".offset"
        index_file = self.tsv_path + ".index"
        if os.path.isfile(cache_file) and os.path.isfile(index_file):
            offset_dict = pickle.load(open(cache_file, 'rb'))
            index_dict = pickle.load(open(index_file, 'rb'))
            return offset_dict, index_dict

        with open(self.tsv_path, encoding="utf-8") as f:

            pbar = tqdm(total=self.size, desc="Computing offset dictionary")
            location = f.tell()
            line = f.readline()
            pbar.update()
            idx = 0
            while line:
                [did, _, _] = line.split("\t")
                offset_dict[did] = location
                index_dict[idx] = did
                location = f.tell()
                line = f.readline()
                pbar.update()
                idx += 1
        with open(cache_file, 'wb') as f:
            pickle.dump(offset_dict, f)
        with open(index_file, 'wb') as f:
            pickle.dump(index_dict, f)

        return offset_dict, index_dict

    def __getitem__(self, did):
        if isinstance(did, str):
            offset = self.offset_dict[did]
        elif isinstance(did, int):
            offset = self.offset_dict[self.index_dict[did]]
        else:
            raise NotImplementedError(
                "can only fetch integer or string indexes")
        with open(self.tsv_path) as f:
            f.seek(offset)
            line = f.readline()
        return self.text_to_features(line)

    def __len__(self):
        return self.size

    def text_to_features(self, sample):

        line = sample.strip().split("\t")
        tokens = eval(line[1])
        label = line[-1]

        sep_index = tokens.index("[SEP]")

        tokens_a = tokens[1:sep_index]
        tokens_b = tokens[sep_index + 1:-1]

        segment_ids = [0] * (len(tokens_a) + 2)
        segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (self.max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq_len, "input_id"
        assert len(input_mask) == self.max_seq_len, "input_mask"
        assert len(segment_ids) == self.max_seq_len, "segment_ids"

        label_id = self.label_map[label]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(input_mask, dtype=torch.long),
            torch.tensor(segment_ids, dtype=torch.long),
            torch.tensor([label_id], dtype=torch.long))


if __name__ == "__main__":
    dataset = MsMarcoDataset(
        "/ssd2/arthur/insy/msmarco/data/dev-triples.0", "/ssd2/arthur/TREC2019/data")
    data_loader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4)
