import csv
import logging
import os
import torch
from torch.utils.data import (
    DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset)
import pickle
import sys
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_transformers.tokenization_bert import BertTokenizer
from tqdm.autonotebook import tqdm
import multiprocessing
multiprocessing.set_start_method('spawn', True)


logger = logging.getLogger(__name__)

csv.field_size_limit(sys.maxsize)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, sample=False, total=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in tqdm(reader, desc="Reading input tsv", total=total):
                if sample and len(lines) > 1000:
                    return lines
                lines.append(line)
            return lines


class MsMarcoProcessor(DataProcessor):
    """Processor class for the MsMarco dataset (triples version)."""

    def get_train_examples(self, data_dir, sample=False, total=None):
        return self._create_examples(self._read_tsv(
            os.path.join(data_dir, "train-samples.tsv"), sample=sample, total=total), "train", total=total)

    def get_dev_examples(self, data_dir, total=None):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "bm25_bert_docs.tsv"), total=total), "dev", total=total)

    def get_labels(self):
        return ['0', '1']

    def _create_examples(self, lines, set_type, total=None):
        examples = []
        for (i, line) in tqdm(enumerate(lines), desc="creating examples...", total=total):
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def convert_sample_to_feature(example, label_map, max_seq_length, tokenizer, output_mode):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length, "input_id"
        assert len(input_mask) == max_seq_length, "input_mask"
        assert len(segment_ids) == max_seq_length, "segment_ids"

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)
        return InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             label_id=label_id)


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode, total=None):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    returns = []
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pbar = tqdm(total=total)

    def update(*a):
        pbar.update()

    for example in examples:
        returns.append(pool.apply_async(convert_sample_to_feature, args=(
            example, label_map, max_seq_length, tokenizer, output_mode), callback=update))
    pool.close()
    pool.join()
    features = [job.get() for job in returns]
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def load_dataset(
        task_name, model_name, max_seq_length,
        data_dir, tokenizer, batch_size, eval=False, sample=False,
        return_examples=False, force_reload=False,
        expected_len=None, load_only=False):

    if eval:
        cached_features_file = os.path.join(data_dir, 'dev_{}_{}_{}'.format(
            list(filter(None, model_name.split("/"))).pop(), str(max_seq_length), task_name))
        print(cached_features_file)
    else:
        cached_features_file = os.path.join(data_dir, 'train_{}_{}_{}'.format(
            list(filter(None, model_name.split("/"))).pop(), str(max_seq_length), task_name))

    processor = MsMarcoProcessor()
    if eval:
        examples = processor.get_dev_examples(data_dir, total=expected_len)
    else:
        examples = processor.get_train_examples(
            data_dir, sample, total=expected_len)

    # if cached file already exists, do not reload it
    if os.path.isfile(cached_features_file) and not force_reload:
        with open(cached_features_file, 'rb') as reader:
            features = pickle.load(reader)
    else:
        features = convert_examples_to_features(
            examples, processor.get_labels(),
            max_seq_length, tokenizer, 'classification', total=expected_len)
        logger.info("Saving features into cached file %s",
                    cached_features_file)
        with open(cached_features_file, 'wb') as writer:
            pickle.dump(features, writer)

    assert len(features) == len(examples)
    if load_only:
        return
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask,
                         all_segment_ids, all_label_ids)

    if not eval:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)

    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    if return_examples:
        return dataloader, examples
    return dataloader


def init_optimizer(model, n_train_steps, learning_rate, warmup_proportion):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    num_train_steps = n_train_steps
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay],
            'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay],
            'weight_decay_rate': 0.0},
    ]
    optimizer = BertAdam(
        optimizer_grouped_parameters, lr=learning_rate,
        warmup=warmup_proportion,
        t_total=num_train_steps)
    return optimizer


class MsMarcoDataset(Dataset):
    """MsMarco preprocessing Dataset"""

    def __init__(self, tsv_file, data_dir, max_seq_len=512, size=None, transform=None):
        """
        Args:
            tsv_file (string): TSV file with triples, generated by TREC_run_to_BERT.py, formatted like
                "qid-did    query   document    label"
            data_dir (string): Directory with the rest of the data, where we can write to
            size (int, optional): Number of lines to process. If None, will run wc -l first.
            transform (callable, optional): Transformations to be performed on the data
        """
        self.tsv_path = tsv_file
        self.data_dir = data_dir
        self.transform = transform
        self.max_seq_len = max_seq_len
        if size is None:
            with open(tsv_file) as f:
                for i, _ in enumerate(f):
                    pass
            self.size = i + 1
        else:
            self.size = size

        self.offset_dict, self.index_dict = self.load_offset_dict()
        assert os.path.isdir(os.path.join(self.data_dir, "models"))
        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(self.data_dir, "models"))
        self.label_map = {label: i for i, label in enumerate(["0", "1"])}

    def load_offset_dict(self):
        offset_dict = {}
        index_dict = {}
        with open(self.tsv_path, encoding="utf-8") as f:
            pbar = tqdm(total=self.size, desc="Computing offset dictionary")
            location = f.tell()
            line = f.readline()
            pbar.update()
            idx = 0
            while line:
                [did, query, document, label] = line.split("\t")
                offset_dict[did] = location
                index_dict[idx] = did
                location = f.tell()
                line = f.readline()
                pbar.update()
                idx += 1

        return offset_dict, index_dict

    def __getitem__(self, did):
        if isinstance(did, str):
            offset = self.offset_dict[did]
        else:
            offset = self.offset_dict[self.index_dict[did]]
        with open(self.tsv_path) as f:
            f.seek(offset)
            line = f.readline()
        return self.text_to_features(line)

    def __len__(self):
        return self.size

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def text_to_features(self, sample):

        line = sample.split("/t")
        text_a = line[1]
        text_b = line[2]
        label = line[-1]
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = self.tokenizer.tokenize(text_b)
        _truncate_seq_pair(tokens_a, tokens_b, self.max_seq_len - 3)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        segment_ids = [0] * len(tokens) + [1] * len(tokens_b) + 1
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq_length, "input_id"
        assert len(input_mask) == self.max_seq_length, "input_mask"
        assert len(segment_ids) == self.max_seq_length, "segment_ids"

        label_id = self.label_map[label]

        return InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id)


if __name__ == "__main__":
    dataset = MsMarcoDataset(
        "/ssd2/arthur/TREC2019/data/tiny_sample.tsv", "/ssd2/arthur/TREC2019/data")
    assert dataset[1] == dataset["174249-D2204704"]
