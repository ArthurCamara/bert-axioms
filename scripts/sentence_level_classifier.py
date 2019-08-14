from msmarco_dataset import MsMarcoDataset
from pytorch_transformers import BertForNextSentencePrediction, AdamW, WarmupLinearSchedule
import random
import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yappi
import multiprocessing

logger = logging.getLogger(__name__)


def init_optimizer(
        model: BertForNextSentencePrediction,
        n_steps, lr,
        warmup_proportion=0.1,
        weight_decay=0.0):

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    warmup_steps = n_steps * warmup_proportion
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=n_steps)
    return optimizer, scheduler


def fine_tune(
        dataset: MsMarcoDataset,
        seed: int = 42,
        limit_gpus: int = -1,
        bert_model="bert-base-uncased",
        batch_size=32,
        n_epochs=3,
        learning_rate=5e-5,
        n_workers=None):

    # Seed random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set CUDA
    n_gpu = 0
    if torch.cuda.is_available():
        logging.info("Using CUDA")
        torch.cuda.manual_seed_all(seed)
        if limit_gpus > -1:
            n_gpu = torch.cuda.device_count()

    device = torch.device("cuda" if torch.cuda.is_available()
                          and limit_gpus > 0 else "cpu")
    logging.info("Using device {}".format(device))
    model = BertForNextSentencePrediction.from_pretrained(bert_model)
    logging.info("Model loaded")
    if n_workers is None:
        n_workers = multiprocessing.cpu_count() - 2
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    num_train_optimization_steps = len(data_loader) // n_epochs

    optimizer, scheduler = init_optimizer(
        model, num_train_optimization_steps, learning_rate)

    if n_gpu > 1:
        model = torch.nn.DataParallel(n_gpu)
    model.to(device)
    logger.info("******Started trainning******")
    logger.info("   Num samples = %d", len(dataset))
    logger.info("   Num Epochs = %d", n_epochs)
    logger.info("   Batch size = %d", batch_size)
    logger.info("   Total optmization steps %d", num_train_optimization_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    for _ in tqdm(range(n_epochs), desc="Epochs"):
        for batch in tqdm(data_loader, desc="Batches"):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'next_sentence_label': batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]


if __name__ == "__main__":
    dataset = MsMarcoDataset(
        "/ssd2/arthur/TREC2019/data/small_sample.tsv", "/ssd2/arthur/TREC2019/data")
    yappi.set_clock_type("wall")
    yappi.start()
    fine_tune(dataset)
    yappi.get_func_stats().print_all()
    yappi.get_thread_stats().print_all()
