from msmarco_dataset import MsMarcoDataset
from pytorch_transformers import BertForNextSentencePrediction, AdamW, WarmupLinearSchedule
import random
import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
import multiprocessing

logger = logging.getLogger(__name__)


def init_optimizer(
        model: BertForNextSentencePrediction,
        n_steps, lr,
        warmup_proportion=0.1):

    param_optimizer = list(model.named_parameters)
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_params = [
        {'params': [p for n, p in param_optimizer if n not in no_decay],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay],
         'weight_decay_rate': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_params, lr=lr, correct_bias=False)
    warmup_steps = n_steps * warmup_proportion
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=n_steps)
    return optimizer, scheduler


def fine_tune(
        dataset: MsMarcoDataset,
        seed=42,
        limit_gpus=None,
        bert_model="bert-base-uncased",
        batch_size=32,
        n_epochs=3,
        learning_rate=5e-5):

    # Seed random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set CUDA
    n_gpu = 0
    if torch.cuda.is_available():
        logging.info("Using CUDA")
        torch.cuda.manual_seed_all(seed)
        if limit_gpus is not None:
            n_gpu = torch.cuda.device_count()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device {}".format(device))
    model = BertForNextSentencePrediction.from_pretrained(bert_model)
    logging.info("Model loaded")
    n_workers = multiprocessing.cpu_count() - 2
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
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



if __name__ == "__main__":
    dataset = MsMarcoDataset(
        "/ssd2/arthur/TREC2019/data/small_sample.tsv", "/ssd2/arthur/TREC2019/data")
    fine_tune(dataset)
