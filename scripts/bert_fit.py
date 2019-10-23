from msmarco_dataset import MsMarcoDataset
from transformers import DistilBertForSequenceClassification, AdamW, WarmupLinearSchedule
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
import os
import random
import torch
import math
import logging
import wandb


def init_optimizer(
        model: DistilBertForSequenceClassification,
        n_steps: int,
        lr: float,
        warmup_proportion: float = 0.1,
        weight_decay: float = 0.0):

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


def fit_bert(config, cut):

    # Set dataset
    train_triples_path = os.path.join(config.data_home, "triples/train-{}.tsv".format(cut))
    dev_triples_path = os.path.join(config.data_home, "triples/dev-{}.tsv".format(cut))
    size = 11 * config.train_queries
    train_dataset = MsMarcoDataset(train_triples_path, config.data_home, invert_label=True, size=size)
    size = 11 * (config.full_dev_queries - config.test_set_size)
    dev_dataset = MsMarcoDataset(dev_triples_path, config.data_home, distil=True, invert_label=True, size=size)
    
    # Set random seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Set CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        visible_gpus = list(range(torch.cuda.device_count()))
        for _id in config.ignore_gpu_ids:
            visible_gpus.remove(_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in visible_gpus)
        logging.info("Running with gpus {}".format(visible_gpus))
    else:
        device = torch.device("cpu")
    logging.info("Using device: %s", str(device))
    model = DistilBertForSequenceClassification.from_pretrained(config.bert_class)
    model.to(device)
    wandb.watch(model)
    data_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.number_of_cpus)
    num_train_optimization_steps = len(data_loader) * config.n_epochs
    optimizer, scheduler = init_optimizer(model, num_train_optimization_steps, config.learning_rate)
    logging.info("******Started trainning******")
    logging.info("   Total optmization steps %d", num_train_optimization_steps)

    global_step = 0
    tr_loss = logging_loss = 0.0
    model.zero_grad()
    for _ in tqdm(range(config.n_epochs), desc="Epochs"):
        for step, batch in tqdm(enumerate(data_loader), desc="Batches", total=len(data_loader)):
            model.train()
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]
            loss = loss.mean()
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.setp()
                scheduler.step()
                model.zero_grad()
                wandb.log({
                    "Train Loss": (tr_loss - logging_loss) / config.train_loss_print,
                    "Leaning Rate": scheduler.get_lr()[0]}, step=global_step)
            global_step += 1
            if global_step % config.train_loss_print == 0:
                logits = outputs[1]
                preds = logits.detach().cpu().numpy()
                preds = np.argmax(preds, axis=1)
                out_label_ids = inputs['labels'].detach().cpu().numpy().flatten()
                wandb.log({
                    "Train Accuracy": accuracy_score(out_label_ids, preds)
                }, step=global_step)
                logging.info("Train accuracy: {}".format(
                    accuracy_score(out_label_ids, preds)))
                logging.info("Training loss: {}".format(
                    (tr_loss - logging_loss) / config.train_loss_print))
                logging_loss = tr_loss
            
            if global_step % config.eval_steps == 0:
                evaluate(dev_dataset,
                         config.data_home,
                         model,
                         device,
                         global_step,
                         eval_batchsize=config.eval_batchsize,
                         n_workers=config.number_of_cpus,
                         sample=config.eval_sample)
                # Save intermediate model
                output_dir = os.path.join(config.data_home, "checkpoints/checkpoint-{}".format(global_step))
                logging.info("Saving model checkpoint to %s", output_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model.save_pretrained(output_dir)
                wandb.save(output_dir)


def evaluate(eval_dataset: MsMarcoDataset,
             output_dir: str,
             model,
             device: str,
             global_step: int,
             eval_batchsize=32,
             n_workers=0,
             sample=1.0):
    results = {}
    starting_index = 0
    max_feasible_index = len(eval_dataset) - math.floor(len(eval_dataset) * sample)
    if max_feasible_index > 0:
        starting_index = random.choice(range(max_feasible_index))
    ending_index = starting_index + math.floor(len(eval_dataset) * sample)
    eval_dataloader = DataLoader(
        eval_dataset[starting_index:ending_index], batch_size=eval_batchsize, shuffle=False, num_workers=n_workers)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Eval batch"):
        model.eval()
        with torch.no_grad():
            inputs = {'input_ids': batch[0].to(device),
                      'attention_mask': batch[1].to(device),
                      'labels': batch[3].to(device)}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy().flatten()
            else:
                batch_predictions = logits.detach().cpu().numpy()
                preds = np.append(preds, batch_predictions, axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy().flatten(), axis=0)
        eval_loss = eval_loss / nb_eval_steps
    results["Acuracy Dev"] = accuracy_score(out_label_ids, preds)
    results["F1 Dev"] = f1_score(out_label_ids, preds)
    results["AP Dev"] = average_precision_score(out_label_ids, preds)
    logging.info("***** Eval results *****")
    wandb.log(results, step=global_step)
    for key in sorted(results.keys()):
        logging.info("  %s = %s", key, str(results[key]))

