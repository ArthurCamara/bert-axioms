from msmarco_dataset import MsMarcoDataset
from pytorch_transformers import (BertForNextSentencePrediction,
                                  AdamW, WarmupLinearSchedule)
import random
import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import multiprocessing
from sklearn.metrics import f1_score
from args_parser import getArgs
import os


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
        train_dataset: MsMarcoDataset,
        dev_dataset: MsMarcoDataset,
        data_dir,
        seed: int = 42,
        limit_gpus: int = -1,
        bert_model="bert-base-uncased",
        batch_size=32,
        eval_batch_size=128,
        n_epochs=3,
        learning_rate=5e-5,
        n_workers=None):
    # Set random seeds
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
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    num_train_optimization_steps = len(data_loader) // n_epochs

    optimizer, scheduler = init_optimizer(
        model, num_train_optimization_steps, learning_rate)

    if n_gpu > 1:
        model = torch.nn.DataParallel(n_gpu)
    model.to(device)
    logger.info("******Started trainning******")
    logger.info("   Num samples = %d", len(train_dataset))
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
            print(loss)
            if n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            tr_loss += loss.item()
            scheduler.step()
            optimizer.step()
            model.zero_grad()
            global_step += 1
            if global_step % 50 == 0:
                results = evaluate(dev_dataset,
                                   data_dir,
                                   model,
                                   device,
                                   data_dir,
                                   eval_batchsize=eval_batch_size,
                                   n_workers=n_workers)
                for key, value in results.items():
                    print('\teval_{}:\t{}'.format(key, value))
                print("\tlr: \t{}".format(scheduler.get_lr()[0]))
                print("\tLoss:\t{}".format(tr_loss - logging_loss) / 50)
                logging_loss = tr_loss
                # Save model
                output_dir = os.path.join(data_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)
    return global_step, tr_loss / global_step


def evaluate(eval_dataset: MsMarcoDataset,
             output_dir: str,
             model: BertForNextSentencePrediction,
             device: str,
             eval_output_dir: str,
             task_name="msmarco",
             prefix="",
             eval_batchsize=32,
             n_workers=2):
    results = {}
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=eval_batchsize, shuffle=False, num_workers=n_workers)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'next_sentence_label': batch[3]
            }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['next_sentence_label'].detach(
            ).cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs['next_sentence_label'].detach().cpu().numpy(), axis=0)
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        assert len(preds) == len(out_label_ids)
        result = {}
        result["acc"] = (preds == out_label_ids).mean()
        result["f1"] = f1_score(y_true=out_label_ids, y_pred=preds)
        result["acc_and_f1"] = (result["acc"] + result["f1"]) / 2
        results.update(result)
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


if __name__ == "__main__":
    argv = ["--data_dir", "/ssd2/arthur/TREC2019/data",
            "--train_file", "/ssd2/arthur/insy/msmarco/data/train-triples.1",
            "--dev_file", "/ssd2/arthur/insy/msmarco/data/dev-triples.1",
            "--bert-model", "bert-base-uncased"]
args = getArgs(argv)
train_dataset = MsMarcoDataset(args.train_file, args.data_dir)
dev_dataset = MsMarcoDataset(args.dev_file, args.data_dir)
fine_tune(train_dataset, dev_dataset, args.data_dir, n_workers=0)
