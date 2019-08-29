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
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
from args_parser import getArgs
import os
import sys
multiprocessing.set_start_method('spawn', True)

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
        args):
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set CUDA
    n_gpu = 0
    if torch.cuda.is_available():
        logging.info("Using CUDA")
        torch.cuda.manual_seed_all(args.seed)
        # We DO NOT want to limit the number of GPUs to be used
        if args.limit_gpus < 0:
            args.limit_gpus = torch.cuda.device_count()
            n_gpu = torch.cuda.device_count()
        # We WANT to limit the number of GPUs to be used
        else:
            n_gpu = min(torch.cuda.device_count(), args.limit_gpus)

    device = torch.device("cuda" if (torch.cuda.is_available()
                                     and n_gpu > 0 and args.limit_gpus != 1) else "cpu")
    logging.info("Using device {}".format(device))
    model = BertForNextSentencePrediction.from_pretrained(args.bert_model)
    logging.info("Model loaded")
    if args.n_workers is None:
        args.n_workers = multiprocessing.cpu_count() - 2

    if n_gpu > 0:
        gpu_ids = list(range(n_gpu))
        # Ignore any GPU? (usefull if there is more users on current machine, already using a GPU)
        if args.ignore_gpu_ids is not None:
            for _id in args.ignore_gpu_ids:
                if _id in gpu_ids:
                    gpu_ids.remove(_id)
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        print("Using device IDs {}".format(str(gpu_ids)))
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, len(gpu_ids))
    if n_gpu > 0:
        device_0 = torch.device("cuda:{}" .format(gpu_ids[0]))
        model.to(device_0)
    else:
        model.to(device)
    data_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.n_workers)
    num_train_optimization_steps = len(
        data_loader) // args.gradient_accumulation_steps * args.n_epochs
    optimizer, scheduler = init_optimizer(
        model, num_train_optimization_steps, args.learning_rate)

    logger.info("******Started trainning******")
    logger.info("   Num samples = %d", len(train_dataset))
    logger.info("   Num Epochs = %d", args.n_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("   Total optmization steps %d", num_train_optimization_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    for _ in tqdm(range(args.n_epochs), desc="Epochs"):
        for step, batch in tqdm(enumerate(data_loader), desc="Batches"):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'next_sentence_label': batch[3]}
            outputs = model(**inputs)

            loss = outputs[0]
            if n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            global_step += 1
            if global_step % args.eval_steps == 0:
                print("Training loss: {}".format(tr_loss))
                _ = evaluate(dev_dataset,
                             args.data_dir,
                             model,
                             device,
                             args.data_dir,
                             eval_batchsize=args.eval_batch_size,
                             n_workers=args.n_workers)
                print("\tlr: \t{}".format(scheduler.get_lr()[0]))
                print("\tLoss:\t{}".format(
                    tr_loss - logging_loss / args.eval_steps))
                logging_loss = tr_loss
                # Save model
                output_dir = os.path.join(
                    args.data_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(
                    model, 'module') else model
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
             eval_batchsize=128,
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
            ).cpu().numpy().flatten()
        else:
            batch_predictions = logits.detach().cpu().numpy()
            preds = np.append(preds, batch_predictions, axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs['next_sentence_label'].detach().cpu().numpy().flatten(), axis=0)
        eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    assert len(preds) == len(out_label_ids)
    result = {}
    result["acc"] = accuracy_score(out_label_ids, preds)
    result["f1"] = f1_score(out_label_ids, preds)
    result["AP"] = average_precision_score(out_label_ids, preds)
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
    data_dir = "/ssd2/arthur/TREC2019/data"
    if len(sys.argv) > 3:
        argv = sys.argv[1:]
    else:
        argv = [
            "--data_dir", data_dir,
            "--train_file", data_dir + "/train-triples.0",
            "--dev_file", data_dir + "/dev-triples.0",
            "--per_gpu_train_batch_size", "8",
            "--train_batch_size", "32",
            "--gradient_accumulation_steps", "10",
            "--ignore_gpu_ids", "0,1,5,7",
            "--limit_gpus", "-1",
            "--eval_steps", "10"
        ]
    args = getArgs(argv)
    logging.basicConfig(level=logging.getLevelName(args.log_level))
    train_dataset = MsMarcoDataset(args.train_file, args.data_dir)
    dev_dataset = MsMarcoDataset(args.dev_file, args.data_dir)
    fine_tune(train_dataset, dev_dataset, args)
