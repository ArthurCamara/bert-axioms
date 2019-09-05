from msmarco_dataset import MsMarcoDataset
from pytorch_transformers import (BertForNextSentencePrediction,
                                  AdamW, WarmupLinearSchedule, DistilBertForSequenceClassification)
import random
import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
from tor
from tqdm.auto import tqdm
import multiprocessing
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
from args_parser import getArgs
import os
import sys
import math

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

    is_distill = "distilbert" in args.bert_model
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
    if is_distill:
        model = DistilBertForSequenceClassification.from_pretrained(
            args.bert_model)
    else:
        model = BertForNextSentencePrediction.from_pretrained(args.bert_model)
    logging.info("Model loaded")
    if n_gpu > 0:
        gpu_ids = list(range(n_gpu))
        # Ignore any GPU? (usefull if there is more users on current machine, already using a GPU)
        if args.ignore_gpu_ids is not None:
            for _id in args.ignore_gpu_ids:
                if _id in gpu_ids:
                    gpu_ids.remove(_id)
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        print("Using device IDs {}".format(str(gpu_ids)))
    if n_gpu > 0 and args.ignore_gpu_ids is not None:
        device_0 = torch.device("cuda:{}" .format(gpu_ids[0]))
        model.to(device_0)
    else:
        model.to(device)
    print(args.train_batch_size)
    data_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=args.n_workers)
    num_train_optimization_steps = len(
        data_loader) // args.gradient_accumulation_steps * args.n_epochs
    optimizer, scheduler = init_optimizer(
        model, num_train_optimization_steps, args.learning_rate)

    logger.info("******Started trainning******")
    logger.info("   Num samples = %d", len(train_dataset))
    logger.info("   Training model %s", args.bert_model)
    logger.info("   Num Epochs = %d", args.n_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.train_batch_size // n_gpu)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("   Total optmization steps %d", num_train_optimization_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    for _ in tqdm(range(args.n_epochs), desc="Epochs"):
        for step, batch in tqdm(enumerate(data_loader), desc="Batches", total=len(data_loader)):
            model.train()
            if not is_distill:
                inputs = {'input_ids': batch[0].to(device),
                          'attention_mask': batch[1].to(device),
                          'token_type_ids': batch[2].to(device),
                          'next_sentence_label': batch[3].to(device)}
            else:
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}

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
            if global_step % args.train_loss_print == 0:
                # print(loss.item())
                logits = outputs[1]
                preds = logits.detach().cpu().numpy()
                preds = np.argmax(preds, axis=1)
                if 'next_sentence_label' in inputs:
                    out_label_ids = inputs['next_sentence_label'].detach().cpu().numpy().flatten()
                else:
                    out_label_ids = inputs['labels'].detach().cpu().numpy().flatten()
                print("Train accuracy: {}".format(
                    accuracy_score(out_label_ids, preds)))
                # print("Train F1: {}".format(
                    # f1_score(out_label_ids, preds)))
                print("Training loss: {}".format((tr_loss - logging_loss)/args.train_loss_print))
                logging_loss = tr_loss

            if global_step % args.eval_steps == 0:
                _ = evaluate(dev_dataset,
                             args.data_dir,
                             model,
                             device,
                             args.data_dir,
                             eval_batchsize=args.eval_batch_size,
                             n_workers=args.n_workers,
                             sample=args.eval_sample)
                print("\tlr: \t{}".format(scheduler.get_lr()[0]))
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
             model,
             device: str,
             eval_output_dir: str,
             task_name="msmarco",
             prefix="",
             eval_batchsize=32,
             n_workers=0,
             sample=1.0):
    results = {}
    starting_index = 0
    max_feasible_index = len(eval_dataset) - \
        math.floor(len(eval_dataset) * sample)
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
            if isinstance(model.module, DistilBertForSequenceClassification):
                inputs = {'input_ids': batch[0].to(device),
                          'attention_mask': batch[1].to(device),
                          'labels': batch[3].to(device)}
            else:
                inputs = {'input_ids': batch[0].to(device),
                          'attention_mask': batch[1].to(device),
                          'token_type_ids': batch[2].to(device),
                          'next_sentence_label': batch[3].to(device)}

        outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            if 'next_sentence_label' in inputs:
                out_label_ids = inputs['next_sentence_label'].detach(
                ).cpu().numpy().flatten()
            else:
                out_label_ids = inputs['labels'].detach(
                ).cpu().numpy().flatten()
        else:
            batch_predictions = logits.detach().cpu().numpy()
            preds = np.append(preds, batch_predictions, axis=0)
            if 'next_sentence_label' in inputs:
                out_label_ids = np.append(
                    out_label_ids, inputs['next_sentence_label'].detach().cpu().numpy().flatten(), axis=0)
            else:
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy().flatten(), axis=0)
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
            "--train_file", data_dir + "/train-triples.top100",
            "--dev_file", data_dir + "/dev-triples.top100",
            "--eval_batch_size", "32",
            "--eval_steps", "200",
            "--bert_model", "distilbert-base-uncased",
            "--train_batch_size", "128",
            "--eval_sample", "0.1", 
            "--train_loss_print", "20",
            "--ignore_gpu_ids", "6", 
        ]

    args = getArgs(argv)

    is_distil = "distilbert" in args.bert_model
    logging.basicConfig(level=logging.getLevelName(args.log_level))
    train_dataset = MsMarcoDataset(
        args.train_file, args.data_dir, distil=is_distil, invert_label=True)
    dev_dataset = MsMarcoDataset(
        args.dev_file, args.data_dir, distil=is_distil, invert_label=True)
    fine_tune(train_dataset, dev_dataset, args)
