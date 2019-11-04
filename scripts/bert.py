from msmarco_dataset import MsMarcoDataset
from transformers import DistilBertForSequenceClassification, AdamW, WarmupLinearSchedule
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, average_precision_score, accuracy_score, roc_auc_score
from collections import defaultdict
import subprocess
import os
import random
import torch
import math
import logging
import wandb
torch.multiprocessing.set_start_method('fork', force=True)
logging.getLogger("transformers").setLevel(logging.WARNING)


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
    if "fit_model-{}".format(cut) not in config.force_steps:
        logging.info("Skipping model fit for %s", cut)
        return
    # Set dataset
    train_triples_path = os.path.join(config.data_home, "triples/train-{}.tsv".format(cut))
    dev_triples_path = os.path.join(config.data_home, "triples/dev-{}.tsv".format(cut))
    size = 11 * config.train_queries
    train_dataset = MsMarcoDataset(train_triples_path, config.data_home, invert_label=True, size=size)
    size = 11 * (config.full_dev_queries - config.test_set_size)
    dev_dataset = MsMarcoDataset(dev_triples_path, config.data_home, invert_label=True, size=size)
    
    # Set random seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Set CUDA
    model = DistilBertForSequenceClassification.from_pretrained(config.bert_class)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        visible_gpus = list(range(torch.cuda.device_count()))
        for _id in config.ignore_gpu_ids:
            visible_gpus.remove(_id)
        logging.info("Running with gpus {}".format(visible_gpus))
        device = torch.device("cuda:{}".format(visible_gpus[0]))
        model = torch.nn.DataParallel(model, device_ids=visible_gpus)
        model.to(device)
        train_batch_size = len(visible_gpus) * config.batch_per_device
        logging.info("Effective train batch size of %d", train_batch_size)
    else:
        device = torch.device("cpu")
        model.to(device)
    logging.info("Using device: %s", str(device))
    wandb.watch(model)
    
    data_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=config.number_of_cpus,
        shuffle=True)
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
            inputs = {
                'input_ids': batch[0].to(device),
                'attention_mask': batch[1].to(device),
                'labels': batch[3].to(device)}
            outputs = model(**inputs)
            loss = outputs[0]
            loss = loss.mean()
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                wandb.log({
                    "Train Loss": loss.item(),
                    "Leaning Rate": scheduler.get_lr()[0]})
                    
            global_step += 1
            if global_step % config.train_loss_print == 0:
                logits = outputs[1]
                preds = logits.detach().cpu().numpy()
                logging.info("Train ROC: {}".format(roc_auc_score(out_label_ids, preds[:, 1])))
                preds = np.argmax(preds, axis=1)
                out_label_ids = inputs['labels'].detach().cpu().numpy().flatten()
                logging.info("Train accuracy: {}".format(
                    accuracy_score(out_label_ids, preds)))
                logging.info("Training loss: {}".format(
                    loss.item()))
            
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
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(output_dir)
    output_dir = os.path.join(config.data_home, "models/{}-{}".format(config.bert_class, cut))
    if not os.path.isfile(os.path.join(config.data_home, "models")):
        os.makedirs(os.path.join(config.data_home, "models"))
    # Force overwrite
    if os.path.isdir(output_dir):
        os.rmdir(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    return model


def evaluate(eval_dataset: MsMarcoDataset,
             output_dir: str,
             model,
             device: str,
             global_step: int,
             eval_batchsize=32,
             n_workers=0,
             sample=1.0):
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
    results = {}
    results["ROC Dev"] = roc_auc_score(out_label_ids, preds[:, 1])
    preds = np.argmax(preds, axis=1)
    results["Acuracy Dev"] = accuracy_score(out_label_ids, preds)
    results["F1 Dev"] = f1_score(out_label_ids, preds)
    results["AP Dev"] = average_precision_score(out_label_ids, preds)
    logging.info("***** Eval results *****")
    wandb.log(results)
    for key in sorted(results.keys()):
        logging.info("  %s = %s", key, str(results[key]))


def generate_run_file(split, cut, triples_file=None):
    if triples_file is not None:
        skip_QL = True
    else:
        skip_QL = False
    config = wandb.config
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    preds_out_file = os.path.join(config.data_home, "predictions/")
    size = 100 * config.test_set_size
    if not os.path.isdir(preds_out_file):
        os.makedirs(preds_out_file)
    preds_out_file = os.path.join(preds_out_file, "{}-{}.tensor".format(split, cut))
    if (os.path.isfile(preds_out_file)
            and "bert_transform-{}-{}".format(split, cut) not in config.force_steps
            and triples_file is None):
        _preds = torch.load(preds_out_file)
        logging.info("Loaded predictions from file %s, with %i samples" % (preds_out_file, _preds.shape[0]))
    else:
        if triples_file is not None:
            preds_out_file = os.path.join(config.data_home, "predictions/")
            preds_out_file = os.path.join(preds_out_file, "{}.tensor".format(triples_file))
        model_path = os.path.join(config.data_home, "models/{}-{}".format(config.bert_class, cut))
        model = DistilBertForSequenceClassification.from_pretrained(model_path)

        # Set CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
            visible_gpus = list(range(torch.cuda.device_count()))
            for _id in config.ignore_gpu_ids:
                visible_gpus.remove(_id)
            logging.info("Running with gpus {}".format(visible_gpus))
            device = torch.device("cuda:{}".format(visible_gpus[0]))
            model = torch.nn.DataParallel(model, device_ids=visible_gpus)
            model.to(device)
            batch_size = len(visible_gpus) * config.batch_per_device * 16
            logging.info("Effective train batch size of %d", batch_size)
        else:
            device = torch.device("cpu")
            model.to(device)
        logging.info("Using device: %s", str(device))
        if triples_file is None:
            triples_path = os.path.join(config.data_home, "triples/{}-{}.tsv".format(split, cut))
        else:
            triples_path = triples_file
            size = None
        assert os.path.isfile(triples_path)
        dataset = MsMarcoDataset(triples_path, config.data_home, invert_label=True, size=size)
        batch_size = len(visible_gpus) * config.batch_per_device * 16
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=config.number_of_cpus, shuffle=False)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = 0
        _preds = None

        for index, batch in tqdm(enumerate(data_loader), desc="Batches", total=len(data_loader)):
            model.eval()
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0].to(device),
                    'attention_mask': batch[1].to(device),
                    'labels': batch[3].to(device)}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
                wandb.log({"Loss test": eval_loss})
                nb_eval_steps += 1

                if _preds is None:
                    _preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs["labels"].detach().cpu().numpy().flatten()
                else:
                    batch_predictions = logits.detach().cpu().numpy()
                    _preds = np.append(_preds, batch_predictions, axis=0)
                    batch_ground_truth = inputs['labels'].detach().cpu().numpy().flatten()
                    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy().flatten(), axis=0)
                    try:
                        roc_auc = roc_auc_score(batch_ground_truth, batch_predictions[:, 1])
                    # Only one class on this batch. oops. Report accuracy instead. (very likely when )
                    except:    # noqa E722
                        roc_auc = accuracy_score(batch_ground_truth, np.argmax(batch_predictions, axis=1))
                    results_dict = {
                        "Test accuracy": accuracy_score(batch_ground_truth, np.argmax(batch_predictions, axis=1)),
                        "Test F1 score": f1_score(batch_ground_truth, np.argmax(batch_predictions, axis=1)),
                        "Test ROC score": roc_auc}
                    if index % 100 == 0:
                        print(results_dict)
                    wandb.log(results_dict)

        assert len(_preds) == len(out_label_ids)
        torch.save(_preds, preds_out_file)
    softmax = torch.nn.Softmax(dim=1)
    preds = softmax(torch.as_tensor(_preds))[:, 0].cpu().numpy()

    # Normalize QL scores
    ql_scores = defaultdict(lambda: 0)
    ordered_topics = []
    scores_per_topic = defaultdict(lambda: [])
    ql_run_file = os.path.join(config.data_home, "runs/QL_{}-{}.run".format(split, cut))
    if not skip_QL:
        assert os.path.isfile(ql_run_file), "Could not find runs file at %s" % ql_run_file
        for counter, line in tqdm(enumerate(open(ql_run_file)), desc="reading run file", total=size):
            [topic_id, _, doc_id, _, score, _] = line.split()
            if topic_id not in ordered_topics:
                ordered_topics.append(topic_id)
            scores_per_topic[topic_id].append((doc_id, score))

        for _id in tqdm(scores_per_topic, desc='normalizing QL scores'):
            _scores = np.asarray([float(x[1]) for x in scores_per_topic[_id]])
            normalized_scores = (_scores - np.min(_scores)) / np.ptp(_scores)
            for (did, _), score in zip(scores_per_topic[_id], normalized_scores):
                guid = "{}-{}".format(_id, did)
                ql_scores[guid] = score
        assert len(ql_scores) == size

    # Actually generate new run file
    runs_format = "{} Q0 {} {} {} DISTILBERT_QL\n"
    if skip_QL:
        alphas = [1.0]
    else:
        # This is to get around weird rounding from numpy
        alphas = [float("{:.2f}".format(x)) for x in np.arange(0.0, 1.05, 0.05)]
    best_alpha = 0.0
    best_score = 0.0
    scores = []
    for alpha in alphas:
        beta = 1 - alpha
        if skip_QL:
            out_run_file = os.path.join(config.data_home, "runs/LNC-BERT-{}.run".format(cut))
            with open(triples_path) as inf, open(out_run_file, 'w') as outf:
                for counter, (example, score) in enumerate(zip(inf, preds)):
                    guid = example.split("\t")[0]
                    if len(guid.split("-")) == 3:
                        topic_id, doc_id, lnc = guid.split("-")
                        doc_id = doc_id + "-" + lnc
                    else:
                        topic_id, doc_id = guid.split("-")
                    # Rank doesn't matter. This is just for capturing the scores.
                    outf.write(runs_format.format(topic_id, doc_id, 1, score))
            return

        out_run_file = os.path.join(config.data_home, "runs/{}-{}-alpha_{}.run".format(cut, split, alpha))
        topic_results = []
        last_topic = -1
        with open(ql_run_file) as inf, open(out_run_file, 'w') as outf:
            for counter, (example, score) in enumerate(zip(inf, preds)): # noqa
                topic_id, _, doc_id, _, _, _ = example.split()
                guid = "{}-{}".format(topic_id, doc_id)
                if topic_id != last_topic and len(topic_results) > 0:
                    topic_results.sort(key=lambda x: x['score'], reverse=True)
                    for rank, topic in enumerate(topic_results):
                        outf.write(runs_format.format(topic['topic_id'], topic['doc_id'], rank, topic['score']))
                    topic_results = []
                topic_results.append({'topic_id': topic_id,
                                      'doc_id': doc_id,
                                      'score': alpha * score + beta * ql_scores[guid]})
                last_topic = topic_id

            # dump last topic
            topic_results.sort(key=lambda x: x['score'], reverse=True)
            for rank, topic in enumerate(topic_results):
                outf.write(runs_format.format(topic['topic_id'], topic['doc_id'], rank, topic['score']))
        # Get score for this alpha
        if split == "train":
            qrel_path = os.path.join(config.data_home, "qrels/msmarco-doctrain-qrels.tsv")
        else:
            qrel_path = os.path.join(config.data_home, "qrels/{}.tsv".format(split))
        assert os.path.isfile(qrel_path), "QREL %s not found" % qrel_path
        trec_eval_cmd = "{} -q -c -m {} {} {}".format(config.trec_eval_path, config.metric, qrel_path, out_run_file)
        # logging.info("Running trec_eval with command: %s", trec_eval_cmd)
        output = subprocess.check_output(trec_eval_cmd.split()).decode("utf-8")
        final_metric = float(output.split("\n")[-2].split("\t")[-1])
        scores.append(final_metric)
        if final_metric > best_score:
            best_score = final_metric
            best_alpha = alpha
        logging.info("%s for %s-%s at alpha %f: %f", config.metric, split, cut, alpha, final_metric)
    wandb.run.summary["best alpha"] = best_alpha
    wandb.run.summary["best {}".format(config.metric)] = best_score
    logging.info("Best alpha is %f with score %f" % (best_alpha, best_score))
    wandb.log({"{} by alpha".format(config.metric): wandb.Histogram(scores)})
