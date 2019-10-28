import torch
from torch.utils.data import DataLoader
from msmarco_dataset import MsMarcoDataset
from transformers import DistilBertForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import logging
import wandb
import random
import numpy as np
import os
import subprocess
from tqdm.auto import tqdm
from collections import defaultdict


def generate_run_file(split, cut):
    config = wandb.config
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    preds_out_file = os.path.join(config.data_home, "predictions/")
    size = 100 * config.test_set_size
    if not os.path.isdir(preds_out_file):
        os.makedirs(preds_out_file)
    preds_out_file = os.path.join(preds_out_file, "{}-{}.tensor".format(split, cut))
    if os.path.isfile(preds_out_file) and "bert_transform-{}-{}".format(split, cut) not in config.force_steps:
        _preds = torch.load(preds_out_file)
        logging.info("Loaded predictions from file %s, with %i samples" % (preds_out_file, _preds.shape[0]))
    else:

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
        triples_path = os.path.join(config.data_home, "triples/{}-{}.tsv".format(split, cut))
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
                    except:  # Only one class on this batch. oops. Report accuracy instead. (very likely when )
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
    # This is to get around weird rounding from numpy
    alphas = [float("{:.2f}".format(x)) for x in np.arange(0.0, 1.05, 0.05)]
    best_alpha = 0.0
    best_score = 0.0
    scores = []
    for alpha in alphas:
        beta = 1 - alpha
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