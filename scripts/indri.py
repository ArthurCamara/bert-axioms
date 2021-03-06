import subprocess
import os
import logging
import warnings
import re
warnings.filterwarnings("ignore")
import wandb # noqa


def generate_index(config, full=True):
    # Create buildindex.param file
    param_file_format = """
    <parameter>
        <corpus>
            <path>{}</path>
            <class>trectext</class>
        </corpus>
        <index>{}</index>
        <memory>32G</memory>
        <threads>{}</threads>
    </parameter>"""
    # Check if index folder exists
    if not os.path.isdir(os.path.join(config.data_home, "indexes")):
        os.mkdir(os.path.join(config.data_home, "indexes"))
    if not os.path.isdir(os.path.join(config.data_home, "indri_params")):
        os.mkdir(os.path.join(config.data_home, "indri_params"))

    if full:
        data_dir = os.path.join(os.path.join(config.data_home, "docs/msmarco-docs.tokenized.trec"))
        index_path = os.path.join(os.path.join(config.data_home, "indexes/full-tokenized"))
        param_file_format = param_file_format.format(data_dir, index_path, config.number_of_cpus)
        param_file = os.path.join(config.data_home, "indri_params/indexing_full.param")
    else:
        data_dir = os.path.join(os.path.join(config.data_home, "docs/msmarco-docs.tokenized.cut.trec"))
        index_path = os.path.join(os.path.join(config.data_home, "indexes/cut-tokenized"))
        param_file_format = param_file_format.format(data_dir, index_path, config.number_of_cpus)
        param_file = os.path.join(config.data_home, "indri_params/indexing_cut.param")
    if full:
        cut = "full"
    else:
        cut = "cut"
    if os.path.isdir(index_path) and "index_{}".format(cut) not in config.force_steps:
        wandb.save(os.path.join(index_path, "index/0/manfest"))
        logging.info("Index  already exists at %s. Skipping it.", index_path)
        wandb.save(param_file)
        return
    with open(param_file, 'w') as outf:
        outf.write(param_file_format)
    wandb.save(param_file)
    # Run indri processing
    cmd = "{} {}".format(os.path.join(config.indri_bin_path, "IndriBuildIndex"), param_file)
    logging.info(cmd)
    subprocess.run(cmd.split())
    # save manifest on wandb
    wandb.save(os.path.join(index_path, "index/0/manfest"))


def run_queries(config, split, cut):
    if not os.path.isdir(os.path.join(config.data_home, "runs")):
        os.mkdir(os.path.join(config.data_home, "runs"))
    # Generate indri params
    indri_param_format = """<parameters>
  <threads>{}</threads>
  <trecFormat>true</trecFormat>
  <index>{}</index>
  <count>{}</count>
  <runID>{}</runID>
{}
</parameters>"""
    query_param_format = "  <query>\n    <number>{}</number>\n    <text>#combine({})</text>\n  </query>"
    if split == "test":
        queries_file = os.path.join(config.data_home, "queries/test.tokenized.tsv")
    elif split == "dev":
        queries_file = os.path.join(config.data_home, "queries/dev.tokenized.tsv")
    elif split == "train":
        queries_file = os.path.join(config.data_home, "queries/msmarco-doctrain-queries.tsv.tokenized")
    if cut:
        index_path = os.path.join(config.data_home, "indexes/cut-tokenized")
        cut = "cut"
    else:
        index_path = os.path.join(config.data_home, "indexes/full-tokenized")
        cut = "full"
    param_path = os.path.join(config.data_home, "indri_params", "QL_{}-{}.indriparam".format(split, cut))
    runID = "QL_{}_{}".format(split, cut)
    if not os.path.isfile(param_path) or "query_{}".format(split) in config.force_steps:
        # Read queries from tsv
        queries_lines = []
        pattern = re.compile('([^\s\w]|_)+')  # noqa W605
        for line in open(queries_file):
            query_id, query = line.strip().split("\t")
            # clean line of non-alpha. Indri doesn't like them. (may interpret as some command)
            query = pattern.sub('', query)
            queries_lines.append(query_param_format.format(query_id, query))
        all_queries_lines = "\n".join(queries_lines)
        indri_param_format = indri_param_format.format(config.number_of_cpus, index_path, config.indri_top_k, runID, all_queries_lines)
        if split == "test":
            assert len(queries_lines) == config.test_set_size
        with open(param_path, 'w') as outf:
            outf.write(indri_param_format)
        logging.info("Saved params file at %s", param_path)
    else:
        logging.info("Already found file %s. Not recreating it", param_path)
    # Actually run Indri
    # wandb.save(param_path)
    if not os.path.isdir(os.path.join(config.data_home, "runs")):
        os.mkdir(os.path.join(config.data_home, "runs"))
        logging.info("Creating runs folder at %s", os.path.join(config.data_home, "runs"))
    run_path = os.path.join(config.data_home, "runs/QL_{}-{}.run".format(split, cut))
    if not os.path.isfile(run_path) or "query_{}".format(split) in config.force_steps:
        indri_path = os.path.join(config.indri_bin_path, "IndriRunQuery")
        logging.info("Running Indri process with command %s %s", indri_path, param_path)
        output = subprocess.check_output([indri_path, param_path])
        with open(run_path, 'w') as outf:
            outf.write(output.decode("utf-8"))
    # Run trec_eval
    wandb.save(run_path)
    if split == "train":
        qrel_path = os.path.join(config.data_home, "qrels/msmarco-doctrain-qrels.tsv")
    else:
        qrel_path = os.path.join(config.data_home, "qrels/{}.tsv".format(split))
    trec_eval_cmd = "{} -q -c -m {} {} {}".format(config.trec_eval_path, config.metric, qrel_path, run_path)
    logging.info("Running trec_eval with command: %s", trec_eval_cmd)
    output = subprocess.check_output(trec_eval_cmd.split()).decode("utf-8")
    final_metric = float(output.split("\n")[-2].split("\t")[-1])
    key_name = "{}_{}_{}".format(config.metric, split, cut)
    wandb.run.summary[key_name] = final_metric
    logging.info("%s for %s-%s: %f", config.metric, split, cut, final_metric)
