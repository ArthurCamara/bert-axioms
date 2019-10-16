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
        data_dir = os.path.join(os.path.join(config.data_home, "docs/msmarco-docs.cut.trec"))
        index_path = os.path.join(os.path.join(config.data_home, "indexes/cut-tokenized"))
        param_file_format = param_file_format.format(data_dir, index_path, config.number_of_cpus)
        param_file = os.path.join(config.data_home, "indri_params/indexing_cut.param")
    if os.path.isdir(index_path) and generate_index not in config.force_steps:
        wandb.save(os.path.join(index_path, "index/0/manfest"))
        logging.info("Index already exists. Skipping it.")
        return
    with open(param_file, 'w') as outf:
        outf.write(param_file_format)

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
    else:
        queries_file = os.path.join(config.data_home, "queries/dev.tokenized.tsv")
    if cut:
        index_path = os.path.join(config.data_home, "indexes/cut-tokenized")
    else:
        index_path = os.path.join(config.data_home, "indexes/full-tokenized")
    if cut:
        cut = "cut"
    else:
        cut = "full"
    param_path = os.path.join(config.data_home, "indri_params", "QL_{}-{}.indriparam".format(split, cut))
    runID = "QL_{}_{}".format(split, cut)
    if not os.path.isfile(param_path) or "run_queries" in config.force_steps:
        # Read queries from tsv
        queries_lines = []
        pattern = re.compile('([^\s\w]|_)+')  # noqa W605
        for line in open(queries_file):
            query_id, query = line.strip().split("\t")
            # clean line of non-alpha. Indri doesn't like them.
            query = pattern.sub('', query)
            queries_lines.append(query_param_format.format(query_id, query))
        all_queries_lines = "\n".join(queries_lines)
        indri_param_format = indri_param_format.format(config.number_of_cpus, index_path, config.indri_top_k, runID, all_queries_lines)
        with open(param_path, 'w') as outf:
            outf.write(indri_param_format)
        logging.info("Saved params file at %s", param_path)
    else:
        logging.info("Already found file %s. Not recreating it", param_path)
    # Actually run Indri
    if not os.path.isdir(os.path.join(config.data_home, "runs")):
        os.mkdir(os.path.join(config.data_home, "runs"))
    
    run_path = os.path.join(config.data_home, "runs/QL_{}-{}.run".format(split, cut))
    if not os.path.isfile(run_path) or "run_queries" in config.force_steps:
        indri_path = os.path.join(config.indri_bin_path, "IndriRunQuery")
        output = subprocess.check_output([indri_path, param_path])
        with open(run_path, 'w') as outf:
            outf.write(output.decode("utf-8"))
