import subprocess
import os
import logging
import warnings
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
