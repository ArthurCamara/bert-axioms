from urllib import request
import os
import logging
import shutil
import gzip


def fetch_data(config):
    """Makes sure docs, queries and qrels are available"""
    # Check for docs file
    # docs_file = os.path.join(config.data_home, config.docs_file)
    files_to_get = ["docs/msmarco-docs.tsv", "docs/msmarco-docs.trec",
                    "queries/msmarco-doctrain-queries.tsv", "qrels/msmarco-doctrain-qrels.tsv",
                    "queries/msmarco-docdev-queries.tsv", "qrels/msmarco-docdev-qrels.tsv"]
    for file in files_to_get:
        local_file_path = os.path.join(config.data_home, file)
        if not os.path.isfile(local_file_path):
            logging.info("downloading file %s", file)
            url_to_fetch_from = config.download_path + file.split("/")[1] + ".gz"
            request.urlretrieve(url_to_fetch_from, local_file_path + ".gz")
        # Uncompress file
            with gzip.open(local_file_path + ".gz", 'rb') as f_in, open(local_file_path, 'wb') as outf:
                logging.info("Extracting file %s", file)
                shutil.copyfileobj(f_in, outf)
                os.remove(local_file_path + ".gz")
