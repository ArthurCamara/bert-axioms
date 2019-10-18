import os
import logging
from tqdm.auto import tqdm


def cut_docs(config):
    """creates a version of the documents file, but with documents limited to a given number of tokens
    """
    trec_format = "<DOC>\n<DOCNO>{}</DOCNO>\n<TEXT>{}</TEXT></DOC>\n"
    doc_file = os.path.join(config.data_home, "docs/msmarco-docs.tokenized.tsv")
    cut_doc_file = os.path.join(config.data_home, "docs/msmarco-docs.tokenized.cut.tsv")
    cut_doc_file_trec = os.path.join(config.data_home, "docs/msmarco-docs.tokenized.cut.trec")
    if os.path.isfile(cut_doc_file) and os.path.isfile(cut_doc_file_trec) and "cut_docs" not in config.force_steps:
        logging.info("Already found cut documents at %s. Skipping", cut_doc_file)
        return
    with open(cut_doc_file, 'w') as outf, open(cut_doc_file_trec, 'w') as outf_trec:
        for line in tqdm(open(doc_file), total=config.corpus_size, desc="Cutting documents"):
            doc_id, doc_text = line[:-1].split("\t")
            doc_cut = " ".join(doc_text.split(" ")[:config.max_document_len])
            outf.write("{}\t{}\n".format(doc_id, doc_cut))
            outf_trec.write(trec_format.format(doc_id, doc_cut))
    logging.info("Created files %s and %s", cut_doc_file, cut_doc_file_trec)
