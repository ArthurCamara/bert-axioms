import os
import logging
from tqdm.auto import tqdm


def cut_docs(config):
    """creates a version of the documents file, but with documents limited to a given number of tokens
    """
    trec_format = "<DOC>\n<DOCNO>{}</DOCNO>\n<TEXT>{}</TEXT></DOC>\n"
    cut_doc_file = os.path.join(config.data_home, "docs/msmarco-docs.tokenized.cut.tsv")
    cut_doc_file_trec = os.path.join(config.data_home, "docs/msmarco-docs.tokenized.cut.trec")
    cut_doc_file_bert = os.path.join(config.data_home, "docs/msmarco-docs.tokenized.cut.bert")
    doc_file_bert = os.path.join(config.data_home, "docs/msmarco-docs.tokenized.bert")
    if (os.path.isfile(cut_doc_file) and os.path.isfile(cut_doc_file_trec) 
            and os.path.isfile(cut_doc_file_bert) and "cut_docs" not in config.force_steps):
        logging.info("Already found cut documents at %s. Skipping", cut_doc_file)
        return
    with open(cut_doc_file, 'w') as outf, open(cut_doc_file_trec, 'w') as outf_trec, open(cut_doc_file_bert, 'w') as outf_bert:  # noqa: E501
        for line in tqdm(open(doc_file_bert), total=config.corpus_size, desc="Cutting documents"):
            doc_id, doc_text = line[:-1].split("\t")
            doc_cut = eval(doc_text)[:config.max_document_len]
            outf_bert.write("{}\t{}\n".format(doc_id, doc_cut))
            doc_cut = ' '.join(doc_cut).replace("##", "")
            break
            outf.write("{}\t{}\n".format(doc_id, doc_cut))
            outf_trec.write(trec_format.format(doc_id, doc_cut))
    logging.info("Created files %s and %s", cut_doc_file, cut_doc_file_trec)
