# Bert Axioms
This is the repository with the code for the Paper Diagnosing _BERT with Retrieval Heuristics_

## Required Data 
In order to run this code, you first need to download the dataset from the [TREC 2019 Deep Learning Track Guidelines](https://github.com/microsoft/TREC-2019-Deep-Learning). The path for these should be specified in the [config file](../release/scripts/config-defaults.yaml)

You also need a working installation of the [Indri Toolkit](https://github.com/diazf/indri) for indexing and retrieval. 

## Parameters
There are a number of hyperparemeter that need to be set (like indri path, number of candidates to be retrieved, random seed etc). These can be set on a config YAML file at [`scripts/config-defaults.yaml`](../release/scripts/config-defaults.yaml). The parameters are handled by [`wandb`](https://wandb.com), but can easily be addapted for any `YAML` reader (take a look at [PyYAML](https://github.com/yaml/pyyaml/).)


## Observations
Note that, for `LNC2`, we use an external `C++` code for dealing with Indri. This is so we can add the duplicated documents to the index without comprimissing scores. This code should be compiled with Indri's `Makefile.app`. This should be as easy as edditing `Makefile.app` from Indri and running `make -f Makefile.app`. (Check https://lemur.sourceforge.io/indri/ for more details).

The removal process of documents from the indri index does not guarantee that the index statistics will change immediately. This can cause slight differences than the more "correct" way to re-create the index from scratch for every duplicated document. 

## Expected Results

The results from this repository may not directly replicate the ones that appear on the paper. This is due to a few performance improvements made after the paper submission. These, however, do not change the final scores and conclusions. Mostly, you may see a increase on alpha-nDCG for all methods, and a increase on QL performance accross the board.

|            |  `nDCG_cut`  |    `TFCI`    |    `TFCII`   |    `MTDC`    |    `LNC1`    |   `LNC2`   |     `TP`     |    `STMC1`   |   `STMC2`   |   `STMC3`  |
|------------|:----------:|:----------:|:----------:|:----------:|:----------:|:---------:|:----------:|:----------:|:----------:|:----------:|
| QL         |   0.3633   | **0.9936** | **0.7008** | **0.8759** | **0.5021** | **1.000** |   0.3852   |   0.4855   | **0.7047** | **0.7011** |
| DistilBERT | **0.4537** |   0.6109   |   0.3945   |   0.5130   |   0.5006   |   0.0003  | **0.4105** | **0.5040** |   0.5120   |   0.5099   |
