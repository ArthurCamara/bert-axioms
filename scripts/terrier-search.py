import subprocess
import os
import re
import psutil
import argparse
print("Searching...")



#512MB of reserved system memory
reserved_memory = (1024**3)//2
available_memory = psutil.virtual_memory()[0]
usable_memory = (available_memory - reserved_memory)//1024**2

my_env = os.environ.copy()
my_env["TERRIER_HEAP_MEM"] = f"{usable_memory}M"

parser = argparse.ArgumentParser()


parser.add_argument("--collection", type=str, required=True, help="Collection Name")
parser.add_argument("--format", type=str, required=False, default="trectext", help="Collection Format")
parser.add_argument("--index_path", type=str, required=True, help="Path for storing the index")
parser.add_argument("--output", type=str, required=True, help="Output file")
parser.add_argument("--top_k", type=int, required=False, default=1000, help="Output file")
parser.add_argument("--topic_path", type=str, required=True, help="Topic file")
parser.add_argument("--terrier_path", type=str, required = True, help="Terrier binary path")
parser.add_argument("--config", type=str, required=False, default='bm25', help="retrieval option to use")
parser.add_argument("--qrel_path", type=str, required=True, help="QRELS file path")

args, _ = parser.parse_known_args()

configs = { 
        "bm25" : {"-w" : "BM25",
                  "-Db": "0.87"},
        "pl2"  : {"-w" : "PL2"},
        "dph"  : {"-w" : "DPH"},
        "ql"  : {"-w" : "DirichletLM"},
        "bm25_qe" : {"-w" : "BM25", "-q" : ""},
        "pl2_qe" : {"-w" : "PL2", "-q" : ""},
		"dph_qe" : {"-w" : "DPH", "-q" : ""},
		"ql_qe"  : {"-w" : "DirichletLM", "-q" : ""},
        "bm25_ltr_features" : { 
            "-w" : "BM25", 
            "-c" : "labels:on", 
            "-F" : "Normalised2LETOROutputFormat", 
            "-Dtrec.matching=FatFeaturedScoringMatching,org.terrier.matching.daat.FatFull" : "",
            "-Dfat.featured.scoring.matching.features=FILE" : "",
            "-Dproximity.dependency.type=SD" : "",
            "-Dfat.featured.scoring.matching.features.file=/output/features.list" : ""
            },
        "bm25_ltr_jforest" : {
            "-w" : "BM25",
            "-Dtrec.matching=JforestsModelMatching,FatFeaturedScoringMatching,org.terrier.matching.daat.FatFull" : "",
            "-Dfat.featured.scoring.matching.features=FILE" : "",
            "-Dfat.featured.scoring.matching.features.file=/output/features.list" : "",
            "-Dfat.matching.learned.jforest.model=/output/ensemble.txt" : "",
            "-Dfat.matching.learned.jforest.statistics=/output/jforests-feature-stats.txt" : ""
            },
        "bm25_prox":{
			"-w" : "BM25",
            "-Dmatching.dsms=DFRDependenceScoreModifier" : "",
            "-Dproximity.dependency.type=SD" : "",
            },
		"bm25_prox_qe":{
            "-w" : "BM25",
            "-Dmatching.dsms=DFRDependenceScoreModifier" : "",
            "-Dproximity.dependency.type=SD" : "",
			"-q": ""
            },
		"pl2_prox":{
            "-w" : "PL2",
            "-Dmatching.dsms=DFRDependenceScoreModifier" : "",
            "-Dproximity.dependency.type=SD" : ""
            },
        "pl2_prox_qe":{
            "-w" : "PL2",
            "-Dmatching.dsms=DFRDependenceScoreModifier" : "",
            "-Dproximity.dependency.type=SD" : "",
            "-q": ""
            },
		"dph_prox":{
            "-w" : "DPH",
            "-Dmatching.dsms=DFRDependenceScoreModifier" : "",
            "-Dproximity.dependency.type=SD" : ""
            },
        "dph_prox_qe":{
            "-w" : "DPH",
            "-Dmatching.dsms=DFRDependenceScoreModifier" : "",
            "-Dproximity.dependency.type=SD" : "",
            "-q": ""
            },
		"ql_prox":{
            "-w" : "DirichletLM",
            "-Dmatching.dsms" : "MRFDependenceScoreModifier",
            "-Dproximity.dependency.type=SD" : "",
            },
        "ql_prox_qe":{
            "-w" : "DirichletLM",
            "-Dmatching.dsms" : "MRFDependenceScoreModifier",
            "-Dproximity.dependency.type=SD" : "",
            "-q": "" 
            },	
        }
bin_path = os.path.join(args.terrier_path, "bin", "terrier")
if re.match(r"prox", args.config):
    blocks = subprocess.check_output(f"{bin_path} indexstats -I {args.index_path}/{args.collection}".split()).decode("utf-8").split("\n")[-2]== 'blocks: true'
    if not blocks:
        raise Exception("Index was not built with --opts block.indexing=true")
   
config = configs[args.config]
output_path = os.path.join(args.terrier_path, "var", "results", "run.{}.{}.res".format(args.collection, args.config))
opts = " ".join( [ k + " " + v for k,v in config.items() ] )
cmd = f"""{bin_path} batchretrieve 
	-I {args.index_path}/{args.collection} 
	-o {output_path}
	-t {args.topic_path}
	{opts}
	-DTrecQueryTags.skip=DESC,NARR,DOM,HEAD,SMRY,CON,FAC,DEF
	-Dtrec.querying.dump.settings=false
	-Dtrec.output.format.length={args.top_k}
	"""
print(cmd.replace("\n", " "))
subprocess.run(cmd.split(), env=my_env)

eval_cmd = f"{bin_path} batchevaluate -f -q {args.qrel_path}"
print(eval_cmd)
subprocess.run(eval_cmd.split())
