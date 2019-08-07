#Heavily inspired by the docker image....
import argparse
import subprocess
import os
import psutil

print("Indexing...")

reserved_memory = (1024**3)//2
available_memory = psutil.virtual_memory()[0]
usable_memory = (available_memory - reserved_memory)//1024**2

my_env = os.environ.copy()

my_env["TERRIER_HEAP_MEM"] = f"{usable_memory}M"

parser = argparse.ArgumentParser()

parser.add_argument("--collection", type=str, required=True, help="Collection Name")
parser.add_argument("--path", type=str, required=True, help="Path to Collection")
parser.add_argument("--format", type=str, required=False, default="trectext", help="Collection Format")
parser.add_argument("--index_path", type=str, required=True, help="Path for storing the index")
parser.add_argument("--terier_path", type=str, required = True, help="Terrier binary path")
parser.add_argument("--opts", type=str, required=False, nargs="*", help="Aditional options to be passed to the indexer")

args, _ = parser.parse_known_args()

index_options = {
        "default" : {},
        #http://ir.dcs.gla.ac.uk/wiki/Terrier/Disks1%262
        "robust04" : {"terrier.mvn.coords": "org.apache.commons:commons-compress:1.18",
            "files.mappings":"z:org.apache.commons.compress.compressors.z.ZCompressorInputStream:null," + 
            "0z:org.apache.commons.compress.compressors.z.ZCompressorInputStream:null," + 
            "1z:org.apache.commons.compress.compressors.z.ZCompressorInputStream:null," +
            "2z:org.apache.commons.compress.compressors.z.ZCompressorInputStream:null",
            "TrecDocTags.process":"TEXT,H3,DOCTITLE,HEADLINE,TTL"},
        #http://ir.dcs.gla.ac.uk/wiki/Terrier/DOTGOV2
        "gov2" : {"trec.collection.class":"TRECWebCollection",
            "indexer.meta.forward.keys":"docno,url",
            "indexer.meta.forward.keylens":"26,256",
            "metaindex.compressed.crop.long":"true"
            },
        #http://ir.dcs.gla.ac.uk/wiki/Terrier/ClueWeb09-B
        "cw09b" : {
            "trec.collection.class":"WARC018Collection",
            "indexer.meta.forward.keys":"docno,url",
            "indexer.meta.forward.keylens":"26,256",
            "indexer.meta.reverse.keys":"docno",
            "metaindex.compressed.crop.long":"true"},
        #http://ir.dcs.gla.ac.uk/wiki/Terrier/ClueWeb12
        "cw12b" : {
            "trec.collection.class":"WARC10Collection",
            "indexer.meta.forward.keys":"docno,url",
            "indexer.meta.forward.keylens":"26,256",
            "indexer.meta.reverse.keys":"docno",
            "metaindex.compressed.crop.long":"true"},
        "core18" : {
            "terrier.mvn.coords":"uk.ac.gla.dcs.terrierteam:terrier-wapo:0.1",
            "trec.collection.class":"uk.ac.gla.terrier.indexing.WAPOCollection",
            "indexer.meta.forward.keylens":"40"
            }
        "msmarco_docs":{
            "TrecDocTags.process":"TEXT",
            "metaindex.compressed.crop.long":"true"
        }

print(args.json["opts"])

name, path = collection, path
params = " ".join([f"-D{k}={v}" for k, v in index_options[name].items()])
opts = ""
if args.opts is not None:
    opts = {k.split("=")[0] = k.split("="[1]) for k in args.opts}
    if "block.indexing" in opts and opts["block.indexing"]=='true':
        block_index=True
        del opts['block.indexing']
    opts = opts + " ".join( [f"-D{k}={v}" for k, v in opts.items()] )
    print(opts)
    
subprocess.run([os.path.join(args.terier_path, "bin", "trec_setup.sh"), args.path, env=my_env)

spec_path = os.path.join(args.terrier_path, 'etc', 'collection.spec')

subprocess.run(["/bin/bash", "-c", "egrep -vi 'readme' /work/terrier-core/etc/collection.spec > /work/terrier-core/etc/collection.spec.new; mv /work/terrier-core/etc/collection.spec.new /work/terrier-core/etc/collection.spec"], env=my_env)

