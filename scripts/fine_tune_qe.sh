#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=144
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2048
#SBATCH --mail-type=END

TERRIER=/tudelft.net/staff-umbrella/punchbag/msmarco/terrier-core/bin/terrier
INDEX=/tudelft.net/staff-umbrella/punchbag/msmarco/data/index/msmarco_docs
OUTPUT=/tudelft.net/staff-umbrella/punchbag/msmarco/terrier-core/var/results
TOPICS=/tudelft.net/staff-umbrella/punchbag/msmarco/data/dev.topics
ARGS1=-DTrecQueryTags.skip=DESC,NARR,DOM,HEAD,SMRY,CON,FAC,DEF
ARGS2=-Dtrec.querying.dump.settings=false
ARGS3=-Dtrec.output.format.length=1000

for i in 5 10 15 20 50 100 150 200 250 300 350 500
do
  for j in 5 10 15 20 50 100 150 200 250 300 350 500
  do
    cmd="$TERRIER batchretrieve -I $INDEX  -o $OUTPUT/qe_$i.$j.res -t $TOPICS -w BM25 -q $ARGS1 $ARGS2 $ARGS3 -c c:0.87 -Dexpansion.terms=$i -Dexpansion.documents=$j"
    echo $cmd
    srun $cmd
  done
done
