import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matchzoo as mz
from tqdm.auto import tqdm

preprocessor = mz.load_preprocessor("processor.mz")
dev_pack_raw = mz.data_pack.load_data_pack("/ssd2/arthur/TREC2019/data/triples-tokenized/dev.datapack/")
dev_pack_processed = preprocessor.transform(dev_pack_raw)
dev_pack_processed.save("dev.processed.datapack")
test_pack_raw = mz.data_pack.load_data_pack("/ssd2/arthur/TREC2019/data/triples-tokenized/test.datapack/")
test_pack_processed = preprocessor.transform(test_pack_raw)

test_pack_processed.save("test.processed.datapack")
