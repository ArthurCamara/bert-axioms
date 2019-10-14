import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from keras.utils import multi_gpu_model
import matchzoo as mz
from tqdm.auto import tqdm
import os
from matchzoo.data_generator import DataGeneratorBuilder
import shutil

train_data = mz.data_pack.load_data_pack("../notebooks/train.processed.datapack/")
dev_data = mz.data_pack.load_data_pack("../notebooks/dev.processed.datapack/")
test_data = mz.data_pack.load_data_pack("../notebooks/test.processed.datapack/")

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"  # specify which GPU(s) to be used

class ANMMConfig():
    # preprocessor = mz.preprocessors.ANMMPreprocessor()
    optimizer = 'SGD'
    model = mz.models.ANMM
    generator_flag = 1
    name = 'anmm'
    num_dup = 1
    num_neg = 1
    shuffle = True

    batch_size = 32
    epoch = 3

    parent_path = '/ssd2/arthur/matchzoo/saved_results'
    model_parent_path = '/ssd2/arthur/matchzoo/ANMM'
    model_save_path = os.path.join(model_parent_path, name)
    parameter_set = []
    parameter_set1 = []
    parameter_name = ['dropout_rate', 'num_layers']


class BasicModel(object):
    def __init__(self, config):
        self.config = config

    def mkdir(self):
        folder = os.path.exists(self.config.model_save_path)
        if not folder:
            os.makedirs(self.config.model_save_path)
            print("---  new folder...  ---")
            print("---  new folder...  ---")
        else:
            print("---  There is this folder!  ---")

    def parameter_get(self):
        return self.config.parameter_set, self.config.parameter_name

    def name(self):
        return self.config.name

    def get_path(self):
        return self.config.model_save_path, self.config.model_parent_path

    def model_delete(self):
        trash_dir = self.config.model_parent_path
        try:
            shutil.rmtree(trash_dir)
        except OSError as e:
            print(f'Error: {trash_dir} : {e.strerror}')
        os.mkdir(trash_dir)

    def auto_prepare(self, train_pack, valid_pack, test_pack):
        model = self.config.model()
        task = mz.tasks.Ranking(metrics=['mean_average_precision'])
        model.params['task'] = task
        model_ok, train_ok, preprocesor_ok = mz.auto.prepare(task=task, model_class=type(model), data_pack=train_pack)
        test_ok = preprocesor_ok.transform(test_pack, verbose=0)
        valid_ok = preprocesor_ok.transform(valid_pack, verbose=0)
        return model_ok, train_ok, test_ok, valid_ok
    
    def get_lr(self):
        return self.config.learning_rate


config = ANMMConfig()
model = BasicModel(config)
model_name = model.name()
model.model_delete()

task = mz.tasks.Ranking(metrics=['mean_average_precision', 'ndcg'])
prepared = mz.auto.Preparer(task)
model_class = mz.models.ANMM
preprocessor = mz.load_preprocessor("../notebooks/processor.mz")

model = model_class()
model.params['task'] = task
model.params.update(preprocessor.context)
model.guess_and_fill_missing_params()
model.build()
model._backend = multi_gpu_model(model._backend, gpus=6)
model.compile()
data_gen = DataGeneratorBuilder(num_neg=0, num_dup=0, mode="point", batch_size=16, shuffle=True).build(train_data)
callback = mz.engine.callbacks.EvaluateAllMetrics(model, *dev_data.unpack(), batch_size=32, verbose=0, model_save_path="/ssd2/arthur/matchzoo/ANMM/")

print("done loading")
history = model.fit_generator(data_gen, epochs=3, callbacks=[callback], use_multiprocessing=True, workers=16)
model.save("anmm.model")
model.evaluate(*test_data.unpack())