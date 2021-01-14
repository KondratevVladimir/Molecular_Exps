import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
import torch
from fast_jtnn.mol_tree import main_mol_tree

#from fast_molvae.preprocess import main_preprocess
#main_preprocess('./QDB9/data/test.txt', './QDB9/jtvae/processed_test/', num_splits=100)

#print("The data has been preprocessed")

from fast_molvae.vae_train import main_vae_train
model = main_vae_train('./QDB9/jtvae/processed/',\
                       './QDB9/data/vocab.txt',\
                       './QDB9/experiments/1/',\
                       warmup=15000,\
                       print_iter=100,\
                       epoch=30,
                       save_iter=5000,
                       save_verbose=True)

#from fast_molvae.vae_test import main_vae_test
#model = main_vae_test('./QDB9/jtvae/processed_test/', './QDB9/data/vocab.txt', './fast_molvae//vae_model/model.epoch-19')


#from fast_molvae.sample import main_sample
#main_sample('./QDB9/data/vocab.txt', './QDB9/jtvae_small/vae_model/sample.txt', './QDB9/jtvae_small/debug/model.epoch-19', 100)