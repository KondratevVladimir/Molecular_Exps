import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch
from fast_jtnn.mol_tree import main_mol_tree
from fast_molvae.sample import main_sample
main_sample('./QDB9/data/vocab.txt', './QDB9/jtvae/vae-model/sample.txt', './QDB9/jtvae/model/model.epoch-19', 100)