import pandas as pd
import rdkit
from rdkit import DataStructs, Chem
from rdkit.Chem import MACCSkeys, Draw
import torch
from fast_jtnn.mol_tree import main_mol_tree

from fast_bo.gen_latent import main_gen_latent
main_gen_latent('./QDB9/data/train.txt', './QDB9/data/vocab.txt', './QDB9/jtvae/model/model.epoch-19', output_path='./QDB9/jtvae/vae-model/descriptors/',)