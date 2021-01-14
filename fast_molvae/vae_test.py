import sys
sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np

import math, random, sys
import argparse
from fast_jtnn import *
import rdkit

def load_model(vocab, model_path, hidden_size=450, latent_size=56, depthT=20, depthG=3):
    vocab = [x.strip("\r\n ") for x in open(vocab)] 
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
    dict_buffer = torch.load(model_path)
    model.load_state_dict(dict_buffer)
    model = model.cuda()

    torch.manual_seed(0)
    return model

def main_vae_test(test, vocab, model_path, batch_size=32, hidden_size=450, latent_size=56, depthT=20, depthG=3):
    vocab = [x.strip("\r\n ") for x in open(vocab)] 
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
    dict_buffer = torch.load(model_path)
    model.load_state_dict(dict_buffer)
    model = model.cuda()
    model.train(False)
    total_step = 0
    
    meters = np.zeros(4)
    
    
    loader = MolTreeFolder(test, vocab, batch_size)#, num_workers=4)
    for batch in loader:
        total_step += 1
        try:
            loss , kl_div, wacc, tacc, sacc = model(batch, 0)
        except Exception as e:
            print(e)
            continue

        meters = (meters*(total_step-1) + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100]))/total_step
        print("[%d] KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f" % (total_step,\
                                                                     meters[0],\
                                                                     meters[1],\
                                                                     meters[2],\
                                                                     meters[3]))
        sys.stdout.flush()
    
    print("\n\t !Final metrics on test dataset!\n")
    print("[%d] KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f" % (total_step,\
                                                                 meters[0],\
                                                                 meters[1],\
                                                                 meters[2],\
                                                                 meters[3]))
if __name__ == '__main__':
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--latent_size', type=int, default=56)
    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--depthG', type=int, default=3)

    args = parser.parse_args()
    
    main_sample(args.vocab, args.model, args.hidden_size, args.latent_size, args.depthT, args.depthG)