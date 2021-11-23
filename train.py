import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from musegan import MuseGAN
from data.utils import MidiDataset
from ipdb import set_trace as bp
from data.utils import postProcess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'top', description='Train MusaGAN.')
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--z_dimension", type=int, default=32, help="Z(noise)-space dimension.")
    parser.add_argument("--g_channels", type=int, default=1024, help="Generator hidden channels.")
    parser.add_argument("--g_features", type=int, default=1024, help="Generator hidden features.")
    parser.add_argument("--g_lr", type=float, default=0.001, help="Generator learning rate.")
    parser.add_argument("--c_channels", type=int, default=128, help="Critic hidden channels.")
    parser.add_argument("--c_features", type=int, default=1024, help="Critic hidden features.")
    parser.add_argument("--c_lr", type=float, default=0.001, help="Critic learning rate.")
    args = parser.parse_args()
    # parameters of musegan
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gan_args = args.__dict__.copy()
    gan_args.pop('epochs', None)
    gan_args.pop('batch_size', None)
    gan_args["device"] = device
    # train
    print("Start training ...")
    print("Loading dataset ...")
    dataset = MidiDataset(path='data/chorales/Jsb16thSeparated.npz')
    #bp()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print("Loading model ...")
    musegan = MuseGAN(**gan_args)
    print("Start training ...")
    finalmodel = musegan.train(dataloader=dataloader, epochs=args.epochs)
    print("Training finished.")
    finalmodel = finalmodel.eval()

    batch_size = 1
    z = torch.randn(batch_size, 10, 32).to(device)
    for i in range(10):
        c = [float(0)]*10
        c[i] = float(1)
        c = torch.tensor([c]).to(device)
        fake = finalmodel(c, z)
        # bp()
        preds = fake.cpu().detach().numpy()
        music_data = postProcess(preds)
        filename = f'myexample_{i}.midi'
        music_data.write('midi', fp=filename)
    
    
