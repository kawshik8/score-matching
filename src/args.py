# This file defines all the configuations of the program
import argparse
import os
import torch

parser = argparse.ArgumentParser()

# cpu vs gpu
parser.add_argument("--device", type=str, default="cuda:0", help="which device to run on")
parser.add_argument("--num-workers", type=int, default=16, help="number of cpu workers in iterator")

# data params
parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"], help="dataset")
parser.add_argument("--data-dir", type=str, default="../../data/", help="directory to access data")

# model params
parser.add_argument("--mfile", type=str, default=None, help="name of the file")
parser.add_argument("--model-dir", type=str, default="./models/", help="directory of the data files")
parser.add_argument("--model-objective", type=str, default="score", choices=["score", "energy"], help="output of the model")
parser.add_argument("--unet-depth", type=str, default=5, help="depth of unet model")

# noise params
parser.add_argument("--noise-std", type=str, default="10", help="Standard deviation of noise (Give multiple noise levels separated by commas)")
parser.add_argument('--reweight',action='store_true',help='reweight action diff and egrad using variance values')

# Loss params
parser.add_argument('-distance-metric', default='l2', type=str, choices = ['l2','ned'],help='l2 distance vs normalized euclidean distance')

# training params
parser.add_argument("--batch-size", type=int, default=64, help="number of images per minibatch")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
# parser.add_argument("--grad-clip", type=float, default=0.5, help="gradient clip")
# parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay for training")
parser.add_argument("--n-epochs", type=int, default=100, help="total number of epochs")
parser.add_argument("--early-stop-patience", type=int, default=10, help="number of epochs to wait for val loss to decrease before training stops early")
parser.add_argument("--optimizer", type=str, default='sgd', choices=['sgd','sgd_mom','adam'], help="optimization strategy")

# Sampling options
parser.add_argument("--sampling-batch-size", type=int, default=1, help="number of images per minibatch")
parser.add_argument("--sampling-strategy", type=str, default='vanilla', choices=['vanilla','langevin'], help="sampling strategy")
parser.add_argument('--init-value', type=str, default='uniform', choices=['zeros','orig','random','uniform'],help='where to start during sampling')
parser.add_argument('--init-noise', type=float, default=10, help='initial noise level to start from during sampling')
parser.add_argument('--step-lr', default=0.1, help='provide a list to use a mixture of learning rates')
parser.add_argument('--step-fgrad', type=float, default=1e-4, help='final grad to stop at')
parser.add_argument('--max-step', type=int, default=250, help='maximum no of steps for MCMC sampling')
parser.add_argument('--step-grad-choice', type=str, default='rgrad', choices=['rgrad','pgrad','mgrad','agrad'], help='step using predicted gradient vs real gradient vs equal combination vs annealed combination (0.25 tgrad 0.5 annealed prob mixture 0.25 pgrad)')
parser.add_argument('--lr-anneal-strategy', type=str, default='const', choices=['const','istep'], help='constantly anneal vs anneal wrt steps taken)')
parser.add_argument('--lr-anneal', type=float, default=1.0, help='anneal constant to multiply at each step')

# Testing params
parser.add_argument('--test-model', action='store_true', help='testing mode')
parser.add_argument('--load-mdir', type=str, default=None, help='load dir for trained model')
parser.add_argument('--n-models-test', type=int, default=1, help='no of models generated by dropout to use for calculating epistemic uncertainty')

def process_args():
    # TODO: some asserts, check the arguments
    args = parser.parse_args()

    args.device = torch.device(args.device)

    args.mfile = args.model_objective + "_" + args.dataset + "_noise-" + str(args.noise_std) + "_metric-" + args.distance_metric + "_bsize-" + str(args.batch_size) + "_lr-" + str(args.lr) + "_e" + str(args.n_epochs)
    args.model_dir += args.mfile + "/"

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    args.noise_std = args.noise_std.split(",")
    for i in range(len(args.noise_std)):
        args.noise_std[i] = float(args.noise_std[i])

    # print(args)
    return args

process_args()
