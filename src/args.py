# This file defines all the configuations of the program
import argparse
import os
import torch
import numpy as np

parser = argparse.ArgumentParser()

# cpu vs gpu
parser.add_argument("--device", type=str, default="cuda:0", help="which device to run on")
parser.add_argument("--num-workers", type=int, default=2, help="number of cpu workers in iterator")

# data params
parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist","celeba"], help="dataset")
parser.add_argument("--data-dir", type=str, default="../../data/", help="directory to access data")
parser.add_argument("--isave-dir", type=str, default=None, help="no of samples to generate")

# model params
parser.add_argument("--mfile", type=str, default=None, help="name of the file")
parser.add_argument("--model-dir", type=str, default="./models/", help="directory of the data files")
parser.add_argument("--model-objective", type=str, default="score", choices=["score", "energy"], help="output of the model")
parser.add_argument("--unet-depth", type=int, default=3, help="depth of unet model")
parser.add_argument("--unet-block", type=str, default='conv_block', choices=['conv_block','res_block'], help="depth of unet model")
parser.add_argument("--norm-type", type=str, default='batch', choices=['batch','instance'], help="depth of unet model")

# noise params
parser.add_argument("--noise-std", type=str, default="10", help="Standard deviation of noise (Give multiple noise levels separated by commas)")
parser.add_argument('--reweight',type=int,default=1, help='reweight action diff and egrad using variance values')

# Loss params
parser.add_argument('--distance-metric', default='l2', type=str, choices = ['l2','ned'],help='l2 distance vs normalized euclidean distance')

# training params
parser.add_argument("--batch-size", type=int, default=64, help="number of images per minibatch")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")

# parser.add_argument("--grad-clip", type=float, default=0.5, help="gradient clip")
# parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay for training")
parser.add_argument("--n-epochs", type=int, default=100, help="total number of epochs")
parser.add_argument("--early-stop-patience", type=int, default=10, help="number of epochs to wait for val loss to decrease before training stops early")
parser.add_argument("--optimizer", type=str, default='adam', choices=['sgd','sgd_mom','adam'], help="optimization strategy")
parser.add_argument("--model-selection", type=str, default='train', choices=['train','sampling'], help="strategy to select best model")
parser.add_argument("--selection-num-samples", type=int, default=1000, help="number of images to sample for selecting best model")
parser.add_argument("--fid-layer", type=int, default=-1, help="which layer to use for activations")

# Sampling options
parser.add_argument('--clamp', action='store_true', help='clamp values after each step to [0,1]')
parser.add_argument("--ntest", type=int, default=1000, help="no of samples to generate")
parser.add_argument("--save-nsamples", type=int, default=100, help="store generated images")
parser.add_argument("--sampling-log-freq", type=int, default=99, help="frequency of logging during sampling")
parser.add_argument("--sampling-batch-size", type=int, default=128, help="number of images per minibatch")
parser.add_argument("--sampling-strategy", type=str, default='vanilla', choices=['vanilla','langevin','ann_langevin'], help="sampling strategy")
parser.add_argument('--init-value', type=str, default='uniform', choices=['zeros','orig','random','uniform'],help='where to start during sampling')
# parser.add_argument('--init-noise', type=float, default=10, help='initial noise level to start from during sampling')
parser.add_argument('--step-lr', default=0.1, type=float, help='provide a list to use a mixture of learning rates')
parser.add_argument('--step-fgrad', type=float, default=1e-4, help='final grad to stop at')
parser.add_argument('--max-step', type=int, default=250, help='maximum no of steps for MCMC sampling')
# parser.add_argument('--step-grad-choice', type=str, default='rgrad', choices=['rgrad','pgrad','mgrad','agrad'], help='step using predicted gradient vs real gradient vs equal combination vs annealed combination (0.25 tgrad 0.5 annealed prob mixture 0.25 pgrad)')
parser.add_argument('--lr-anneal-strategy', type=str, default='const', choices=['const','istep'], help='constantly anneal vs anneal wrt steps taken)')
parser.add_argument('--lr-anneal', type=float, default=1.0, help='anneal constant to multiply at each step')

# Testing params
parser.add_argument('--test-model', action='store_true', help='testing mode')
parser.add_argument('--test-split', type=str, default='test', help='testing mode')
parser.add_argument('--load-mdir', type=str, default=None, help='load dir for trained model')
parser.add_argument('--n-models-test', type=int, default=1, help='no of models generated by dropout to use for calculating epistemic uncertainty')

def process_args():
    # TODO: some asserts, check the arguments
    args = parser.parse_args()

    args.reweight = True if args.reweight==1 else False

    args.device = torch.device(args.device)

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    if args.mfile is None:
        if args.test_model:
            args.mfile = args.load_mdir.split("/")[-1] + "/testing"
        else:
            args.mfile = args.model_objective + "_" + args.dataset + "_noise-" + str(args.noise_std) + "_metric-" + args.distance_metric + "_bsize-" + str(args.batch_size) + "_opt-" + str(args.optimizer) + "_lr-" + str(args.lr) + "_e" + str(args.n_epochs) + "_select-" + str(args.model_selection)

        if args.model_selection == 'sampling':
            args.mfile += "_nsamples-" + str(args.ntest) + "_strategy-" + str(args.sampling_strategy) + "_stepsize-" + str(args.step_lr) + "_maxstep-" + str(args.max_step)
            if args.lr_anneal_strategy=='const' and args.lr_anneal < 1.0:
                args.mfile += "_lr-anneal-" + str(args.lr_anneal)

    args.model_dir += args.mfile + "/"

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    args.isave_dir = args.model_dir + "simages/"
    if not os.path.exists(args.isave_dir):
        os.mkdir(args.isave_dir)

    if len(args.noise_std.split(",")) == 1:
        args.noise_std = args.noise_std.split(",")
        args.noise_std = [float(args.noise_std[0])]
    else:
        std1,stdl,l = args.noise_std.split(",")
        l = int(l)
        a = float(std1)
        al = float(stdl) 
        # print(np.exp(np.linspace(np.log(a),np.log(al),l)))
        args.noise_std = np.exp(np.linspace(np.log(a),np.log(al),l))
        # print(type(args.noise_std[0]))
        # print(args.noise_std)

    # print(args)
    return args

if __name__ == '__main__':
    process_args()
