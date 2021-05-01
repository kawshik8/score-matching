# This file defines all the configuations of the program
import argparse
import os
import torch
import numpy as np
import json

parser = argparse.ArgumentParser()

# cpu vs gpu
device_parser = parser.add_argument_group("device","device params")
device_parser.add_argument("--device", type=str, default="cuda:0", help="which device to run on")
device_parser.add_argument("--num-workers", type=int, default=2, help="number of cpu workers in iterator")

# data params
data_parser = parser.add_argument_group("data","params of data")
data_parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist","celeba"], help="dataset")
data_parser.add_argument("--data-dir", type=str, default="../../data/", help="directory to access data")

# model params
model_parser = parser.add_argument_group("model","params of model")
model_parser.add_argument("--mfile", type=str, default=None, help="name of the file")
model_parser.add_argument("--model-dir", type=str, default="./models/", help="directory of the data files")
model_parser.add_argument("--model-objective", type=str, default="score", choices=["score", "energy"], help="output of the model")
model_parser.add_argument("--unet-depth", type=int, default=3, help="depth of unet model")
model_parser.add_argument("--unet-block", type=str, default='conv_block', choices=['conv_block','res_block'], help="depth of unet model")
model_parser.add_argument("--norm-type", type=str, default='batch', choices=['batch','instance','instance++'], help="depth of unet model")
model_parser.add_argument('--reweight',type=int,default=1, help='reweight action diff and egrad using variance values')

# training params
train_parser = parser.add_argument_group("train","train_opt params")
train_parser.add_argument("--batch-size", type=int, default=64, help="number of images per minibatch")
train_parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
# parser.add_argument("--grad-clip", type=float, default=0.5, help="gradient clip")
train_parser.add_argument("--n-epochs", type=int, default=100, help="total number of epochs")
train_parser.add_argument("--early-stop-patience", type=int, default=10, help="number of epochs to wait for val loss to decrease before training stops early")
train_parser.add_argument("--optimizer", type=str, default='adam', choices=['sgd','sgd_mom','adam'], help="optimization strategy")
train_parser.add_argument("--scheduler", type=str, default=None, choices=['reduce_plateau','one_cycle'], help="lr schedule strategy")
###
train_parser.add_argument("--reduce-lr-patience", type=int, default=4, help="patience for reduce lr on plateau")
train_parser.add_argument("--reduce-lr-factor", type=float, default=0.5, help="factor for reduce lr on plateau")
###
train_parser.add_argument("--cycle-pct", type=float, default=0.2, help="pct factor for one cycle lr")
###
train_parser.add_argument("--model-selection", type=str, default='train', choices=['train','sampling'], help="strategy to select best model")
train_parser.add_argument("--selection-num-samples", type=int, default=1000, help="number of images to sample for selecting best model")
train_parser.add_argument("--fid-layer", type=int, default=-1, help="which layer to use for activations")
train_parser.add_argument('--distance-metric', default='l2', type=str, choices = ['l2','ned'],help='l2 distance vs normalized euclidean distance')
train_parser.add_argument("--weight-decay", type=float, default=0, help="weight decay for training")
train_parser.add_argument("--noise-std", type=str, default="10", help="Standard deviation of noise (Give multiple noise levels separated by commas)")

# Sampling options
sampling_parser = parser.add_argument_group("sampling","sampling params")
sampling_parser.add_argument('--clamp', action='store_true', help='clamp values after each step to [0,1]')
sampling_parser.add_argument("--ntest", type=int, default=1000, help="no of samples to generate")
sampling_parser.add_argument("--save-nsamples", type=int, default=100, help="store generated images")
sampling_parser.add_argument("--sampling-log-freq", type=int, default=99, help="frequency of logging during sampling")
sampling_parser.add_argument("--sampling-batch-size", type=int, default=128, help="number of images per minibatch")
sampling_parser.add_argument("--sampling-strategy", type=str, default='vanilla', choices=['vanilla','langevin','ann_langevin'], help="sampling strategy")
sampling_parser.add_argument('--init-value', type=str, default='uniform', choices=['zeros','orig','random','uniform'],help='where to start during sampling')
# parser.add_argument('--init-noise', type=float, default=10, help='initial noise level to start from during sampling')
sampling_parser.add_argument('--step-lr', default=0.1, type=float, help='provide a list to use a mixture of learning rates')
sampling_parser.add_argument('--step-fgrad', type=float, default=1e-4, help='final grad to stop at')
sampling_parser.add_argument('--max-step', type=int, default=250, help='maximum no of steps for MCMC sampling')
# parser.add_argument('--step-grad-choice', type=str, default='rgrad', choices=['rgrad','pgrad','mgrad','agrad'], help='step using predicted gradient vs real gradient vs equal combination vs annealed combination (0.25 tgrad 0.5 annealed prob mixture 0.25 pgrad)')
sampling_parser.add_argument('--lr-anneal-strategy', type=str, default='const', choices=['const','istep'], help='constantly anneal vs anneal wrt steps taken)')
sampling_parser.add_argument('--lr-anneal', type=float, default=1.0, help='anneal constant to multiply at each step')

# Testing params
test_parser = parser.add_argument_group("test","sampling params")
test_parser.add_argument('--test-model', action='store_true', help='testing mode')
test_parser.add_argument('--ckpt-type', type=str, default='best', choices = ['best','epoch','best_fid','best_iscore'], help='testing mode')
test_parser.add_argument('--test-split', type=str, default='test', help='testing mode')
test_parser.add_argument('--load-mdir', type=str, default=None, help='load dir for trained model')
test_parser.add_argument('--n-models-test', type=int, default=1, help='no of models generated by dropout to use for calculating epistemic uncertainty')
test_parser.add_argument("--isave-dir", type=str, default=None, help="no of samples to generate")

def process_args():

    args = parser.parse_args()

    if args.test_model:
        if not os.path.exists(args.load_mdir + "/config.json"):
            print("Make sure you use similar model params to the one used during training")

        else:
            config = json.load(open(args.load_mdir + "/config.json",'r'))
            
            for group in config:
                if group == 'model':
                    for key in config[group]:
                        setattr(args,key,config[group][key])

    else:
        if args.mfile is None:
            print("Please give a name for the experiment")
            exit(0)
        elif os.path.exists(args.model_dir + args.mfile):
            for i in range(100):
                args.mfile = args.mfile + '_' + str(i)
                if not os.path.exists(args.model_dir + args.mfile):
                    break

    arg_groups={}

    for group in parser._action_groups:
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title]=group_dict #argparse.Namespace(**group_dict)

    # print(arg_groups)

    args.reweight = True if args.reweight==1 else False

    args.device = torch.device(args.device)

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    if not args.test_model:
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
        args.noise_std = np.exp(np.linspace(np.log(a),np.log(al),l))

    config = arg_groups
    json_file = json.dumps(config)
    f = open((args.model_dir if not args.test_model else args.load_mdir) + "/" + ("config.json" if not args.test_model else "test_config.json"),"w+")
    f.write(json_file)
    f.close()

    if args.test_model:
        args.model_dir = args.load_mdir
        args.load_mdir += "/" + args.ckpt_type + ".ckpt"
        
    
    return args

if __name__ == '__main__':
    process_args()
