#!/bin/bash
#SBATCH --job-name=trial_job
#SBATCH --account=csci_ga_2565_0001
#SBATCH --partition=n1s8-t4-1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue

#bash /scratch/SP21ML/run-pytorch.bash 

#### Training!!
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --batch-size 128 --noise-std 1,0.01,10 --optimizer sgd_mom --lr 1e-4
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --mfile toomanyn-232-cifar-ema-ncsnv2 --model-choice ncsnv2 --norm-type instance++ --n-blocks 1 --unet-block res_block --min-max-normalize --act elu --unet-depth 3 --data-dir ../data --lr 1e-3 --batch-size 32 --noise-std 50,0.01,232 --optimizer adam --scheduler reduce_plateau --dataset cifar10 --use-ema 1 --ema-mu 0.999
bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --mfile toomanyn-232-cifar-ema-ncsnv2-lr_1e-5 --model-choice ncsnv2 --norm-type instance++ --min-max-normalize --act elu --data-dir ../data --lr 1e-5 --batch-size 32 --noise-std 50,0.01,232 --optimizer adam --scheduler reduce_plateau --dataset cifar10 --use-ema 1 --ema-mu 0.999
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --mfile toomanyn-100-cifar-ema-elu-minmaxnorm --min-max-normalize --act elu --use-ema 1 --norm-type instance --data-dir ../data --lr 1e-3 --batch-size 32 --noise-std 50,0.01,100 --optimizer adam --scheduler reduce_plateau --dataset cifar10  --ema-mu 0.999 #--model-selection sampling --sampling-strategy ann_langevin --step-lr 1e-5 --max-step 100 #one_cycle --cycle-pct 0.2 #--weight-decay 1e-3 #--model-selection sampling --step-lr 5e-6 --sampling-strategy ann_langevin --max-step 100

#### Sampling!!!
############# MNIST
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-5 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset mnist
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-5 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 1 --dataset mnist
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-5 --lr-anneal 0.99 --max-step 100 --reweight 0 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset mnist
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-5 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset mnist --clamp
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-4 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset mnist
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-3 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset mnist
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-2 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset mnist
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-6 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset mnist
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-6 --lr-anneal 0.99 --max-step 250 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset mnist
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 10,0.01,100 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-5 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset mnist
########### MNIST vanilla
### lr  1e-5, 1e-1, 1e-2, (1e-2,0.99,250), (1,0.99,250), (1e-2,0.999,250), (1e-2,1000), (1e-3,1000), (1e-2,1000,denoise), (1e-2,1000,pgrad), (1e-3,1000,pgrad), (1e-4,1000,pgrad),  (1e-3,5000,pgrad), (5e-3,1000,pgrad), (7.5e-3,1000,pgrad), (5e-3,1000,0.998,pgrad), (5e-3, 1000, 0.999, pgrad), (1e-2, 1000, 0.995,pgrad), (5e-3, 0.9995, 1500)
##### Best vanilla  (5e-3, 1000, pgrad)
##### Best ann langebin (1e-5, 100, 1,0.01,10)
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy vanilla --step-grad-choice rgrad --step-lr 1e-5 --lr-anneal 1.0 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset mnist
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy vanilla --step-grad-choice rgrad --step-lr 1e-2 --lr-anneal 0.999 --max-step 250 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset mnist
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy vanilla --step-grad-choice rgrad --step-lr 1e-2 --lr-anneal 1.0 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset mnist
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy vanilla --step-grad-choice rgrad --step-lr 1e-3 --lr-anneal 1.0 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset mnist
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy vanilla --step-grad-choice pgrad --step-lr 5e-3 --lr-anneal 0.9995 --max-step 1500 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset mnist

############ CIFAR
### more than trained noise (1,0.01) : (1e-5, 50, 25), (1e-5, 100, 10), ( 1e-5, 25, 25)
### nsteps (1,0.01,10) (1e-5) : 50, 75, 100, 125, 150 
## more than trained noise : (1e-5, 50, 20)
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,50 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-cifar-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-5 --lr-anneal 0.99 --max-step 20 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset cifar10

### too many 500 [50,0.01,500] : (1e-5, 5), (1e-5, 5, ema), (1e-5, 2), (1e-6, 25), (5e-6,5), (1e-6, 5), (5e-6, 10), (5e-6, 25), (5e-6, 2), 
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --change-keys --test-model --noise-std 50,0.01,500 --sampling-batch-size 32 --load-mdir models/toomanyn-500-cifar-ema --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 5e-6 --denoise 2 --use-ema 1 --lr-anneal 0.99 --max-step 2 --reweight 1 --sampling-log-freq 99 --ntest 1000 --renormalize 0 --dataset cifar10

#### EMA sampling
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --change-keys --test-model --noise-std 50,0.01,500 --sampling-batch-size 32 --load-mdir models/toomanyn-500-cifar-ema --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-6 --denoise 2 --use-ema 1 --lr-anneal 0.99 --max-step 15 --reweight 1 --ntest 1000 --renormalize 0 --dataset cifar10

### 500 minmaxnorm sampling 
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 50,0.01,500 --sampling-batch-size 32 --load-mdir models/toomanyn-500-cifar-ema-elu-minmaxnorm --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-6 --denoise 2 --use-ema 0 --lr-anneal 0.99 --max-step 10 --min-max-normalize --reweight 1 --ntest 1000 --renormalize 0 --dataset cifar10
### EMA
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 50,0.01,500 --sampling-batch-size 32 --load-mdir models/toomanyn-500-cifar-ema-elu-minmaxnorm --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-6 --denoise 1 --use-ema 1 --lr-anneal 0.99 --max-step 25 --min-max-normalize --reweight 1 --ntest 1000 --renormalize 0 --dataset cifar10

## too many 232  minmaxnorm sampling
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 50,0.01,232 --sampling-batch-size 32 --load-mdir models/toomanyn-232-cifar-ema-i++-mmnorm-elu --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 5e-6 --denoise 1 --use-ema 0 --lr-anneal 0.99 --max-step 3 --min-max-normalize --reweight 1 --ntest 1000 --renormalize 0 --dataset cifar10

## too many 250:
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 50,0.01,250 --sampling-batch-size 32 --load-mdir models/toomanyn-250-cifar-ema --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-6 --lr-anneal 0.99 --use-ema 1 --max-step 1 --denoise --reweight 1 --sampling-log-freq 0 --ntest 1000 --renormalize 0 --dataset cifar10

#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-cifar-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-6 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset cifar10
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-cifar-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-4 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset cifar10
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-cifar-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 5e-5 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset cifar10
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-cifar-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 5e-6 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset cifar10
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-cifar-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 4e-5 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset cifar10
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-cifar-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 3e-5 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset cifar10
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-cifar-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 2e-5 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset cifar10
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-cifar-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 7.5e-6 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset cifar10
# ann_langevin (1e-5,100,denoise), (1e-5,250), (1e-5,100,denoise), (1e-5,125)


################### ema
# 1e-5, 5e-5, 5e-6 
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-cifar-ema_2 --use-ema 1 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 5e-6 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset cifar10

################## res block
# 1e-5, 5e-5, 5e-6, 1e-6, 5e-7, 2.5e-6, 2e-6, 3e-6, 4e-6
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-cifar-lr_1e-3-resblock --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 2e-6 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset cifar10



#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 32 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-5 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0 --dataset mnist --clamp
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 1 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-4 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 1000 --renormalize 0
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 1 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-5 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 100 --renormalize 0
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 1 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-6 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 100 --renormalize 0
#bash /scratch/SP21ML/run-pytorch.bash python3 -u src/trainer.py --data-dir ../data --test-model --noise-std 1,0.01,10 --sampling-batch-size 1 --load-mdir models/multin-instance-rplat-mnist-lr_1e-3 --sampling-strategy ann_langevin --step-grad-choice pgrad --step-lr 1e-5 --lr-anneal 0.99 --max-step 100 --reweight 1 --sampling-log-freq 49 --ntest 100 --renormalize 0 --clamp 
