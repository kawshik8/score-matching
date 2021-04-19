# This file manages procedures like pretraining, training and evaluation.
import torch
import math
import logging as log
import os
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *
from data import *
from models import *
from args import * 

class Trainer(object):
    def __init__(self, args):
        """
        Setup training / evaluating
        """
        
        self.args = args

        if args.model_objective == 'score':
            self.model = UNet()
        elif args.model_objective == 'energy':
            self.model = Encoder()

        self.model.to(self.args.device)
        
        if args.dataset == 'cifar10':
            self.train_data = cifar10(args, 'train')
            self.val_data = cifar10(args, 'val')
            self.test_data = cifar10(args, 'test')

        self.train_loader = DataLoader(self.train_data, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)
        self.val_loader = DataLoader(self.val_data, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)
        self.test_loader = DataLoader(self.test_data, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)

        if 'sgd' in args.optimizer:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = args.lr, momentum = 0.9 if "mom" in args.optimizer else 0)
        elif args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.lr)

        self.sampling_trainloader = DataLoader(self.train_data, batch_size = args.sampling_batch_size, shuffle=True, num_workers = args.num_workers)
        self.sampling_testloader = DataLoader(self.test_data, batch_size = args.sampling_batch_size, shuffle=True, num_workers = args.num_workers)

        self.log = create_logger(__name__, silent=False, to_disk=True,
                                 log_file=args.model_dir + "log.txt")
        self.log.info("Setup trainer for %s" % args.dataset)

        tensorboard_log_dir = os.path.join(args.model_dir, 'tensorboard/tb_log.info')
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)


    def get_loader(self, split):
        if split == 'train':
            data_loader = self.train_loader
        elif split == 'val':
            data_loader = self.val_loader
        else:
            data_loader = self.test_loader

        return data_loader

    def train_test(self, what, split, epoch):
        """
        Train the model
        """
        assert split in {'train','val','test'}
        assert what  in {'train','eval'}

        if what == 'train':
            self.model.train()
        else:
            self.model.eval()

        data_loader = self.get_loader(split)
        total_losses = []

        for i, batch in enumerate(data_loader):
            print(i,batch.shape)
            if i > 5:
                break
            step = epoch * len(data_loader) + i
            batch = batch.to(self.args.device)

            all_losses = []
            for noise_level in self.args.noise_std:
                noise = torch.normal(mean=0, std=noise_level, size=batch.shape).to(self.args.device)

                noisy_batch = batch + noise 

                image_diff = (noisy_batch - batch)

                if self.args.reweight:
                    image_diff = image_diff / (noise_level**2)

                if self.args.model_objective == 'score':
                    energy_gradient = self.model(noisy_batch)                        

                else:
                    energy = self.model(noisy_batch)
                    noisy_batch.requires_grad = True
                    energy_gradient = torch.autograd.grad(outputs=energy, inputs=noisy_batch,
                                                            grad_outputs=torch.ones(energy.size()).to(self.args.device),
                                                            create_graph=True, retain_graph=True, only_inputs=True)[0]

                    self.writer.add_scalar(what + "_" + split + '-set/energy_batch_noise-' + str(noise_level), energy.mean(), step)

                self.writer.add_scalar(what + "_" + split + '-set/energy_gradient_l2norm_batch_noise-' + str(noise_level), torch.norm(energy_gradient,dim=1).mean(), step)

                loss = (energy_gradient - image_diff)**2
                loss = torch.mean(loss)  

                self.writer.add_scalar(what + "_" + split + '-set/loss_batch_noise-' + str(noise_level), loss, step)  

                all_losses.append(loss)

                print(i, loss, torch.norm(energy_gradient,dim=1).mean(), torch.norm(image_diff,dim=1).mean())

            loss = torch.stack(all_losses).mean().to(self.args.device)
            self.writer.add_scalar(what + "_" + split + '-set/total_loss_batch', loss, step)

            if what == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_losses.append(loss.detach().cpu().numpy())

        return np.mean(total_losses)    


    def training(self):

        best_loss = None
        patient_steps = 0
        for epoch in range(self.args.n_epochs):
            
            self.log.info("\nepoch: " + str(epoch))

            train_loss = self.train_test('train', 'train', epoch)
            self.writer.add_scalar('train_train-set/epoch_loss', train_loss, epoch)

            eval_loss = self.train_test('eval', 'val', epoch)
            self.writer.add_scalar('eval_val-set/epoch_loss', train_loss, epoch)

            self.log.info("\t\tTrain Loss: " + str(train_loss) + "\n\t\tVal Loss: " + str(eval_loss))

            if best_loss is None or eval_loss < best_loss:
                best_loss = eval_loss
                self.log.info("Better model found at epoch " + str(epoch) + " with eval loss " + str(best_loss))
                patient_steps = 0
                torch.save(self.model.state_dict(), self.args.model_dir + "best.ckpt")
            else:
                patient_steps += 1
                if patient_steps == self.args.early_stop_patience:
                    self.log.info("Early stoppin at epoch " + str(epoch) + " since eval loss " + str(best_loss) + " didn't improve for " + str(patient_steps) + " epochs")
                    break

            torch.save(self.model.state_dict(), self.args.model_dir + "epoch.ckpt")
            if self.args.optimizer=='adam':
                torch.save(self.optimizer.state_dict(), self.args.model_dir + "epoch_optimizer.ckpt")


        test_loss = self.train_test('eval', 'test', 0)
        self.log.info("Test Loss: " + str(test_loss))

    def sampling(self, split):

        assert split in {'train','val','test'}

        self.model.eval()

        data_loader = self.get_loader(split)

        for i, batch in enumerate(data_loader):
            batch = batch.to(self.args.device)
            
            step = 0

            if self.args.init_value == 'uniform':
                curr_batch = torch.rand(batch.shape)

            self.log.info("batch: " + str(i))

            prev_batch = curr_batch.clone() + 100

            while ((curr_batch - prev_batch)**2).mean() > 1e-4: #torch.norm(curr_batch - batch ,dim=1).mean() > 0.01 and #step <= self.args.max_step:
            

                if self.args.model_objective == 'score':
                    energy_gradient = self.model(curr_batch)  

                else:
                    energy = self.model(curr_batch)
                    curr_batch.requires_grad = True
                    energy_gradient = torch.autograd.grad(outputs=energy, inputs=curr_batch,
                                                            grad_outputs=torch.ones(energy.size()).to(self.args.device),
                                                            create_graph=True, retain_graph=True, only_inputs=True)[0]

                self.log.info("step: " + str(step) + "\nPredicted gradient: " + str(torch.norm(energy_gradient,dim=1).mean()) + "\nTarget gradient: " + str(torch.norm(curr_batch - batch, dim=1).mean()) + "\nImage step Diff: " + str(((curr_batch - prev_batch)**2).mean()))

                prev_batch = curr_batch.clone()

                if self.args.sampling_strategy == 'vanilla':
                    print("vanilla")
                    curr_batch = curr_batch + self.args.step_lr * energy_gradient
                elif self.args.sampling_strategy == 'langevin':
                    curr_batch = curr_batch + (self.args.step_lr/2) * energy_gradient + (self.args.step_lr**0.5)*torch.normal(mean=0,std=1,size=energy_gradient.size())

                self.args.step_lr = self.args.step_lr * self.args.lr_anneal

                self.log.info("\nImage step Diff: " + str(((curr_batch - prev_batch)**2).mean()))
                
                step += 1 
            

if __name__ == '__main__':
    args = process_args()
    trainer = Trainer(args)
    trainer.training()
    # trainer.sampling('test')




                





