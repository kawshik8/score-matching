# This file manages procedures like pretraining, training and evaluation.
import torch
import math
import logging as log
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *
from data import *
from models import *
from NCSNv2 import *
from args import * 
import cv2
from collections import OrderedDict

class Trainer(object):
    def __init__(self, args):
        """
        Setup training / evaluating
        """
        
        self.args = args

        if args.model_objective == 'score':
            if args.model_choice == 'custom':
                self.model = UNet(depth=self.args.unet_depth, norm_type=self.args.norm_type, block=self.args.unet_block, n_channels=(1 if self.args.dataset=='mnist' else 3), act_type = self.args.act)
            elif args.model_choice == 'ncsnv2':
                self.model = NCSNv2(n_channels=(1 if self.args.dataset=='mnist' else 3), norm_type=self.args.norm_type, act_type=self.args.act)
            elif args.model_choice == 'ncsnv2':
                self.model = NCSNv2Deeper(n_channels=(1 if self.args.dataset=='mnist' else 3), norm_type=self.args.norm_type, act_type=self.args.act)
            elif args.model_choice == 'ncsnv2':
                self.model = NCSNv2Deepest(n_channels=(1 if self.args.dataset=='mnist' else 3), norm_type=self.args.norm_type, act_type=self.args.act)

        elif args.model_objective == 'energy':
            self.model = Encoder(depth=self.args.unet_depth, norm_type=self.args.norm_type, block=self.args.unet_block, n_channels=(1 if self.args.dataset=='mnist' else 3), act_type = self.args.act)

        self.inception = Inception(args.fid_layer).eval()
        
        # if args.dataset == 'cifar10':
        self.train_data = Data(args, 'train')
        self.val_data = Data(args, 'valid')
        self.test_data = Data(args, 'test')
        self.data = {"train": self.train_data, "valid":self.val_data, "test":self.test_data}

        self.train_loader = DataLoader(self.train_data, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)
        self.val_loader = DataLoader(self.val_data, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)
        self.test_loader = DataLoader(self.test_data, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)
        self.tloader = {'train':self.train_loader, 'valid':self.val_loader, 'test': self.test_loader}

        if 'sgd' in args.optimizer:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = args.lr, momentum = 0.9 if "mom" in args.optimizer else 0, weight_decay= args.weight_decay)
        elif args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.lr, weight_decay= args.weight_decay)

        if args.scheduler:
            if args.scheduler == 'reduce_plateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.args.reduce_lr_factor, patience=self.args.reduce_lr_patience, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
            elif args.scheduler == 'one_cycle':
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, args.lr, epochs=args.n_epochs, steps_per_epoch=len(self.train_loader), pct_start=self.args.cycle_pct, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=100.0, three_phase=False, last_epoch=-1, verbose=True)

        self.sampling_trainloader = DataLoader(self.train_data, batch_size = args.sampling_batch_size, shuffle=True, num_workers = args.num_workers)
        self.sampling_valloader = DataLoader(self.val_data, batch_size = args.sampling_batch_size, shuffle=True, num_workers = args.num_workers)
        self.sampling_testloader = DataLoader(self.test_data, batch_size = args.sampling_batch_size, shuffle=True, num_workers = args.num_workers)
        self.sloader = {'train':self.sampling_trainloader, 'valid':self.sampling_valloader, 'test': self.sampling_testloader}

        if args.test_model:
            fname = "test_log"
            if os.path.exists(args.load_mdir + "/" + fname + ".txt"):
                fno = 1
                fnamet = fname + "_" + str(fno)
                while os.path.exists(args.load_mdir+ "/" + fnamet + ".txt"):
                    fno += 1
                    fnamet = fname + "_" + str(fno)
                fname += "_" + str(fno)
            fname += ".txt"

            print(fname)

        self.log = create_logger(__name__, silent=False, to_disk=True,
                                 log_file=args.model_dir + (fname if args.test_model else "log.txt"))
        self.log.info("Setup trainer for %s" % args.dataset)

        tensorboard_log_dir = os.path.join(args.model_dir, 'tensorboard/tb_log.info')
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

        fname = args.dataset + "_stats_" + str(args.fid_layer) + ".npz"
        #if os.path.exists(fname):
        #    f = open(fname,'rb')
        #    fid_stats = np.load(f)
        #    self.fid_mean = fid_stats['mean']
        #    self.fid_covar = fid_stats['covar']
            # print(self.fid_mean.shape,self.fid_covar.shape)
        #else:
        self.fid_mean, self.fid_covar = self.get_representative_stats('train')
        #f = open(fname,'wb+')
        #    np.savez(f, mean=self.fid_mean,covar=self.fid_covar)
            # print(self.fid_mean.shape,self.fid_covar.shape)

        if self.args.ema_mu < 1:
            self.ema = EMA(model=self.model, decay=self.args.ema_mu)

        self.model.to(self.args.device)
            
            


    def get_representative_stats(self,split):
        
        self.inception.to(self.args.device)
        means = []
        covars = []
        acts = []
        up = torch.nn.Upsample(size=(299, 299), mode='bilinear')
        loader = DataLoader(self.data[split], batch_size = 8, shuffle=True, num_workers = args.num_workers)
        for i,batch in enumerate(loader):
            batch = batch.to(self.args.device)
            if self.args.dataset == 'mnist':
                #print(batch.shape)
                batch = torch.cat([batch,batch,batch],dim=1)
            if self.args.renormalize:
                batch = (batch - torch.min(batch,dim=0)[0])/(torch.max(batch,dim=0)[0] - torch.min(batch,dim=0)[0])
            batch = (batch * 2) - 1
            # if i%10==0:
            #     print(i)
            # if i>100:
            #     break
            batch = up(batch)
            fid_act,_ = self.inception(batch)
            acts.append(fid_act.detach().cpu())

        act = torch.cat(acts).numpy()
        
        print(np.unique(act))
        mean = np.mean(act,axis=0)
        covar = np.cov(act,rowvar=False)
        self.inception.to(torch.device('cpu'))
        # all_fid = torch.stack(all_fid).mean()
        return mean,covar

    def train_test(self, what, split, epoch):
        """
        Train the model
        """
        assert split in {'train','valid','test'}
        assert what  in {'train','eval'}

        if what == 'train':
            self.model.train()
        else:
            self.model.eval()

        data_loader = self.tloader[split]
        total_losses = []

        with (torch.enable_grad() if what=='train' else torch.no_grad()), tqdm(total=len(data_loader.dataset)) as progress:
          for i, batch in enumerate(data_loader):

            step = epoch * len(data_loader) + i
            batch = batch.to(self.args.device)

            noise_idx = torch.randint(0, len(self.args.noise_std), size=(batch.size(0),), device=batch.device)
            noise_levels = torch.tensor(self.args.noise_std).float()[noise_idx].view(batch.size(0),*([1]*len(batch.shape[1:])))
            noise_levels = noise_levels.to(self.args.device)

            noise = torch.randn_like(batch).to(self.args.device) * noise_levels #torch.normal(mean=0, std=noise_level, size=batch.shape).to(self.args.device)

            if self.args.min_max_normalize:
                batch = (2 * batch) - 1

            noisy_batch = batch + noise 

            image_diff = - noise

            if self.args.reweight:
                image_diff = image_diff / (noise_levels**2)

            if self.args.model_objective == 'score':
                print(torch.min(noisy_batch),torch.max(noisy_batch))
                energy_gradient = self.model(noisy_batch)  
                print(torch.min(energy_gradient),torch.max(energy_gradient))

            else:
                energy = self.model(noisy_batch)
                noisy_batch.requires_grad = True
                energy_gradient = torch.autograd.grad(outputs=energy.sum(), inputs=noisy_batch, create_graph=True)[0]

                self.writer.add_scalar(what + "_" + split + '-set/energy_batch', energy.mean(), step)

            self.writer.add_scalar(what + "_" + split + '-set/energy_gradient_norm_batch', torch.norm(energy_gradient.view(batch.size(0),-1),dim=1).mean(), step)

            if self.args.reweight:
                energy_gradient = energy_gradient/noise_levels

            energy_gradient = energy_gradient.view(energy_gradient.shape[0], -1)
            image_diff = image_diff.view(image_diff.shape[0],-1)

            if len(self.args.noise_std) > 1 and self.args.reweight:
                loss = (1/2.) * ((energy_gradient - image_diff)**2).sum(dim=-1) * noise_levels.squeeze() **2
                loss = loss.mean(dim=0)
            else:
                loss = (1/2.) * ((energy_gradient - image_diff)**2).sum(dim=-1).mean(dim=0)

            self.writer.add_scalar(what + "_" + split + '-set/loss_batch', loss, step)

            if self.args.penalty == 'contraction':
                penalty = (energy_gradient**2).sum(dim=-1).mean()
                loss += (self.args.plambda * penalty)              
                
                self.writer.add_scalar(what + "_" + split + '-set/penalty_batch', penalty, step)
                self.writer.add_scalar(what + "_" + split + '-set/lossp_batch', loss, step)
            

            if what == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.args.ema_mu < 1:
                self.model.to(torch.device('cpu'))
                self.ema.update(self.model)
                self.model.to(self.args.device)

            total_losses.append(loss.detach().cpu().numpy())
            progress.update(len(batch))
            progress.set_postfix(loss=loss.item())

            if self.args.scheduler:
                if self.args.scheduler == 'one_cycle':
                    self.scheduler.step()

        return np.mean(total_losses)    


    def training(self):

        best_loss = None
        patient_steps = 0
        if args.model_selection == 'sampling':
            best_fid = None
            best_iscore = None

        for epoch in range(self.args.n_epochs):
            
            self.log.info("\nepoch: " + str(epoch))

            train_loss = self.train_test('train', 'train', epoch)
            self.writer.add_scalar('train_train-set/epoch_loss', train_loss, epoch)

            eval_loss = self.train_test('eval', 'valid', epoch)
            self.writer.add_scalar('eval_val-set/epoch_loss', train_loss, epoch)

            if self.args.scheduler:
                if self.args.scheduler == 'reduce_plateau':
                    self.scheduler.step(eval_loss)

            self.log.info("\t\tTrain Loss: " + str(train_loss) + "\n\t\tVal Loss: " + str(eval_loss))

            if args.model_selection == 'sampling':
                iscore, fid = self.sampling('valid',args.selection_num_samples, model=(self.ema.shadow if self.args.use_ema else self.model))

                self.writer.add_scalar('eval_val-set/iscore', iscore, epoch)
                if best_iscore is None or best_iscore > iscore:
                    best_iscore = iscore
                    self.log.info("Best model found at epoch " + str(epoch) + " with eval inception score " + str(best_iscore))
                    torch.save(self.model.state_dict(), self.args.model_dir + "best_iscore.ckpt")
                    if self.args.ema_mu < 1: torch.save(self.ema.shadow.state_dict(), self.args.model_dir + "best_iscore_ema.ckpt")
                
                self.writer.add_scalar('eval_val-set/fid', fid, epoch)
                if best_fid is None or best_fid < fid:
                    best_fid = fid
                    self.log.info("Best model found at epoch " + str(epoch) + " with eval fid score " + str(best_fid))
                    torch.save(self.model.state_dict(), self.args.model_dir + "best_fid.ckpt")
                    if self.args.ema_mu < 1: torch.save(self.ema.shadow.state_dict(), self.args.model_dir + "best_fid_ema.ckpt")

                self.log.info("\t\tInception Score: " + str(iscore) + "\n\t\tFid Score: " + str(fid))
           

            if best_loss is None or eval_loss < best_loss:
                best_loss = eval_loss
                self.log.info("Best model found at epoch " + str(epoch) + " with eval loss " + str(best_loss))
                patient_steps = 0
                torch.save(self.model.state_dict(), self.args.model_dir + "best.ckpt")
                if self.args.ema_mu < 1: torch.save(self.ema.shadow.state_dict(), self.args.model_dir + "best_ema.ckpt")
            else:
                patient_steps += 1
                if patient_steps == self.args.early_stop_patience:
                    self.log.info("Early stoppin at epoch " + str(epoch) + " since eval loss " + str(best_loss) + " didn't improve for " + str(patient_steps) + " epochs")
                    break

            torch.save(self.model.state_dict(), self.args.model_dir + "epoch.ckpt")
            if self.args.ema_mu < 1: torch.save(self.ema.shadow.state_dict(), self.args.model_dir + "epoch_ema.ckpt")
            if self.args.optimizer=='adam':
                torch.save(self.optimizer.state_dict(), self.args.model_dir + "epoch_optimizer.ckpt")


        test_loss = self.train_test('eval', 'test', 0)
        self.log.info("Test Loss: " + str(test_loss))

    def testing(self):

        if self.args.load_mdir is None:
            print("load model directory is None")
            exit(0)
        else:
            ckpt = torch.load(args.load_mdir + "/" + args.ckpt_type + ("_ema" if self.args.use_ema else "") + ".ckpt")
            if self.args.change_keys:

                keys = self.model.state_dict().keys()
                values = ckpt.values()
                ckpt = OrderedDict(zip(keys,values))

            self.model.load_state_dict(ckpt)

        test_loss = self.train_test('eval', args.test_split, 0)
        self.log.info("Test Loss: " + str(test_loss))

        iscore, fid = self.sampling(args.test_split,self.args.ntest, model=self.model)
        self.log.info("Test Inception Score: " + str(iscore) + "\t FID Score: " + str(fid))

    def sampling(self, split, num_samples=5000, model=None):

        if self.args.use_ema and not self.args.test_model:
            self.ema.shadow.to(self.args.device)

        assert split in {'train','valid','test'}

        model.eval()

        data_loader = self.sloader[split]

        all_samples = []
        ns = 0

        self.log.info(str(self.args.sampling_strategy) + " Sampling\n")

        # with tqdm(total=num_samples) as progress:
        for i, batch in enumerate(data_loader):
            if num_samples is not None:
                ns += len(batch)

            # batch = batch.to(self.args.device)
            
            #torch.normal(mean=0,std=1,size=batch.size())#.to(self.args.device)

            if self.args.init_value == 'uniform':
                curr_batch = torch.rand(batch.shape)
                if self.args.min_max_normalize:
                    curr_batch = (2 * curr_batch) - 1

            self.log.info("batch: " + str(i))

            prev_batch = curr_batch.clone() + 100

            if self.args.sampling_strategy == 'ann_langevin':
                for noise_level in self.args.noise_std:
                    step_size = args.step_lr * ((noise_level**2)/(self.args.noise_std[-1]**2))
                    for step in range(self.args.max_step):
                        
                        noise = torch.randn_like(batch)

                        if self.args.model_objective == 'score':
                            energy_gradient = model(curr_batch.to(self.args.device)).detach().cpu()

                        else:
                            energy = model(curr_batch.to(self.args.device)).detach().cpu()
                            curr_batch.requires_grad = True
                            energy_gradient = torch.autograd.grad(outputs=energy, inputs=curr_batch,
                                                                    grad_outputs=torch.ones(energy.size()).to(self.args.device),
                                                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

                        if self.args.reweight:
                            energy_gradient = energy_gradient / noise_level

                        prev_batch = curr_batch.clone()

                  #      print(type(step_size),type(args.step_lr),type(noise_level),type(self.args.noise_std[-1]))
                        curr_batch = curr_batch + step_size * energy_gradient + ((2*step_size)**0.5)*noise

                        if self.args.clamp:
                            curr_batch = torch.clamp(curr_batch, (-1.0 if self.args.min_max_normalize else 0.0), 1.0)

                        grad_norm = torch.norm(energy_gradient.view(energy_gradient.size(0),-1),dim=1).mean()
                        image_norm = torch.norm(curr_batch.view(curr_batch.size(0),-1),dim=1).mean()
                        noise_norm = torch.norm(noise.view(noise.size(0),-1),dim=1).mean()
                        snr = ((step_size / 2.)**0.5) * grad_norm / noise_norm
                        grad_mean_norm = torch.norm(energy_gradient.mean(dim=0).view(-1)) ** 2 * noise_level ** 2

                        if step > 0 and step%self.args.sampling_log_freq==0:
                            self.log.info("Noise Level: " + str(noise_level) + "\tstep: " + str(step) + "\nPredicted gradient: " + str(grad_norm) + "\tImage Norm: " + str(image_norm) + "\tNoise norm: " + str(noise_norm) + "\nsnr: " + str(snr) + "\tgrad mean norm: " + str(grad_mean_norm) + "\nImage step Diff: " + str(((curr_batch - prev_batch)**2).mean()))

                    if self.args.denoise==2:
                        energy_gradient = model(curr_batch.to(self.args.device)).detach().cpu()
                        noise_level = float(noise_level)
                        if self.args.reweight:
                            energy_gradient = energy_gradient / noise_level
                        curr_batch = curr_batch + (noise_level**2) * energy_gradient  

            else:
                for step in range(self.args.max_step):# and ((curr_batch - prev_batch)**2).mean() > 1e-4: #torch.norm(curr_batch - batch ,dim=1).mean() > 0.01 and #step <= self.args.max_step:
                
                    noise_level = 1
                    noise = torch.randn_like(batch)
                    
                    if self.args.model_objective == 'score':
                        energy_gradient = model(curr_batch.to(self.args.device)).detach().cpu()

                    else:
                        energy = model(curr_batch)
                        curr_batch.requires_grad = True
                        energy_gradient = torch.autograd.grad(outputs=energy, inputs=curr_batch,
                                                                grad_outputs=torch.ones(energy.size()).to(self.args.device),
                                                                create_graph=True, retain_graph=True, only_inputs=True)[0]

                    
                    if self.args.reweight:
                        energy_gradient = energy_gradient / float(noise_level)

                    prev_batch = curr_batch.clone()

                    if self.args.step_grad_choice == 'rgrad':
                        step_grad = -(curr_batch - batch)
                        if self.args.reweight:
                            step_grad = step_grad/(float(noise_level)**2)
                    elif self.args.step_grad_choice == 'pgrad':
                        step_grad = energy_gradient


                    # print(torch.norm(step_grad.view(step_grad.size(0),-1),-1), torch.norm(energy_gradient.view(step_grad.size(0),-1),-1))


                    if self.args.sampling_strategy == 'vanilla':
                        curr_batch = curr_batch + self.args.step_lr * step_grad

                        grad_norm = torch.norm(energy_gradient.view(energy_gradient.size(0),-1),dim=1).mean()
                        image_norm = torch.norm(curr_batch.view(curr_batch.size(0),-1),dim=1).mean()
                        grad_mean_norm = torch.norm(energy_gradient.mean(dim=0).view(-1)) ** 2 * noise_level ** 2

                        if step > 0 and step%self.args.sampling_log_freq==0:
                            self.log.info("step: " + str(step) + "\nPredicted gradient: " + str(grad_norm) + "\tImage Norm: " + str(image_norm) + "\ngrad mean norm: " + str(grad_mean_norm) + "\nImage step Diff: " + str(((curr_batch - prev_batch)**2).mean()))

                    elif self.args.sampling_strategy == 'langevin':
                        curr_batch = curr_batch + (self.args.step_lr/2) * step_grad + ((2*self.args.step_lr)**0.5) * noise
                        
                        grad_norm = torch.norm(energy_gradient.view(energy_gradient.size(0),-1),dim=1).mean()
                        image_norm = torch.norm(curr_batch.view(curr_batch.size(0),-1),dim=1).mean()
                        noise_norm = torch.norm(noise.view(noise.size(0),-1),dim=1).mean()
                        snr = ((step_size / 2.)**0.5) * grad_norm / noise_norm
                        grad_mean_norm = torch.norm(energy_gradient.mean(dim=0).view(-1)) ** 2 * noise_level ** 2

                        if step > 0 and step%self.args.sampling_log_freq==0:
                            self.log.info("step: " + str(step) + "\nPredicted gradient: " + str(grad_norm) + "\tImage Norm: " + str(image_norm) + "\tNoise norm: " + str(noise_norm) + "\nsnr: " + str(snr) + "\tgrad mean norm: " + str(grad_mean_norm) + "\nImage step Diff: " + str(((curr_batch - prev_batch)**2).mean()))
           
                    self.args.step_lr = self.args.step_lr * self.args.lr_anneal

                    if self.args.clamp:
                        self.log.info(str(torch.unique(curr_batch)))
                        curr_batch = torch.clamp(curr_batch, (-1.0 if self.args.min_max_normalize else 0.0), 1.0)
                        self.log.info(str(torch.unique(curr_batch))) 

            if self.args.denoise==1:
                energy_gradient = model(curr_batch.to(self.args.device)).detach().cpu()
                noise_level = float(noise_level)
                if self.args.reweight:
                    energy_gradient = energy_gradient / noise_level
                curr_batch = curr_batch + (noise_level**2) * energy_gradient  

                    
                    

            if self.args.clamp:
                self.log.info(str(torch.unique(curr_batch)))
                curr_batch = torch.clamp(curr_batch,(-1.0 if self.args.min_max_normalize else 0.0),1.0)
                self.log.info(str(torch.unique(curr_batch)))
            all_samples.append(curr_batch.detach())
            # norm_batch = (curr_batch - np.min(curr_batch))
            
            # progress.update(len(batch))
            
            if num_samples and ns >= num_samples:
                break

        all_samples = torch.cat(all_samples).detach()
        # print(all_samples.shape)

        idx = torch.randperm(all_samples.size(0))[:self.args.save_nsamples]
        save_samples = np.transpose(all_samples[idx].numpy(),(0,2,3,1))
        if self.args.min_max_normalize:
            save_samples = (save_samples + 1)/2

        for i in range(len(save_samples)):
            nsample = save_samples[i]*255
            cv2.imwrite((self.args.isave_dir if not self.args.test_model else self.args.tsave_dir) + str(i) + ".jpg", nsample)

        # print(torch.unique(all_samples))
        if self.args.renormalize:
            all_samples = (all_samples - torch.min(all_samples,dim=0)[0])/(torch.max(all_samples,dim=0)[0] - torch.min(all_samples,dim=0)[0])
        if not self.args.min_max_normalize:
            all_samples = (all_samples * 2) - 1
        mean_inception,std_inception,fid = inception_score(inception_model = self.inception.to(self.args.device), images=all_samples, cuda=True, fid_mean=self.fid_mean, fid_covar=self.fid_covar, mnist=(self.args.dataset=='mnist'))
        # print(mean_inception,std_inception,fid)

        if self.args.use_ema and not self.args.test_model:
            self.ema.shadow.to(torch.device('cpu'))

        return mean_inception,fid
            

if __name__ == '__main__':
    args = process_args()
    trainer = Trainer(args)
    if not args.test_model:
        trainer.training()
    else:
        trainer.testing()
    # trainer.sampling('test')




                





