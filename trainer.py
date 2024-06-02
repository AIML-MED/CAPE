import sys
import pickle
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import copy
import random
import shutil
import datetime
import matplotlib.pyplot as plt
from path import Path
from torch.utils.tensorboard import SummaryWriter
from models.losses import SoftTargetKDLoss

import helpers

"""
Trainer class to train and evaluate the model
"""        
class Trainer:
    def __init__(self, opts, net, optimizer, scheduler, train_dataloader, test_dataloader):
        self.opts = opts
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader
        # Log
        if not self.opts['eval']:
            self.best_accuracy_orig, self.best_accuracy_cape, self.best_epoch = 0, 0, 0
            self.train_losses, self.test_losses = [], []
            self.train_accuracies_orig, self.test_accuracies_orig = [], []
            self.train_accuracies_cape, self.test_accuracies_cape = [], []
            self.writer = SummaryWriter(log_dir=self.opts['experiment_dir'])
        # Loss
        self.kl_loss = SoftTargetKDLoss(T=self.opts['T_kld']).to(self.opts['device'])
        self.classification_loss = nn.CrossEntropyLoss().to(self.opts['device'])
    
    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    
    def run(self):
        for epoch_no in range(self.opts['num_epochs']):
            disp_epoch_no = epoch_no + 1
            print('Learning Rate: {}'.format(self.optimizer.param_groups[0]['lr']))
            self.writer.add_scalar('Learning rate', self.optimizer.param_groups[0]['lr'], epoch_no)
            
            self.train(epoch_no)
            train_results = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {disp_epoch_no}/{self.opts['num_epochs']} Train Loss: {self.train_losses[-1]}, Train Accuracy [orig]: {self.train_accuracies_orig[-1]}  Train Accuracy [cape]: {self.train_accuracies_cape[-1]}"
            print(train_results)
    
            self.test(epoch_no)
            test_results = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {disp_epoch_no}/{self.opts['num_epochs']} Test Loss: {self.test_losses[-1]}, Test Accuracy [orig]: {self.test_accuracies_orig[-1]} Test Accuracy [cape]: {self.test_accuracies_cape[-1]}"
            print(test_results)
            # break
            
            # scheduler 
            if self.scheduler is not None:
                self.scheduler.step()
            
            if np.mod(disp_epoch_no, self.opts['save_interval']) == 0:
                if isinstance(self.net, nn.DataParallel):
                    net_state_dict = self.net.module.state_dict()
                else:
                    net_state_dict = self.net.state_dict()
                torch.save(net_state_dict, os.path.join(self.opts['experiment_dir'], 'net_e{}.ckpt'.format(disp_epoch_no)))

            if self.best_accuracy_orig <= self.test_accuracies_orig[-1] and self.best_accuracy_cape <= self.test_accuracies_cape[-1]:
                self.best_epoch = disp_epoch_no
                log = f"Improve accuracy from [orig]: {self.best_accuracy_orig} to {self.test_accuracies_orig[-1]} and [cape]: {self.best_accuracy_cape} to {self.test_accuracies_cape[-1]}"
                print(log)
                self.best_accuracy_orig = self.test_accuracies_orig[-1]
                self.best_accuracy_cape = self.test_accuracies_cape[-1]
                if isinstance(self.net, nn.DataParallel):
                    net_state_dict = self.net.module.state_dict()
                else:
                    net_state_dict = self.net.state_dict()
                torch.save(net_state_dict, os.path.join(self.opts['experiment_dir'], f'best.pth'))
        
        print(f"Best Accuracy [orig]: {self.best_accuracy_orig} and [cape]: {self.best_accuracy_cape} in epoch {self.best_epoch}")
        self.writer.close()

    
    def train(self, epoch):
        t = tqdm(enumerate(self.train_loader, 0), total=len(self.train_loader), 
                smoothing=0.9, position=0, leave=True, 
                desc="Train: Epoch: "+str(epoch+1)+"/"+str(self.opts['num_epochs']))
        
        self.net.train()
        
        if self.opts['settings'] == 'PF':
            # here batch normalization layers are now fully freezed by putting them in eval mode
            if isinstance(self.net, nn.DataParallel):
                self.net.module.net.eval()
            else:
                self.net.net.eval()
        
        num_total, acc_orig, acc_cape = 0, 0, 0
        running_total_loss, running_kld_loss, running_cls_loss = 0, 0, 0
        
        for i, (images, targets) in t:
            images = images.to(self.opts['device'])
            targets = targets.type(torch.LongTensor).to(self.opts['device'])
            
            self.optimizer.zero_grad()
            
            outputs = self.net(images)
            
            class_loss = self.opts['loss_alpha'] * self.classification_loss(outputs['orig']['outcome'], targets)
            kld_loss = self.opts['loss_beta'] * self.kl_loss(outputs['cape']['outcome_soft'], outputs['orig']['outcome'].detach())
            loss = class_loss + kld_loss
            
            loss.backward()
            self.optimizer.step()
            
            _, pred_orig = torch.max(outputs['orig']['outcome'].softmax(dim=1), 1)
            acc_orig += (pred_orig == targets).sum().item()
            _, pred_cape = torch.max(outputs['cape']['outcome'], 1)
            acc_cape += (pred_cape == targets).sum().item()
            num_total += targets.size(0)
            
            running_total_loss += loss.item()
            running_cls_loss += class_loss.item()
            running_kld_loss += kld_loss.item()
            
            t.set_postfix_str(f'class_loss: {class_loss.item():.4} kld_loss: {kld_loss.item():.4}') 
            
        
        self.train_accuracies_orig.append(100*acc_orig / num_total)
        self.train_accuracies_cape.append(100*acc_cape / num_total)
        self.train_losses.append(running_total_loss / len(self.train_loader))
        
        # save summary
        self.writer.add_scalar('Loss/train', running_total_loss / len(self.train_loader), epoch)
        self.writer.add_scalar('KLD Loss/train', running_kld_loss / len(self.train_loader), epoch)
        self.writer.add_scalar('CLS Loss/train', running_cls_loss / len(self.train_loader), epoch)
        self.writer.add_scalar('Accuracy [orig]/train', 100*acc_orig / num_total, epoch)
        self.writer.add_scalar('Accuracy [cape]/train', 100*acc_cape / num_total, epoch)
        
        
        
    
    def test(self, epoch):
        t = tqdm(enumerate(self.test_loader, 0), total=len(self.test_loader), 
                smoothing=0.9, position=0, leave=True, 
                desc="Test: Epoch: "+str(epoch+1)+"/"+str(self.opts['num_epochs']))
        
        self.net.eval()
        
        num_total, acc_orig, acc_cape = 0, 0, 0
        running_total_loss, running_kld_loss, running_cls_loss = 0, 0, 0
        
        with torch.no_grad():
            for i, (images, targets) in t:
                images = images.to(self.opts['device'])
                targets = targets.type(torch.LongTensor).to(self.opts['device'])
                
                outputs = self.net(images)
                
                class_loss = self.opts['loss_alpha'] * self.classification_loss(outputs['orig']['outcome'], targets)
                kld_loss = self.opts['loss_beta'] * self.kl_loss(outputs['cape']['outcome_soft'], outputs['orig']['outcome'].detach())
                loss = class_loss + kld_loss
                
                _, pred_orig = torch.max(outputs['orig']['outcome'].softmax(dim=1), 1)
                acc_orig += (pred_orig == targets).sum().item()
                _, pred_cape = torch.max(outputs['cape']['outcome'], 1)
                acc_cape += (pred_cape == targets).sum().item()
                num_total += targets.size(0)
                
                running_total_loss += loss.item()
                running_cls_loss += class_loss.item()
                running_kld_loss += kld_loss.item()
                # break

                
                t.set_postfix_str(f'class_loss: {class_loss.item():.4} kld_loss: {kld_loss.item():.4}') 
            
        self.test_accuracies_orig.append(100*acc_orig / num_total)
        self.test_accuracies_cape.append(100*acc_cape / num_total)
        self.test_losses.append(running_total_loss / len(self.test_loader))
        
        # save summary
        self.writer.add_scalar('Loss/test', running_total_loss / len(self.test_loader), epoch)
        self.writer.add_scalar('KLD Loss/test', running_kld_loss / len(self.test_loader), epoch)
        self.writer.add_scalar('CLS Loss/test', running_cls_loss / len(self.test_loader), epoch)
        self.writer.add_scalar('Accuracy [orig]/test', 100*acc_orig / num_total, epoch)
        self.writer.add_scalar('Accuracy [cape]/test', 100*acc_cape / num_total, epoch)
        
    def eval(self):
        t = tqdm(enumerate(self.test_loader, 0), total=len(self.test_loader), 
                smoothing=0.9, position=0, leave=True, 
                desc="Evaluation: ")
        
        self.net.eval()
        
        with torch.no_grad():
            num_acc_orig, num_acc_camp = 0, 0
            num_total = 0
            vanilla_sum, campe_sum = 0, 0
            for i, (images, targets) in t:
                images = images.to(self.opts['device'])
                num_total+=images.size(0)
                targets = targets.float().to(self.opts['device'])
                outputs = self.net(images)
                pred_vals, pred_orig = torch.max(outputs['orig']['outcome'].softmax(dim=1), 1)
                vanilla_sum+=pred_vals.sum().item()
                num_acc_orig += (pred_orig == targets).sum().item()
                pred_vals, pred = torch.max(outputs['cape']['outcome'], 1)
                campe_sum+=pred_vals.sum().item() 
                num_acc_camp += (pred == targets).sum().item()
            print(f'Net=> Acc orig: {round(num_acc_orig*100/num_total, 3)} Acc camp: {round(num_acc_camp*100/num_total, 3)}')
            # empirical mean of prediction confidence
            print(f'Net=> Avg. conf orig: {round(vanilla_sum*100/num_total, 3)} Avg. conf. camp: {round(campe_sum*100/num_total, 3)}')