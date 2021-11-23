import time
import argparse
from copy import deepcopy
from progress.bar import IncrementalBar
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import utils as vutils
from itertools import chain
from gan.generator import MuseGenerator
from gan.critic import MuseCritic
from gan.utils import initialize_weights, Normal
from criterion import WassersteinLoss, GradientPenalty

def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def get_item(pred):
    return pred.mean().item()

dist = Normal()
class MuseGAN():
    def __init__(self,
                 c_dimension=10,
                 z_dimension=32,
                 g_channels=1024,
                 g_features=1024,
                 c_channels=128,
                 c_features=1024,
                 g_lr=0.001,
                 c_lr=0.001,
                 device='cpu'):
        
        self.c_dim = c_dimension
        self.z_dim = z_dimension
        self.device = device
        self.one_hot = True
        # generator and optimizer
        self.generator = MuseGenerator(c_dimension = c_dimension,
                                       z_dimension = z_dimension,
                                       hid_channels = g_channels,
                                       hid_features = g_features,
                                       out_channels = 1).to(device)
        self.generator = self.generator.apply(initialize_weights)
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(),
                                            lr=g_lr, betas=(0.5, 0.9))
        
        # critic and optimizer
        self.critic = MuseCritic(c_dimension = c_dimension,
                                 hid_channels = c_channels,
                                 hid_features = c_features,
                                 out_features = 1).to(device)
        self.critic = self.critic.apply(initialize_weights)
        self.c_optimizer = torch.optim.Adam(self.critic.parameters(),
                                            lr=c_lr, betas=(0.5, 0.9))

        
        self.opt_info = optim.Adam(chain(self.generator.parameters(), self.critic.pred_c.parameters()), lr=(g_lr+c_lr)/2, betas=(0.5, 0.99))

        self.running_avg_g = None 
        
        self.real_images = None
        self.prob_c = False
        self.recon_weight = 1.0
        self.onehot_weight = 1.0
        
        # loss functions and gradient penalty (critic is wasserstein-like gan)
        self.g_criterion = WassersteinLoss().to(device)
        self.c_criterion = WassersteinLoss().to(device)
        self.c_penalty = GradientPenalty().to(device)
        # dictionary to save history
        self.data = {'g_loss':[],
                     'c_loss':[],
                     'cf_loss':[],
                     'cr_loss':[],
                     'cp_loss':[],
                     'cs_loss':[],
                     'o_loss':[]}
        print('MuseGAN initialized.')


    def generate_random_sample(self, save_path, z=None, c=None, batch_size=16):
        backup_para = copy_G_params(self.generator)
        load_params(self.generator, self.running_avg_g)
        #self.generator.eval()
        if self.z_dim > 0 and z is None:
            z = torch.randn(batch_size, self.z_dim).to(self.device)
        if self.c_dim > 0 and c is None:
            c = torch.randn(batch_size, self.c_dim).uniform_(0,1).to(self.device)
            #c = torch.randn(1, self.c_dim).uniform_(0,1).repeat(batch_size,1).to(self.device)
        with torch.no_grad():
            g_img = self.generator(c=c, z=z).cpu()
            vutils.save_image(g_img.add_(1).mul_(0.5), save_path.replace(".jpg", "_random.jpg"), pad_value=0)
            del g_img
        #self.generator.train()
        load_params(self.generator, backup_para)

    def generate_each_dim(self, save_path, dim=0, z=None, c=None, num_interpolate=10, num_samples=8):
        #self.generator.eval()
        if self.running_avg_g is not None:
            backup_para = copy_G_params(self.generator)
            load_params(self.generator, self.running_avg_g)

        if self.z_dim > 0 and z is None:
            z = torch.randn(num_samples, self.z_dim).to(self.device)
            z = z.unsqueeze(1).repeat(1, num_interpolate, 1).view(-1, self.z_dim)
        if self.c_dim > 0 and c is None:
            c = torch.randn(num_samples, self.c_dim).uniform_(0.2, 0.6).to(self.device)
            c = c.unsqueeze(1).repeat(1, num_interpolate, 1)   #.view(-1, self.z_dim)
            c_line = torch.linspace(0, 1, num_interpolate).to(self.device)
            c_line = c_line.unsqueeze(0).repeat(num_samples, 1)
            c[:,:,dim] = c_line
            c = c.view(-1, self.c_dim)
        with torch.no_grad():
            g_img = self.generator(c=c, z=z)
            vutils.save_image(g_img.add_(1).mul_(0.5), \
                save_path.replace(".jpg", "_dim_%d.jpg"%(dim)), pad_value=0,nrow=num_interpolate)
            del g_img
        if self.running_avg_g is not None:
            load_params(self.generator, backup_para)

    def neg_log_density(self, sample, params):
        constant = torch.Tensor([np.log(2 * np.pi)]).to(self.device)
        mu = params[:,:self.c_dim]
        logsigma = params[:,self.c_dim:]
        inv_sigma = torch.exp(-logsigma)
        tmp = (sample - mu) * inv_sigma
        return 0.5 * (tmp * tmp + 2 * logsigma + constant)

    def sample_hot_c(self, batch_size, c_dim, num_hot=1):
        y_onehot = torch.zeros(batch_size, c_dim)
        # print("num_hot: ", num_hot)
        if num_hot==1:
            y = torch.LongTensor(batch_size, 1).random_() % c_dim
            # print("y_onehot: ", y_onehot.shape, y_onehot.dtype)
            # print("torch.ones_like(y_onehot).to(y_onehot): ", torch.ones_like(y_onehot).to(y_onehot).shape, torch.ones_like(y_onehot).to(y_onehot).dtype)
            # print("y: ", y.shape, y.dtype)
            # print("torch.ones_like(y): ", torch.ones_like(y).shape, torch.ones_like(y).dtype)
            # print("torch.ones_like(y).to(y): ", torch.ones_like(y).to(y).shape, torch.ones_like(y).to(y).dtype)
            y_onehot.scatter_(1, y, 1.0)
            # print("c_idx: ", y.view(-1).shape, y.view(-1, 1).shape, y.view([-1, 1]).shape)
            return y_onehot.to(self.device), y.to(self.device)
        else:
            for _ in range(num_hot):
                y = torch.LongTensor(batch_size,1).random_() % c_dim
                y_onehot.scatter_(1, y, torch.ones_like(y).to(y))
        return y_onehot.to(self.device)

    def sample_z_and_c(self, batch_size, n_iter):
        # sample z from Normal distribution
        z = None
        if self.z_dim > 0:
            z = torch.randn(batch_size, 10, self.z_dim).to(self.device)
        
        # sample c alternativaly from uniform and onehot
        c_idx = None
        if n_iter%4==0 and self.one_hot:
            c = torch.Tensor(batch_size, self.c_dim).uniform_(0.2,0.6).to(self.device)
            # chosen_section = np.random.randint(0, 10)
            choosen_dim = np.random.randint(0, self.c_dim)
            c[:, choosen_dim] = 1
            c_idx = torch.Tensor(batch_size).fill_(choosen_dim).long().to(self.device)
        elif n_iter%2==0 and self.one_hot:
            c, c_idx = self.sample_hot_c(batch_size, c_dim=self.c_dim, num_hot=1)
        else:
            c = torch.Tensor(batch_size, self.c_dim).uniform_(0, 1).to(self.device)
        return z, c, c_idx

    def compute_gradient_penalty(self, real_images, fake_images):
        # Compute gradient penalty
        # print("real_images, fake_images: ", real_images.shape, fake_images.shape)
        alpha = torch.rand(real_images.size(0), 1, 1, 1, 1).expand_as(real_images).to(self.device)
        interpolated = (alpha * real_images + (1 - alpha) * fake_images).clone().detach().requires_grad_(True)
        
        out = self.critic(interpolated)[0]
        
        exp_grad = torch.ones(out.size()).to(self.device)
        grad = torch.autograd.grad(outputs=out,
                                    inputs=interpolated,
                                    grad_outputs=exp_grad,
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)[0]
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
        return d_loss_gp

    def compute_total_correlation(self):
        real_images = torch.cat(self.real_images, dim=0)
        
        batch_size = real_images.size(0)
        # print(batch_size)
        self.critic.eval()
        with torch.no_grad():
            c_params = self.critic(real_images)[1]
        self.critic.train()

        sample_c = dist.sample(params=c_params.view(batch_size, self.c_dim, 2))
        _logqc = dist.log_density( sample_c.view(-1, 1, self.c_dim), c_params.view(1, -1, self.c_dim, 2) )

        logqc_prodmarginals = (logsumexp(_logqc, dim=1, keepdim=False) - math.log(batch_size)).sum(1)
        logqc = (logsumexp(_logqc.sum(2), dim=1, keepdim=False) - math.log(batch_size))
        
        #print( logqc, logqc_prodmarginals )
        self.real_images = None
        return (logqc - logqc_prodmarginals).mean().item()


    def train_step(self, real_image, n_iter):
        cfb_loss, crb_loss, cpb_loss, cb_loss = 0, 0, 0, 0
        if self.running_avg_g is None:
            self.running_avg_g = copy_G_params(self.generator)

        batch_size = real_image.size(0)
        c_ratio = 5
        for _ in range(c_ratio):
            ### prepare data part
            z, c, c_idx = self.sample_z_and_c(batch_size, n_iter)
            g_img = self.generator(c=c, z=z)        
            r_img = real_image.to(self.device)

            ### critic part
            self.critic.zero_grad()
            # pred_r, _ = self.critic(r_img)
            # pred_f, _ = self.critic(g_img.detach())

            # get critic's `fake` loss
            fake_pred, _ = self.critic(g_img)
            fake_target = - torch.ones_like(fake_pred)
            fake_loss = self.c_criterion(fake_pred, fake_target)
            # get critic's `real` loss
            real_pred, _ = self.critic(r_img)
            real_target = torch.ones_like(real_pred)
            real_loss = self.c_criterion(real_pred,  real_target)

            # mix `real` and `fake` melody
            realfake = self.alpha * r_img + (1. - self.alpha) * g_img
            # get critic's penalty
            realfake_pred, _ = self.critic(realfake)
            # print("realfake: ", realfake.shape)
            # print("realfake_pred: ", realfake_pred.shape)
            penalty = self.c_penalty(realfake, realfake_pred)

            # sum up losses
            closs = fake_loss + real_loss + 10 * penalty 
            # retain graph
            closs.backward(retain_graph=True)
            # update critic parameters
            self.c_optimizer.step()
            # devide by number of critic updates in the loop (5)
            cfb_loss += fake_loss.item()/c_ratio
            crb_loss += real_loss.item()/c_ratio
            cpb_loss += 10* penalty.item()/c_ratio
            cb_loss += closs.item()/c_ratio
        
        
        ### prepare data part
        z, c, c_idx = self.sample_z_and_c(batch_size, n_iter)
        g_img = self.generator(c=c, z=z)        
        r_img = real_image.to(self.device)

        ### Generator part
        self.generator.zero_grad()
        pred_g, _ = self.critic(g_img)
        loss_g = -pred_g.mean()
        loss_g.backward()
        self.g_optimizer.step()

        ### Mutual Information between c and c' Part
        self.generator.zero_grad()
        self.critic.zero_grad()

        z, c, c_idx = self.sample_z_and_c(batch_size, n_iter)
        g_img = self.generator(c=c, z=z)        
        pred_g, pred_c_params = self.critic(g_img)
        # print("c, c_pred: ", c.shape, pred_c_params.shape)
        if self.prob_c:
            loss_g_recon_c = self.neg_log_density(c, pred_c_params).mean()
        else:
            loss_g_recon_c = F.l1_loss(pred_c_params, c)

        loss_g_onehot = torch.Tensor([0]).to(self.device)
        # if n_iter%2==0 and self.one_hot:
        #     print("cp, c_idx: ", pred_c_params[:,:self.c_dim].shape, c_idx.shape)
        if n_iter%4==0 and self.one_hot:
            loss_g_onehot = 0.2*F.cross_entropy(pred_c_params[:,:self.c_dim], c_idx.view(-1))
        elif n_iter%2==0 and self.one_hot:
            loss_g_onehot = 0.8*F.cross_entropy(pred_c_params[:,:self.c_dim], c_idx.view(-1))
           
        loss_info = self.recon_weight * loss_g_recon_c + self.onehot_weight * loss_g_onehot
        loss_info.backward()

        self.opt_info.step()

        for p, avg_p in zip(self.generator.parameters(), self.running_avg_g):
            avg_p.mul_(0.999).add_(0.001, p.data)

        # print("losses: ", cfb_loss, crb_loss, cpb_loss, cb_loss, loss_g.item(), loss_g_onehot.item(), loss_g_recon_c.item())
        return cfb_loss, crb_loss, cpb_loss, cb_loss, loss_g.item(), loss_g_onehot.item(), loss_g_recon_c.item()
    


    def train(self,
              dataloader,
              epochs=500,
              batch_size=64,
              display_epoch=10,
              device='cpu'):
        # alpha parameter for mixing images
        self.alpha = torch.rand((batch_size, 1, 1, 1, 1)).requires_grad_().to(device)
        for epoch in range(epochs):
            ge_loss, ce_loss = 0, 0
            cfe_loss, cre_loss, cpe_loss, oe_loss, cse_loss = 0, 0, 0, 0, 0
            start = time.time()
            bar = IncrementalBar(f'[Epoch {epoch+1}/{epochs}]', max=len(dataloader))
            # print("epoch: ", epoch)
            for real in dataloader:
                # real: real image batch
                # print("real: ", real.shape)
                cfb_loss, crb_loss, cpb_loss, cb_loss, gb_loss, ob_loss, csb_loss = self.train_step(real, epoch)
                """
                    real = real.to(device)
                    # train Critic
                    cb_loss=0
                    cfb_loss, crb_loss, cpb_loss = 0, 0, 0
                    for _ in range(5):
                        # create random `noises`
                        cords = torch.randn(batch_size, 32).to(device)
                        style = torch.randn(batch_size, 32).to(device)
                        melody = torch.randn(batch_size, 4, 32).to(device)
                        groove = torch.randn(batch_size, 4, 32).to(device)
                        # forward to generator
                        self.c_optimizer.zero_grad()
                        with torch.no_grad():
                            fake = self.generator(cords, style, melody, groove).detach()


                        # get critic's `fake` loss
                        fake_pred, c_pred = self.critic(fake)
                        fake_target = - torch.ones_like(fake_pred)
                        fake_loss = self.c_criterion(fake_pred, fake_target)
                        # get critic's `real` loss
                        real_pred = self.critic(real)
                        real_target = torch.ones_like(real_pred)
                        real_loss = self.c_criterion(real_pred,  real_target)

                        # mix `real` and `fake` melody
                        realfake = self.alpha * real + (1. - self.alpha) * fake
                        # get critic's penalty
                        realfake_pred = self.critic(realfake)
                        penalty = self.c_penalty(realfake, realfake_pred)
                        # sum up losses
                        closs = fake_loss + real_loss + 10 * penalty 
                        # retain graph
                        closs.backward(retain_graph=True)
                        # update critic parameters
                        self.c_optimizer.step()
                        # devide by number of critic updates in the loop (5)
                        cfb_loss += fake_loss.item()/5
                        crb_loss += real_loss.item()/5
                        cpb_loss += 10* penalty.item()/5
                        cb_loss += closs.item()/5
                        
                    cfe_loss += cfb_loss/len(dataloader)
                    cre_loss += crb_loss/len(dataloader)
                    cpe_loss += cpb_loss/len(dataloader)
                    ce_loss += cb_loss/len(dataloader)
                    
                    # train generator
                    self.g_optimizer.zero_grad()
                    # create random `noises`
                    cords = torch.randn(batch_size, 32).to(device)
                    style = torch.randn(batch_size, 32).to(device)
                    melody = torch.randn(batch_size, 4, 32).to(device)
                    groove = torch.randn(batch_size, 4, 32).to(device)
                    # forward to generator
                    fake = self.generator(cords, style, melody, groove)
                    # forward to critic (to make prediction)
                    fake_pred = self.critic(fake)
                    # get generator loss (idea is to fool critic)
                    gb_loss = self.g_criterion(fake_pred, torch.ones_like(fake_pred))
                    gb_loss.backward()
                    # update critic parameters
                    self.g_optimizer.step()
                    ge_loss += gb_loss.item()/len(dataloader)
                """
                cfe_loss += cfb_loss/len(dataloader)
                cre_loss += crb_loss/len(dataloader)
                cpe_loss += cpb_loss/len(dataloader)
                ce_loss += cb_loss/len(dataloader)
                ge_loss += gb_loss/len(dataloader)
                oe_loss += ob_loss/len(dataloader)
                cse_loss += csb_loss/len(dataloader)


                bar.next()
            
            bar.finish()  
            end = time.time()
            tm = (end - start)
            # save history
            self.data['g_loss'].append(ge_loss)
            self.data['c_loss'].append(ce_loss)
            self.data['cf_loss'].append(cfe_loss)
            self.data['cr_loss'].append(cre_loss)
            self.data['cp_loss'].append(cpe_loss)
            self.data['cs_loss'].append(cse_loss)
            self.data['o_loss'].append(oe_loss)
            # display losses
            if epoch%10==0:
                print("[Epoch %d/%d] [G loss: %.3f] [D loss: %.3f] ETA: %.3fs" % (epoch+1, epochs, ge_loss, ce_loss, tm))
                print(f"[C loss | (fake: {cfe_loss:.3f}, real: {cre_loss:.3f}, penalty: {cpe_loss:.3f})]")
                print(f"[c similarity loss | {cse_loss:.3f}]")
                print(f"[onehot loss | {oe_loss:.3f}]")
        return self.generator
