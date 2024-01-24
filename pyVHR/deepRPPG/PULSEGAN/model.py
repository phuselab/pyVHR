import matplotlib.pyplot as plt
from torch import optim
import torch
import torch.nn as nn
from scipy.signal import welch, coherence
import random

import numpy as np



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # encoding layers
        self.conv1 = nn.Conv1d(1, 16, kernel_size=16, stride=1, padding=6)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=16, stride=1, padding=6)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=16, stride=1, padding=6)
        self.conv4 = nn.Conv1d(64, 96, kernel_size=16, stride=1, padding=6)
        self.conv5 = nn.Conv1d(96, 128, kernel_size=16, stride=1, padding=6)
        self.conv6 = nn.Conv1d(128, 156, kernel_size=16, stride=1, padding=6)
        

        # decoding layers
        self.deconv6 = nn.ConvTranspose1d(156, 128, kernel_size=16, stride=1, padding=6)
        self.deconv5 = nn.ConvTranspose1d(128, 96, kernel_size=16, stride=1, padding=6)
        self.deconv4 = nn.ConvTranspose1d(96, 64, kernel_size=16, stride=1, padding=6)
        self.deconv3 = nn.ConvTranspose1d(64, 32, kernel_size=16, stride=1, padding=6)
        self.deconv2 = nn.ConvTranspose1d(32, 16, kernel_size=16, stride=1, padding=6)
        self.deconv1 = nn.ConvTranspose1d(16, 1, kernel_size=16, stride=1, padding=6)
        
        # activations
        self.activ1 = nn.PReLU()
        self.activ2 = nn.Tanh()
        

    def forward(self, x):
        
        # encoder
        x = self.conv1(x)
        res1 = self.activ1(x)
        
        x = self.conv2(res1)
        res2 = self.activ1(x)
        
        x = self.conv3(res2)
        res3 = self.activ1(x)
 
        x = self.conv4(res3)
        res4 = self.activ1(x)
        
        x = self.conv5(res4)
        res5 = self.activ1(x)
        
        x = self.conv6(res5)
        x = self.activ1(x)   
        # print("weights: ", self.deconv1.weight.data.cpu().numpy())
        #print("shape between encoder and decoder is : ", x.shape)
        
        # decoder
        
        x = self.deconv6(x)
        x = self.activ1(x)

        x += res5
        
        x = self.deconv5(x)
        x = self.activ1(x)
        x += res4
        
        x = self.deconv4(x)
        x = self.activ1(x)
        x += res3
        
        x = self.deconv3(x)
        x = self.activ1(x)
        x += res2
        
        x = self.deconv2(x)
        x = self.activ1(x)
        x += res1
        
        x = self.deconv1(x)
        x = self.activ2(x)
           
        return x
        
        


class Discriminator(nn.Module):
    '''Discrimator: Classifier CNN '''
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.convolutional_layers = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=16, stride=2, padding=6),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(16, 32, kernel_size=16, stride=2, padding=6),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),            
            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=6),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),            
            nn.Conv1d(64, 96, kernel_size=16, stride=2, padding=6),
            nn.BatchNorm1d(96),
            nn.LeakyReLU(0.1),
            nn.Conv1d(96, 128, kernel_size=16, stride=2, padding=6),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),            
            nn.Conv1d(128,156,kernel_size=16, stride=2, padding=6),
            nn.LeakyReLU(0.1)
        )

        self.embedding = nn.Sequential(
            nn.Conv1d(128, 128, 3, stride=1),
            nn.LeakyReLU(0.1)
        )
    
     
        self.fully_connected_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(156,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
        
    def forward(self, X):
        Xi = self.convolutional_layers(X)
        # Xi = self.embedding(Xi)
        proba = self.fully_connected_layers(Xi)
        return proba
        
        
class PulseGAN():
    def __init__(self, netG, netD, lr_g, lr_d, batch_size=16):
        self.netG = netG
        self.netD = netD
        self.batch_size = batch_size
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=self.lr_d, betas=(0.5,0.999))
        self.optimizer_G = optim.Adam(self.netG.parameters(), lr=self.lr_g, betas=(0.5,0.999))
        self.lambda_regularization = 10
        self.beta_regularization = 10
        
    def step_discriminator(self, Xc, X, is_training):
        # Xc gt signal window
        # X chrom signal window
        
        with torch.set_grad_enabled(is_training):
            self.netD.zero_grad()
            
            # discriminator recognizes clean signal as clean
            clean_output = self.netD(torch.cat((Xc, X), dim=1))
            clean_loss = torch.mean((clean_output - 1.0) ** 2)  # they must all be ones

            # discriminator recognize noisy signal as noisy
            generator_output = self.netG(X)
            noisy_pair = torch.cat((generator_output.detach(), X), dim=1)
            noisy_output = self.netD(noisy_pair)

            noisy_loss = torch.mean(noisy_output**2) # they must all be zeros
            discriminator_loss = 0.5 * (clean_loss + noisy_loss)
            try:
                discriminator_loss.backward()
                self.optimizer_D.step()
            except:
                pass
            return clean_loss, noisy_loss

    def step_generator(self, Xc, X, is_training):
        with torch.set_grad_enabled(is_training):
            self.netG.zero_grad()
            generator_output = self.netG(X)
            noisy_pair = torch.cat((generator_output, X), dim=1)
            discriminator_output=self.netD(noisy_pair)
            idx = random.randint(0,self.batch_size-1)

            g_loss = 0.5 * torch.mean((discriminator_output - 1.0) ** 2)
            
            l1_distance = torch.abs(torch.add(generator_output, torch.neg(Xc)))

            generator_fft = torch.fft.rfft(generator_output, norm="ortho")
            gt_fft = torch.fft.rfft(Xc, norm="ortho")

            l1_spectrum_distance =  torch.abs(torch.add(generator_fft,torch.neg(gt_fft)))
            spectrum_loss = self.beta_regularization * torch.mean(l1_spectrum_distance)
            waveform_loss = self.lambda_regularization * torch.mean(l1_distance)
            
            generator_loss =  g_loss + waveform_loss + spectrum_loss
            try:
                generator_loss.backward()
                self.optimizer_G.step()
            except RuntimeError:
                pass
            return g_loss, waveform_loss, spectrum_loss
    def test_model(self, Xc, X, generator_output, fps):
        # taking welch transform of signals window
        fc,CHROM_w = welch(Xc.squeeze().cpu(), fs=fps.item(), nfft=2048)
        fg,gt_w = welch(X.squeeze().cpu(),fs=fps.item(),nfft=2048)
        fp, PulseGAN_w = welch(generator_output.detach().squeeze().cpu(),fs=fps.item(),nfft=2048)

        # taking welch peak in bpm
        bpm_chrom = fc[np.argmax(CHROM_w)]*60
        bpm_pulseGAN = fp[np.argmax(PulseGAN_w)]*60
        bpm_gt = fg[np.argmax(gt_w)]*60
        
        return bpm_pulseGAN, bpm_chrom, bpm_gt
    def generate_sample(self, X):
        pass
