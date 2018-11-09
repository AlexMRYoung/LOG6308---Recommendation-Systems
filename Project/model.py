# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 17:31:24 2018

@author: alexa
"""

#########################
### MODEL
#########################

import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()
print('Using cuda device ' + torch.cuda.get_device_name(0) if use_cuda else 'Not using cuda')

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).cuda()

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 


class textEncoder(nn.Module):
    def __init__(self, input_size, noise_sigma=0.3, 
                 layers=(512,256,128), dropout=0.9, batch_norm=False):
        super().__init__()
        self.noiseLayer = GaussianNoise(sigma=noise_sigma)
        
        self.layers = torch.nn.ModuleList()
        for i, size in enumerate(layers):
            if i == 0:
                self.layers.append(nn.Linear(input_size, size))
            else:
                self.layers.append(nn.Linear(layers[i-1], size))
            self.layers[-1].bias.data.fill_(0)
            nn.init.xavier_normal_(self.layers[-1].weight.data)
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(layers[i+1]))

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output
        

class textDecoder(nn.Module):
    def __init__(self, output_size, layers=(128,256,512), dropout=0.9, batch_norm=True):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i, size in enumerate(layers[:-1]):
            self.layers.append(nn.Linear(size, layers[i+1]))
            self.layers[-1].bias.data.fill_(0)
            nn.init.xavier_normal_(self.layers[-1].weight.data)
            self.layers.append(nn.ReLU())
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(layers[i+1]))
            self.layers.append(nn.Dropout(p=dropout))
        self.layers.append(nn.Linear(layers[-1], output_size))
        self.layers[-1].bias.data.fill_(0)
        nn.init.xavier_normal_(self.layers[-1].weight.data)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output


class userEncoder(nn.Module):
    def __init__(self, nb_users, layers=(64,), batch_norm=True, dropout= 0.5):
        super().__init__()
        
        self.layers = torch.nn.ModuleList()
        for i, size in enumerate(layers):
            if i == 0:
                self.layers.append(nn.Embedding(nb_users, size))
            else:
                self.layers.append(nn.Linear(layers[i-1], size))
                self.layers[-1].bias.data.fill_(0)
                nn.init.xavier_normal_(self.layers[-1].weight.data)
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(p=dropout))
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(size))

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output


class itemEncoder(nn.Module):
    def __init__(self, nb_items, encoder_text_size=64, layers_encoder=(64,), layers_dense=(64,), batch_norm=True, dropout= 0.5):
        super().__init__()
        
        self.layers_encoder = torch.nn.ModuleList()
        for i, size in enumerate(layers_encoder):
            if i == 0:
                self.layers_encoder.append(nn.Embedding(nb_items, size))
            else:
                self.layers_encoder.append(nn.Linear(layers[i-1], size))
                self.layers_encoder[-1].bias.data.fill_(0)
                nn.init.xavier_normal_(self.layers_encoder[-1].weight.data)
                self.layers_encoder.append(nn.ReLU())
                self.layers_encoder.append(nn.Dropout(p=dropout))
                if batch_norm:
                    self.layers_encoder.append(nn.BatchNorm1d(size))
        
        self.layers_dense = torch.nn.ModuleList()
        for i, size in enumerate(layers_dense):
            if i == 0:
                self.layers_dense.append(nn.Linear(encoder_text_size+layers_encoder[-1], size))
            else:
                self.layers_dense.append(nn.Linear(layers_dense[i-1], size))
            self.layers_dense[-1].bias.data.fill_(0)
            nn.init.xavier_normal_(self.layers_dense[-1].weight.data)
            self.layers_dense.append(nn.ReLU())
            self.layers_dense.append(nn.Dropout(p=dropout))
            if batch_norm:
                self.layers_dense.append(nn.BatchNorm1d(size))

    def forward(self, input, encoder_output):
        output = input
        for layer in self.layers_encoder:
            output = layer(output)
        
        output = torch.cat((output, encoder_output), dim=-1)
        for layer in self.layers_dense:
            output = layer(output)
        return output

class CDL:
    def __init__(self, input_size, nb_users, nb_items,
                 encoding_arch={'name': 'SDAE', 'layers':(256,128,64), 'lr':0.01, 'lambda_w':0.001, 'dropout':0.9},
                 colab_arch={'name':'classic', 'lr':0.01, 'lambda_u': 0.001, 'lambda_v': 0.001}):
        self.encoder = textEncoder(input_size, noise_sigma=0.3, 
                                   layers=encoding_arch['layers'], 
                                   dropout=encoding_arch['dropout'])
        self.decoder = textDecoder(input_size, 
                                   layers=encoding_arch['layers'][::-1], 
                                   dropout=encoding_arch['dropout'])
        self.userEncoder = userEncoder(nb_users)
        self.itemEncoder = itemEncoder(nb_items)
        
        if use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.itemEncoder = self.itemEncoder.cuda()
            self.userEncoder = self.userEncoder.cuda()
        
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), 
                                                  lr=encoding_arch['lr'], 
                                                  weight_decay=encoding_arch['lambda_w'])
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), 
                                                  lr=encoding_arch['lr'], 
                                                  weight_decay=encoding_arch['lambda_w'])
        self.itemEncoder_optimizer = torch.optim.Adam(self.itemEncoder.parameters(), 
                                                      lr=colab_arch['lr'], 
                                                      weight_decay=colab_arch['lambda_u'])
        self.userEncoder_optimizer = torch.optim.Adam(self.userEncoder.parameters(), 
                                                      lr=colab_arch['lr'], 
                                                      weight_decay=colab_arch['lambda_v'])
        
        scheduler_params = {'factor':0.5, 
                            'patience':2, 
                            'verbose':True, 
                            'threshold':0.0001, 
                            'threshold_mode':'rel', 
                            'cooldown':0, 
                            'min_lr':0, 
                            'eps':1e-08}
        
        self.encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.encoder_optimizer, **scheduler_params)
        self.decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, **scheduler_params)
        self.itemEncoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.itemEncoder_optimizer, **scheduler_params)
        self.userEncoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.userEncoder_optimizer, **scheduler_params)
        
        
    def pretrain(self, input_variable, target_variable, train=False, last_batch=False):
        if use_cuda:
            input_variable = input_variable.cuda()
            target_variable = target_variable.cuda()
            
        criterion = nn.MSELoss(reduction='elementwise_mean')

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        loss = 0
        
        encoder_outputs = self.encoder(input_variable)
        decoder_outputs = self.decoder(encoder_outputs)
        raw_loss = criterion(decoder_outputs, target_variable)

        loss = raw_loss

        if train:
            loss.backward()

    #         torch.nn.utils.clip_grad_norm(encoder.parameters(), grad_clip)
            self.encoder_optimizer.step()

    #         torch.nn.utils.clip_grad_norm(decoder.parameters(), grad_clip)
            self.decoder_optimizer.step()
        elif last_batch:
            self.encoder_scheduler.step(loss)
            self.decoder_scheduler.step(loss)

        return raw_loss.item()
    
    def finetune(self, input_variable, target_variable, train=False, last_batch=False):
        criterion = nn.MSELoss(reduction='elementwise_mean')
        text = input_variable[0]
        user_indexes = input_variable[1]
        item_indexes = input_variable[2]
        
        batch_size = user_indexes.shape[0]
        
        ratings = target_variable
        
        if use_cuda:
            text = text.cuda()
            user_indexes = user_indexes.cuda()
            item_indexes = item_indexes.cuda()
            ratings = ratings.cuda()
            print(text.shape, user_indexes.shape, item_indexes.shape, ratings.shape)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.userEncoder_optimizer.zero_grad()
        self.itemEncoder_optimizer.zero_grad()

        loss = 0
        
        encoder_outputs = self.encoder(text)
        user_embedding = self.userEncoder(user_indexes)
        item_embedding = self.itemEncoder(item_indexes, encoder_outputs)
        
        predictions = torch.bmm(user_embedding.view(batch_size,1,-1), 
                                item_embedding.view(batch_size,-1,1)).view(-1,1)
        
        raw_loss = criterion(predictions, ratings)

        loss = raw_loss

        if train:
            loss.backward()

    #         torch.nn.utils.clip_grad_norm(encoder.parameters(), grad_clip)
            self.encoder_optimizer.step()

    #         torch.nn.utils.clip_grad_norm(decoder.parameters(), grad_clip)
            self.decoder_optimizer.step()
        
    #         torch.nn.utils.clip_grad_norm(encoder.parameters(), grad_clip)
            self.itemEncoder_optimizer.step()

    #         torch.nn.utils.clip_grad_norm(decoder.parameters(), grad_clip)
            self.userEncoder_optimizer.step()
        elif last_batch:
            self.encoder_scheduler.step(loss)
            self.decoder_scheduler.step(loss)
            self.itemEncoder_scheduler.step(loss)
            self.userEncoder_scheduler.step(loss)
            
        return raw_loss.item()
    
    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.itemEncoder.train()
        self.userEncoder.train()
        
    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.itemEncoder.eval()
        self.userEncoder.eval()