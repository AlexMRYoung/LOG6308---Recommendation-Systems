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
    def __init__(self, input_size, layers, 
                 dropout=0.5, noise_sigma=0.3, batch_norm=True):
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
                self.layers.append(nn.BatchNorm1d(size))

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output
        

class textDecoder(nn.Module):
    def __init__(self, output_size, layers, dropout=0.5, batch_norm=True):
        super().__init__()
        layers = layers[::-1]
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

class neuralCollaborativeFilter(nn.Module):
    def __init__(self, nb_users, nb_items, item_feat_size, layers, embed_size_mf, embed_size_mlp, dropout_rate_mf, dropout_rate_mlp, batch_norm=True):
        super(neuralCollaborativeFilter, self).__init__()
        self.dropout_rate_mf = dropout_rate_mf
        self.dropout_rate_mlp = dropout_rate_mlp
        
        layers = [2 * (embed_size_mlp+item_feat_size)] + layers
    
        #mf part
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=nb_users, embedding_dim=embed_size_mf+item_feat_size)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=nb_items, embedding_dim=embed_size_mf)
        self.embedding_mf_dropout = torch.nn.Dropout(dropout_rate_mf)
    
        #mlp part
        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=nb_users, embedding_dim=embed_size_mlp+item_feat_size)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=nb_items, embedding_dim=embed_size_mlp)
        self.embedding_mlp_dropout = torch.nn.Dropout(dropout_rate_mlp)
    
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(torch.nn.ReLU())
            self.fc_layers.append(torch.nn.Dropout(dropout_rate_mlp))
            if batch_norm:
                self.fc_layers.append(nn.BatchNorm1d(out_size))
        
        self.logits = torch.nn.Linear(in_features=layers[-1] + embed_size_mf + item_feat_size, out_features=1)
        
    def forward(self, user_indices, item_indices, item_feat):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)
        
        #### dropout embeddings
        user_embedding_mlp = self.embedding_mlp_dropout(user_embedding_mlp)
        item_embedding_mlp = self.embedding_mlp_dropout(item_embedding_mlp)
        user_embedding_mf = self.embedding_mf_dropout(user_embedding_mf)
        item_embedding_mf = self.embedding_mf_dropout(item_embedding_mf)
        
        #### item features concatenation
        item_embedding_mlp = torch.cat([item_embedding_mlp, item_feat], dim=-1)
        item_embedding_mf = torch.cat([item_embedding_mf, item_feat], dim=-1)
    
        #### mf part
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)
        mf_vector = torch.nn.Dropout(self.dropout_rate_mf)(mf_vector)
    
        #### mlp part
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
    
        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
    
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.logits(vector)
        return logits

class NHR:
    def __init__(self, input_size, nb_users, nb_items,
                 encoder_arch, encoder_config,
                 colab_arch, colab_config,
                 seed=1):
        torch.manual_seed(seed)
        self.encoder = textEncoder(input_size, noise_sigma=0.3, **encoder_arch)
        self.decoder = textDecoder(input_size, **encoder_arch)
        self.NCF = neuralCollaborativeFilter(nb_users, nb_items, item_feat_size=encoder_arch['layers'][-1], **colab_arch)
        
        if use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.NCF = self.NCF.cuda()
        
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), **encoder_config)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), **encoder_config)
        self.NCF_optimizer = torch.optim.Adam(self.NCF.parameters(), **colab_config)
        
        scheduler_params = {'factor':0.5, 
                            'patience':2, 
                            'verbose':True, 
                            'threshold':0.0001, 
                            'threshold_mode':'rel', 
                            'cooldown':0, 
                            'min_lr':0, 
                            'eps':1e-08}
        
        self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=2, gamma=0.5)
        self.decoder_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=2, gamma=0.5)
        self.NCF_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.NCF_optimizer, **scheduler_params)
        
        
    def pretrain(self, input_variable, target_variable, train=False, last_batch=False):
        if use_cuda:
            input_variable = input_variable.cuda()
            target_variable = target_variable.cuda()
        if not train:
            input_variable = input_variable.detach()
            target_variable = target_variable.detach()
            
        criterion = nn.MSELoss()

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
            self.encoder_scheduler.step()
            self.decoder_scheduler.step()

        return raw_loss.item()
    
    def finetune(self, input_variable, target_variable, train=False, last_batch=False):
        criterion = nn.MSELoss()
        text = input_variable[0]
        user_indexes = input_variable[1]
        item_indexes = input_variable[2]
        
        ratings = target_variable.unsqueeze(1)
        
        if use_cuda:
            text = text.cuda()
            user_indexes = user_indexes.cuda()
            item_indexes = item_indexes.cuda()
            ratings = ratings.cuda()
        if not train:
            text = text.detach()
            user_indexes = user_indexes.detach()
            item_indexes = item_indexes.detach()
            ratings = ratings.detach()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.NCF_optimizer.zero_grad()

        loss = 0
        
        encoder_outputs = self.encoder(text)
        predictions = self.NCF(user_indexes, item_indexes, encoder_outputs)
        
        raw_loss = criterion(predictions, ratings)

        loss = raw_loss

        if train:
            loss.backward()

    #         torch.nn.utils.clip_grad_norm(encoder.parameters(), grad_clip)
            self.encoder_optimizer.step()

    #         torch.nn.utils.clip_grad_norm(decoder.parameters(), grad_clip)
            self.decoder_optimizer.step()
        
    #         torch.nn.utils.clip_grad_norm(encoder.parameters(), grad_clip)
            self.NCF_optimizer.step()
            
        elif last_batch:
#             self.encoder_scheduler.step(loss)
#             self.decoder_scheduler.step(loss)
            self.NCF_scheduler.step(loss)
            
        return raw_loss.item()
    
    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.NCF.train()
        
    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.NCF.eval()
        
    def freeze_encoder(self):
        for params in self.encoder.parameters():
            params.requires_grad = False
            
    def unfreeze_encoder(self):
        for params in self.encoder.parameters():
            params.requires_grad = True

    def save(self, PATH, epoch, iter):
        if epoch == 0:
            self.unfreeze_encoder()
        checkpoint = {
            'epoch':epoch,
            'iter':iter,
            'encoder':self.encoder.state_dict(),
            'decoder':self.decoder.state_dict(),
            'NCF':self.NCF.state_dict(),
            'encoder_optimizer':self.encoder_optimizer.state_dict(),
            'decoder_optimizer':self.decoder_optimizer.state_dict(),
            'NCF_optimizer':self.NCF_optimizer.state_dict(),
        }
        if epoch == 0:
            self.freeze_encoder()
        torch.save(checkpoint,PATH+'checkpoint.pkl')
        torch.save(checkpoint,PATH+'checkpoint_backup.pkl')

    def load(self,PATH):
        try:
            checkpoint = torch.load(PATH+'checkpoint.pkl')
        except:
            checkpoint = torch.load(PATH+'checkpoint_backup.pkl')
        
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.NCF.load_state_dict(checkpoint['NCF'])
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        self.NCF_optimizer.load_state_dict(checkpoint['NCF_optimizer'])
        
        return checkpoint['iter'], checkpoint['epoch']