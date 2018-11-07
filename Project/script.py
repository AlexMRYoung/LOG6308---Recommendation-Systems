import numpy as np
import pickle as pkl

# Loading data
encoding = 'utf-8'
path_to_data = './data/'
path = path_to_data+"dataEmbeded.pkl"
with open(path, 'rb') as pickler:
    data = pkl.load(pickler)


#########################
### TOKENIZER
#########################


from utils.tokenizer import tokenize_corpus
processed_data = tokenize_corpus(data[:,0], stop_words = False, BoW = True)

chunk_size = 5000
count = 0

nb_chunks = int(processed_data.shape[0]*0.8)//chunk_size

for chunk_nb in range(nb_chunks):
    path = './data/train-'+ str(chunk_nb) +'.npy'
    start = nb_chunks * chunk_nb
    end =  nb_chunks * (chunk_nb + 1) if (chunk_nb + 1) < nb_chunks else -1
    with open(path, 'wb') as file:
        pkl.dump(processed_data[start:end], file)

nb_chunks = int(processed_data.shape[0]*0.2)//chunk_size

for chunk_nb in range(nb_chunks):
    path = './data/test-'+ str(chunk_nb) +'.npy'
    start = nb_chunks * chunk_nb
    end =  nb_chunks * (chunk_nb + 1) if (chunk_nb + 1) < nb_chunks else -1
    with open(path, 'wb') as file:
        pkl.dump(processed_data[start:end], file)


#########################
### MODEL
#########################

import time, math, pickle, torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()

print('Using cuda device ' + torch.cuda.get_device_name(0) if use_cuda else 'Not using cuda')


path = './data/processed_data.npy'
with open(path, 'rb') as file:
    processed_data = pickle.load(file).toarray()
    
path = './data/processed_data.npy'
with open(path, 'rb') as file:
    processed_data = pickle.load(file).toarray()



class batchify:
    def __init__(self, data, bsz, training=False, split=0.8):
        data = data[int(len(data)*split):] if training else data[:int(len(data)*split)]
        self.training = training
        self.batches = []
        batch = []
        for line in data:
            if len(batch) != bsz:
                batch.append(line)
            else:
                self.batches.append(batch)
                batch = []

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        input_variable = Variable(torch.FloatTensor(self.batches[index]), require_grad=self.training)
        target_variable = Variable(torch.FloatTensor(self.batches[index]), require_grad=self.training)
        if use_cuda:
            input_variable = input_variable.cuda()
            target_variable = target_variable.cuda()
        return (input_variable, target_variable)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (-%s)' % (asMinutes(s), asMinutes(rs))

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%3.fm %2.fs' % (m, s)

def save_model(name,encoder,decoder,iteration,local=False):
    with open("{}/{}_iter.txt".format(models_path, name), 'w') as f:
        f.write(str(iteration))
    torch.save(encoder.state_dict(),"{}/{}_encoder.pt".format(models_path, name))
    torch.save(decoder.state_dict(),"{}/{}_decoder.pt".format(models_path, name))

def load_model(fname,model):
    loaded_state_dict = torch.load(models_path+'/'+fname, map_location=None if use_cuda else {'cuda:0':'cpu'})
    state_dict = model.state_dict()
    state_dict.update(loaded_state_dict)
    model.load_state_dict(loaded_state_dict)

def load_loss(name, loss_type): 
    losses = []
    try:
        with open("{}/{}_{}_losses.txt".format(models_path, name, loss_type), 'r') as f:
            for line in f:
                losses = [float(value) for value in line.split(";")]
                break
    except:
        print('No loss file')
    return losses

def save_loss(name, loss_type, losses):
    with open("{}/{}_{}_losses.txt".format(models_path, name, loss_type), 'w') as f:
        f.write(';'.join([str(value) for value in losses]))

def load_iter(name):
    try:
        with open("{}/{}_iter.txt".format(models_path, name), 'r') as f:
            for line in f:
                start_iter = int(line)
                break
    except:
        start_iter = 1
        print('error during downloading of file')
    return start_iter

def log_func(x, a, b, c):
    return a*x**2+b*x+c

def showPlot(points, interpol=False):
    fig, ax = plt.subplots(figsize=(20,15))
    # this locator puts ticks at regular intervals
    interval = (max(points)-min(points))/20
    loc = ticker.MultipleLocator(base=interval)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    if interpol:
        x = np.arange(1e-6, len(points[15:]))
        y = points[15:]
        popt, pcov = scipy.optimize.curve_fit(log_func, x, y, p0=(1, 1, 1))
        xx = np.linspace(1e-6, int(max(x)*1.5), 500)
        yy = log_func(xx, *popt)
        yy[0] = y[0]
        plt.plot(xx,yy)
    plt.grid()
    plt.show()


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
    def __init__(self, nb_users, layers=(128,), batch_norm=True, dropout= 0.5):
          
        self.layers = torch.nn.ModuleList()
        for i, size in layers:
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
    def __init__(self, nb_items, encoder_text_size=128, layers_encoder=(128,), layers_dense=(128,), batch_norm=True, dropout= 0.5):
          
        self.layers_encoder = torch.nn.ModuleList()
        for i, size in layers_encoder:
            if i == 0:
                self.layers_encoder.append(nn.Embedding(nb_items, size))
            else:
                self.layers_encoder.append(nn.Linear(layers[i-1], size))
                self.layers_encoder[-1].bias.data.fill_(0)
                nn.init.xavier_normal_(self.layers[-1].weight.data)
                self.layers_encoder.append(nn.ReLU())
                self.layers_encoder.append(nn.Dropout(p=dropout))
                if batch_norm:
                    self.layers_encoder.append(nn.BatchNorm1d(size))
        
        self.layers_dense = torch.nn.ModuleList()
        for i, size in layers_dense:
            if i == 0:
                self.layers_dense.append(nn.Linear(encoder_text_size+layers_encoder[-1], size))
            else:
                self.layers_dense.append(nn.Linear(layers_dense[i-1], size))
            self.layers_dense[-1].bias.data.fill_(0)
            nn.init.xavier_normal_(self.layers[-1].weight.data)
            self.layers.append_dense(nn.ReLU())
            self.layers.append_dense(nn.Dropout(p=dropout))
            if batch_norm:
                self.layers_dense.append(nn.BatchNorm1d(size))

    def forward(self, input, encoder_output):
        output = input
        for layer in self.layers_encoder:
            output = layer(output)
        
        output = torch.cat((output, encoder_output), dim=1)
        for layer in self.layers_dense:
            output = layer(output)
        return output



def trainAutoEncoder(X, encoder, decoder, itemEncoder=None, userEncoder=None, n_epochs=10, iter=1, start_epoch=1, local_save_every=1000, 
               print_every=10, plot_every=100, save_every=5000, batch_size=64, lr=0.01, lambda_u=0.001):
    start = time.time()
    total_iter = 0
    best_loss = 99999
    
    model = CDL()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=lambda_u)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr, weight_decay=lambda_u)
    
    if itemEncoder != None and userEncoder != None:
        itemEncoder_optimizer = optim.Adam(itemEncoder.parameters(), lr=lr, weight_decay=lambda_u)
        userEncoder_optimizer = optim.Adam(userEncoder.parameters(), lr=lr, weight_decay=lambda_u)
    
    training_generator = batchify(X, batch_size, finetune = itemEncoder == None)
    test_generator = batchify(X, batch_size, training=False, finetune = itemEncoder == None)
    
    no_improv = 0
    
    for epoch in range(n_epochs):
        try:
            plot_losses = []
            print_loss_total = []  # Reset every print_every
            plot_loss_total = []  # Reset every plot_every
            save_loss_total = []

            encoder.train()
            decoder.train()

            for input_variable, target_variable in training_generator:     
                if itemEncoder == None and userEncoder == None:
                    loss = model.pretrain(input_variable, target_variable, 
                                          encoder, encoder_optimizer, decoder, decoder_optimizer, 
                                          train=True)
                else:
                    loss =  model.finetune(input_variable, target_variable, 
                                          encoder, encoder_optimizer, decoder, decoder_optimizer, 
                                          userEncoder, userEncoder_optimizer, ItemEncoder, ItemEncoder_optimizer, 
                                          train=True)
                print_loss_total.append(loss)
                plot_loss_total.append(loss)
                save_loss_total.append(loss)

                if iter % print_every == 0 or iter == len(training_generator):
                    print_loss_avg = np.mean(print_loss_total)
                    print_loss_total = []
                    print('%s (%6.f %3.f%%) | Training loss: %.4e' % (timeSince(start, total_iter / len(training_generator) / n_epochs),
                                                 iter, iter / len(training_generator) * 100, print_loss_avg))

#                 if iter % plot_every == 0:
#                     plot_loss_avg = np.mean(plot_loss_total)
#                     train_losses = load_loss(hp.name, 'train')
#                     train_losses.append(plot_loss_avg)
#                     save_loss(hp.name, 'train', train_losses)
#                     plot_loss_total = []
                iter += 1
                total_iter += 1

                if iter >= len(training_generator):
                    break
            encoder.eval()
            decoder.eval()

            iter=1
            print_loss_total = []
            t0 = time.time()
            for input_variable, target_variable in test_generator:
                if itemEncoder == None and userEncoder == None:
                    loss = model.pretrain(input_variable, target_variable, 
                                          encoder, encoder_optimizer, decoder, decoder_optimizer, 
                                          train=False)
                else:
                    loss =  model.finetune(input_variable, target_variable, 
                                          encoder, encoder_optimizer, decoder, decoder_optimizer, 
                                          userEncoder, userEncoder_optimizer, ItemEncoder, ItemEncoder_optimizer, 
                                          train=False)
                
                print_loss_total.append(loss)
                if iter % len(test_generator) == 0:
                    tf = time.time()
                    print_loss_avg = np.mean(print_loss_total)
                    print_loss_total = []
                    print('Validation loss: %.4e | Time/sample: %dms' % (print_loss_avg, int((tf-t0)/len(test_generator)/batch_size*1000)))
                    
#                     valid_losses = load_loss(hp.name, 'valid')
#                     if len(valid_losses) < 2:
                    if epoch < 1:
#                         save_model(hp.name, encoder, decoder, iter)
                        pass
                    else:
                        if min(valid_losses[-2:]) < print_loss_avg :
                            no_improv += 1
                        else:
                            no_improv = 0
#                             save_model(hp.name, encoder, decoder, iter)
#                             print('Model Saved')
                    if no_improv > 1:
                        lr = encoder_optimizer.param_groups[0]['lr']
                        encoder_optimizer.param_groups[0]['lr'] = lr / 2
                        decoder_optimizer.param_groups[0]['lr'] = lr / 2
#                         print('No Improvement for 2 epoch, dividing the learning rate by 2')
#                     valid_losses.append(print_loss_avg)
#                     save_loss(hp.name, 'valid', valid_losses)
                iter += 1
                total_iter += 1
            iter=1
            
            if epoch+start_epoch >= n_epochs:
                break
        except KeyboardInterrupt:
            print('User stopped training')
            break



class CDL:
    def __init__(self):
        return 

    def pretrain(self, input_variable, target_variable, encoder, encoder_optimizer, decoder, decoder_optimizer, train=False):
        
        criterion = nn.MSELoss(reduction='elementwise_mean')

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss = 0

        encoder_outputs = encoder(input_variable)
        decoder_outputs = decoder(encoder_outputs)
        raw_loss = criterion(decoder_outputs, target_variable)

        loss = raw_loss

        if train:
            loss.backward()

    #         torch.nn.utils.clip_grad_norm(encoder.parameters(), grad_clip)
            encoder_optimizer.step()

    #         torch.nn.utils.clip_grad_norm(decoder.parameters(), grad_clip)
            decoder_optimizer.step()

        return raw_loss.item()
    
    def finetune(self, input_variable, target_variable, encoder, encoder_optimizer, decoder, decoder_optimizer, userEncoder, userEncoder_optimizer, ItemEncoder, ItemEncoder_optimizer, train=False)
        criterion = nn.MSELoss(reduction='elementwise_mean')
        text = input_variable[0]
        user_index = input_variable[1]
        item_index = input_variable[2]
        
        ratings = target_variable

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        userEncoder_optimizer.zero_grad()
        itemEncoder_optimizer.zero_grad()

        loss = 0

        encoder_outputs = encoder(input_variable)
        user_embedding = userEncoder(user_index)
        item_embedding = itemEncoder(item_index, encoder_outputs)
        
        predictions = torch.dot(user_embedding, item_embedding.t())
        
        raw_loss = criterion(predictions, ratings)

        loss = raw_loss

        if train:
            loss.backward()

    #         torch.nn.utils.clip_grad_norm(encoder.parameters(), grad_clip)
            encoder_optimizer.step()

    #         torch.nn.utils.clip_grad_norm(decoder.parameters(), grad_clip)
            decoder_optimizer.step()
        
    #         torch.nn.utils.clip_grad_norm(encoder.parameters(), grad_clip)
            itemEncoder_optimizer.step()

    #         torch.nn.utils.clip_grad_norm(decoder.parameters(), grad_clip)
            userEncoder_optimizer.step()
        
        return raw_loss.item()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datas', type=str,default=['./hpIntegral123.txt','./hpIntegral45.txt','./hpIntegral67.txt'],help='Directory for storing input data')
    parser.add_argument('--restore', type=str,default='n',help='restore ? (y/n)')
    parser.add_argument('--summaries_dir', type=str,default='./summaries',help='summary directory')
    FLAGS, unparsed = parser.parse_known_args()
    #FLAGS.datas
    encoder1 = textEncoder(input_size=processed_data.shape[1])
    decoder1 = textDecoder(output_size=processed_data.shape[1])
    if use_cuda:
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()
    start_iter = 1
    trainAutoEncoder(processed_data, encoder1, decoder1, 10, iter=start_iter)
















































