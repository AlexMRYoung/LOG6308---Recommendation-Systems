import numpy as np
from utils.helpers import timeSince
from model import CDL
import argparse, time
from torch.utils.data import DataLoader
from data import pretrainDataset, finetuneDataset

def trainIters(file_path, model, finetune=False, n_epochs=10, iter=1, start_epoch=1, local_save_every=1000, 
               print_every=10, plot_every=100, save_every=5000, batch_size=64):
    start = time.time()
    total_iter = 0
    
    if finetune:
        training_dataset = finetuneDataset(file_path, "./data/data.npy", training=True)
        valid_dataset = finetuneDataset(file_path, "./data/data.npy", training=False)
    else:
        training_dataset = pretrainDataset(file_path)
        valid_dataset = pretrainDataset(file_path,training=False)
    training_generator = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_generator = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    for epoch in range(n_epochs):
        try:            
            plot_losses = []
            print_loss_total = []  
            plot_loss_total = []  
            save_loss_total = []

            model.train()
            for input_variable, target_variable in training_generator:
                if finetune:
                    loss =  model.finetune(input_variable, target_variable, train=True)
                else:
                    loss = model.pretrain(input_variable, target_variable, train=True)
                    
                print_loss_total.append(loss)
                plot_loss_total.append(loss)
                save_loss_total.append(loss)

                if iter % print_every == 0 or iter == len(training_generator):
                    print_loss_avg = np.mean(print_loss_total)
                    print_loss_total = []
                    print('%s (%6.f %3.f%%) | Training loss: %.4e' % (timeSince(start, total_iter / len(training_generator) / n_epochs),
                                                 iter, iter / len(training_generator) * 100, np.sqrt(print_loss_avg)))

#                if iter % plot_every == 0:
#                    plot_loss_avg = np.mean(plot_loss_total)
#                     train_losses = load_loss(hp.name, 'train')
#                     train_losses.append(plot_loss_avg)
#                     save_loss(hp.name, 'train', train_losses)
#                    plot_loss_total = []
                iter += 1
                total_iter += 1

#                if iter >= len(training_generator):
#                    break
            model.eval()

            iter=1
            print_loss_total = []
            t0 = time.time()
            
            batch_no = 0
            for input_variable, target_variable in valid_generator:
                last_batch = batch_no == len(valid_generator) - 1
                batch_no += 1
                if finetune:
                    loss =  model.finetune(input_variable, target_variable, train=False, last_batch=last_batch)
                else:
                    loss = model.pretrain(input_variable, target_variable, train=False, last_batch=last_batch)
                
                print_loss_total.append(loss)
                if iter % len(valid_generator) == 0:
                    tf = time.time()
                    print_loss_avg = np.mean(print_loss_total)
                    print_loss_total = []
                    print('Validation loss: %.4e | Time/sample: %dms' % (print_loss_avg, int((tf-t0)/len(valid_generator)/batch_size*1000)))
                iter += 1
                total_iter += 1
            iter=1
        except KeyboardInterrupt:
            print('User stopped training')
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datas', type=str,default=['./hpIntegral123.txt','./hpIntegral45.txt','./hpIntegral67.txt'],help='Directory for storing input data')
    parser.add_argument('--restore', type=str,default='n',help='restore ? (y/n)')
    parser.add_argument('--summaries_dir', type=str,default='./summaries',help='summary directory')
    FLAGS, unparsed = parser.parse_known_args()

#    with open('./data/ratings/chunk_259.npy', 'rb') as pickler:
#        nb_users = pickle.load(pickler)[-1][0]
#    print(nb_users)
    
    encoder_arch={'layers':(256,128,128), 'dropout':0.5} 
    encoder_config={'lr':0.1, 'weight_decay':0.001}
    colab_arch={'layers':[128,64,64], 'embed_size_mf':32, 'embed_size_mlp':32 ,
                'dropout_rate_mf':0.5, 'dropout_rate_mlp':0.5}
    colab_config={'lr':0.05, 'weight_decay':0.001}
    
    model = CDL(input_size=90366, nb_users=270895, nb_items=46000, 
                encoder_arch=encoder_arch, encoder_config=encoder_config, 
                colab_arch=colab_arch, colab_config=colab_config)
    
    start_iter = 1
    
    trainIters("./data/data.npy", model, finetune=False, iter=start_iter)
    trainIters("./data/ratings.csv", model, finetune=True, iter=start_iter)
    

