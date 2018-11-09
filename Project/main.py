import numpy as np
from utils.helpers import batchify, timeSince
from utils.tokenizer import tokenize_corpus
from model import CDL
import torch, pickle, argparse, os, time, random


def trainIters(X, model, finetune=False, n_epochs=10, iter=1, start_epoch=1, local_save_every=1000, 
               print_every=10, plot_every=100, save_every=5000, batch_size=64):
    start = time.time()
    total_iter = 0
    
    if finetune:
        ratings = X[1]
        X = X[0]
    
    indexes = list(range(len(ratings)))
    random.shuffle(indexes)
    
    X_train = X if finetune else X[indexes[:-len(indexes)//5]]
    X_valid = X if finetune else X[indexes[-len(indexes)//5:]]
    ratings_train = ratings[indexes[:-len(indexes)//5]] if finetune else None
    ratings_valid = ratings[indexes[-len(indexes)//5:]] if finetune else None
    
    training_generator = batchify((X_train, ratings_train), finetune, batch_size, training=True)
    valid_generator = batchify((X_valid, ratings_valid), finetune, batch_size)
    
    for epoch in range(n_epochs):
        try:            
            plot_losses = []
            print_loss_total = []  
            plot_loss_total = []  
            save_loss_total = []

            model.train()

            for input_variable, target_variable in training_generator: 
                print(target_variable)
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
                                                 iter, iter / len(training_generator) * 100, print_loss_avg))

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
    
    # Loading data
    encoding = 'utf-8'
    path_to_data = './data/'
    path = path_to_data+"dataEmbeded.pkl"
    
    if not os.path.isfile(path_to_data+'processed_data.npy'):
        with open(path, 'rb') as pickler:
            data = pickle.load(pickler)
        processed_data = tokenize_corpus(data[:,0], stop_words = False, BoW = True)
        path = path_to_data+'processed_data.npy'
        with open(path, 'wb') as file:
            pickle.dump(processed_data, file)
    else:
        path = './data/processed_data.npy'
        with open(path, 'rb') as file:
            processed_data = pickle.load(file).toarray()
            
    import csv
    ratings = np.zeros((1000,3))
    
    with open('./data/ratings_small.csv', 'r', newline='', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for i, line in enumerate(csv_reader):
            if i != 0:
                ratings[i][0] = int(line[0]) - 1
                ratings[i][1] = int(line[1]) - 1
                ratings[i][2] = line[2]
                if i == ratings.shape[0] - 1:
                    break
#    num_user = int( max(train_ratings[:,0].max(), test_ratings[:,0].max()) + 1 )
#    num_item = int( max(train_ratings[:,1].max(), test_ratings[:,1].max()) + 1 )
    
    i = [1,2,3,4]
    
#    print(processed_data[i])
    
    
    model = CDL(input_size=processed_data.shape[1], nb_users=1, nb_items=1)
    
    start_iter = 1
    
    trainIters((processed_data, ratings), model, finetune=True, iter=start_iter)
