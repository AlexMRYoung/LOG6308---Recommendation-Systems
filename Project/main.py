import numpy as np
from utils.helpers import timeSince
from model import NHR
import argparse, time
from torch.utils.data import DataLoader
from data import pretrainDataset, finetuneDataset

def trainIters(file_path, model, training_dataset,finetune=False, n_epochs=10, iter=1, start_epoch=1, local_save_every=1000, 
               print_every=10, plot_every=100, save_every=5000, batch_size=64):
    start = time.time()
    total_iter = 0
    
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
    return save_loss_total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune', type=bool,default=False,help='finetune ?')
    parser.add_argument('--pretrain', type=bool, default=True, help='pretrain ?')
    parser.add_argument('--opt', type=bool, default=True, help='search optimal parameters ?')
    parser.add_argument('--resume', type=bool,default=False,help='resume ?')
    FLAGS, unparsed = parser.parse_known_args()

#    with open('./data/ratings/chunk_259.npy', 'rb') as pickler:
#        nb_users = pickle.load(pickler)[-1][0]
#    print(nb_users)

    if FLAGS.finetune:
        training_dataset = finetuneDataset(file_path, "./data/data.npy", training=True)
        valid_dataset = finetuneDataset(file_path, "./data/data.npy", training=False)
        if FLAGS.opt:
            file_path = "./data/ratings_small.csv"
        else:
            file_path = "./data/ratings.csv"
        input_size, nb_users, nb_items = pickle.load('.'.join(file_path.split('.')[:-1]) + '/params.pkl')
    else:
        training_dataset = pretrainDataset(file_path)
        valid_dataset = pretrainDataset(file_path,training=False)
        file_path = "./data/data.npy"
        input_size, nb_users, nb_items = 1,1,1

    if FLAGS.opt and FLAGS.pretrain:
        colab_arch={'layers':[128,64,64], 'embed_size_mf':32, 'embed_size_mlp':32 ,
                'dropout_rate_mf':0.5, 'dropout_rate_mlp':0.5}
        colab_config = {'lr': 0.05, 'weight_decay': 0.001}
        curLoss = 9999999
        for layers in ((256,128,128),(256,128,64),(256,64,64),(128,64,64)):
            for dropout in (0.6,0.5,0.4):
                for lr in (0.1,0.05,0.01,0.005):
                    for weight_decay in (0.005,0.001):
                        encoder_arch={'layers':layers, 'dropout':dropout} 
                        encoder_config = {'lr': lr, 'weight_decay': weight_decay}
                        model = NHR(input_size=input_size, nb_users=nb_users, nb_items=nb_items, 
                                encoder_arch=encoder_arch, encoder_config=encoder_config, 
                                colab_arch=colab_arch, colab_config=colab_config)
                        start_iter = 1
                        losses = trainIters(file_path, model, finetune=FLAGS.finetune, iter=start_iter, n_epochs=2)
                        if np.mean(losses[-150:]) < curLoss:
                            curLoss = np.mean(losses[-150:])
                            pickle.dump((layers,dropout,lr,weight_decay),'./model/opt_params_pre.pkl')
    else if FLAGS.opt and FLAGS.finetune:
        layers, dropout, lr, weight_decay = pickle.load('./model/opt_params_pre.pkl')
        encoder_arch={'layers':layers, 'dropout':dropout} 
        encoder_config = {'lr': lr, 'weight_decay': weight_decay}
        curLoss = 9999999
        for layers in ((256,128,128),(256,128,64),(256,64,64),(128,64,64)):
            for dropout in (0.6,0.5,0.4):
                for lr in (0.1,0.05,0.01,0.005):
                    for weight_decay in (0.005, 0.001):
                        for embed_size_mf in (16,32,64):
                            for embed_size_mlp in (16,32,64):
                                colab_arch={'layers':layers, 'embed_size_mf':embed_size_mf, 'embed_size_mlp':embed_size_mlp ,
                                            'dropout_rate_mf':dropout, 'dropout_rate_mlp':dropout}
                                colab_config = {'lr': lr, 'weight_decay': weight_decay}
                                model = NHR(input_size=input_size, nb_users=nb_users, nb_items=nb_items, 
                                        encoder_arch=encoder_arch, encoder_config=encoder_config, 
                                        colab_arch=colab_arch, colab_config=colab_config)
                                start_iter = 1
                                losses = trainIters(file_path, model, finetune=FLAGS.finetune, iter=start_iter, n_epochs=2)
                                if np.mean(losses[-150:]) < curLoss:
                                    curLoss = np.mean(losses[-150:])
                                    pickle.dump((layers, dropout, lr, weight_decay, embed_size_mf, embed_size_mlp), './model/opt_params_fine.pkl')
                                    model.save('./model/')
    else:
        layers, dropout, lr, weight_decay = pickle.load('./model/opt_params_pre.pkl')
        layersF, dropoutF, lrF, weight_decayF,embed_size_mf,embed_size_mlp = pickle.load('./model/opt_params_fine.pkl')
        encoder_arch={'layers':layers, 'dropout':dropout} 
        encoder_config = {'lr': lr, 'weight_decay': weight_decay}                    
        colab_arch={'layers':layersF, 'embed_size_mf':embed_size_mf, 'embed_size_mlp':embed_size_mlp ,
            'dropout_rate_mf':dropoutF, 'dropout_rate_mlp':dropoutF}
        colab_config = {'lr': lrF, 'weight_decay': weight_decayF}
        model = NHR(input_size=input_size, nb_users=nb_users, nb_items=nb_items, 
        encoder_arch=encoder_arch, encoder_config=encoder_config, 
        colab_arch=colab_arch, colab_config=colab_config)
        if FLAGS.resume:
            model.load('./model/')
        start_iter = 1
        losses = trainIters(file_path, model, finetune=FLAGS.finetune, iter=start_iter)
    

