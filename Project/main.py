import numpy as np
import argparse, time, pickle, gc
from torch.utils.data import DataLoader

from data import pretrainDataset, finetuneDataset
from utils.helpers import timeSince, savePlot, load_loss, save_loss, stringArgToList
from model import NHR
home_path = '/home/aldosa/log6308/'
model_path = home_path + 'model/'
data_path = home_path + 'data/'

def trainIters(file_path, model, training_dataset, valid_dataset, finetune=False, n_epochs=10, iter=1, start_epoch=0, 
               print_every=50, plot_every=20, save_every=101, batch_size=64):
    
    start = time.time()
    total_iter = 0
    if iter == 0 and start_epoch == 0:
        save_loss(losses=[], name='finetune'if finetune else 'pretrain', loss_type='train', models_path=model_path)
        save_loss(losses=[], name='finetune'if finetune else 'pretrain', loss_type='valid', models_path=model_path)
    
    training_generator = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
    valid_generator = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(start_epoch, n_epochs+start_epoch):
        try:            
            print_loss = 0
            plot_loss = 0

            model.train()
            if finetune:
                if epoch != 0:
                    model.unfreeze_encoder()
                else:
                    model.freeze_encoder()
            
            for input_variable, target_variable in training_generator:
                if finetune:
                    loss =  model.finetune(input_variable, target_variable, train=True)
                else:
                    loss = model.pretrain(input_variable, target_variable, train=True)
                
                print_loss = (print_loss + loss ** 0.5) / 2
                plot_loss = (plot_loss + loss ** 0.5) / 2
                
                if iter % int(print_every/100*len(training_generator)) == 0:
                    print('%s (%6.f %3.f%%) | Training loss: %.4e' % 
                          (timeSince(start, total_iter / (len(training_generator)+len(valid_generator)) / n_epochs),
                           iter, 
                           iter / len(training_generator) * 100, 
                           print_loss))
                    print_loss = 0
                
                if iter % int(save_every/100*len(training_generator)) == 0:
                    model.save(model_path,epoch,iter)
                
                if iter % int(plot_every/100*len(training_generator)) == 0:
                    save_loss(loss=plot_loss, name='finetune'if finetune else 'pretrain', loss_type='train', models_path=model_path)
                    plot_loss = 0
                
                if iter == len(training_generator):
                    break
                
                iter += 1
                total_iter += 1
            
            iter=1
            print_loss = 0
            t0 = time.time()
            model.eval()
            
            for batch_no, (input_variable, target_variable) in enumerate(valid_generator):
                last_batch = batch_no == len(valid_generator) - 1
                if finetune:
                    loss =  model.finetune(input_variable, target_variable, train=False, last_batch=last_batch)
                else:
                    loss = model.pretrain(input_variable, target_variable, train=False, last_batch=last_batch)
                
                print_loss = (print_loss + loss ** 0.5) / 2
                
                if iter % len(valid_generator) == 0:
                    tf = time.time()
                    save_loss(loss=print_loss, name='finetune'if finetune else 'pretrain', loss_type='valid', models_path=model_path)
                    print('Validation loss: %.4e | Time/sample: %dms' % 
                          (print_loss, int((tf-t0)/len(valid_generator)/batch_size*1000)))
                
                iter += 1
                total_iter += 1
            
            iter=1
            if finetune:
                model.save(model_path,epoch+1,iter)
            gc.collect()
            
        except KeyboardInterrupt:
            print('User stopped training')
            break
            
    if not finetune:
        model.save(model_path,epoch+1,iter)
        
    savePlot('RMSE', model_path, name='finetune'if finetune else 'pretrain')
    return print_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune', help='finetune ?', action='store_true')
    parser.add_argument('--opt', help='search optimal parameters ?', action='store_false')
    parser.add_argument('--resume',help='resume ?', action='store_true')
    parser.add_argument('--encoder_layers',type=str,default='[1024, 1024, 128]',help='encoder layers ?')
    parser.add_argument('--encoder_dropout',type=float,default=0.2,help='encoder dropout ?')
    parser.add_argument('--encoder_lr',type=float,default=0.002,help='encoder learning rate ?')
    parser.add_argument('--encoder_lambda',type=float,default=1e-5,help='encoder weight decay ?')
    parser.add_argument('--colab_layers',type=str,default='[2028, 1024, 256]',help='colab layers ?')
    parser.add_argument('--colab_embed_mf',type=int,default=8,help='colab embedding size for matrix multiplication ?')
    parser.add_argument('--colab_embed_mlp',type=int,default=8,help='colab embedding size for mlp ?')
    parser.add_argument('--colab_dropout_mf',type=float,default=0.1,help='colab dropout rate for matrix multiplication?')
    parser.add_argument('--colab_dropout_mlp',type=float,default=0.1,help='colab dropout rate for mlp ?')
    parser.add_argument('--colab_lr',type=float,default=0.001,help='colab learning rate ?')
    parser.add_argument('--colab_lambda',type=float,default=1e-4,help='colab weight decay ?')
    parser.add_argument('--batch_size', type=int,default=512,help='batch size ?')
    parser.add_argument('--seed', type=int,default=1,help='seed ?')
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.finetune:
        if FLAGS.opt:
            file_path = data_path+"ratings.csv"
            save_every = 10
        else:
            file_path = data_path+"ratings_small.csv"
            save_every = 101
        training_dataset = finetuneDataset(file_path, data_path+"data.npy")
        valid_dataset = finetuneDataset(file_path, data_path+"data.npy", training=False)
        input_size, nb_users, nb_items = pickle.load(open('.'.join(file_path.split('.')[:-1]) + '/params.pkl', 'rb'))
        FLAGS.encoder_lr /= 2
        epochs = 2
        print_every = 1
        plot_every = 0.5
        
    else:
        file_path = data_path+"data.npy"
        training_dataset = pretrainDataset(file_path)
        valid_dataset = pretrainDataset(file_path,training=False)
        with open(file_path, 'rb') as file:
            input_size, nb_users, nb_items = pickle.load(file)['data'].shape[1],1,1
        epochs = 10
        print_every = 50
        plot_every = 20
        save_every = 9999999
    
    encoder_arch={'layers': stringArgToList(FLAGS.encoder_layers), 'dropout':FLAGS.encoder_dropout} 
    encoder_config = {'lr': FLAGS.encoder_lr, 'weight_decay': FLAGS.encoder_lambda}                    
    colab_arch={'layers':stringArgToList(FLAGS.colab_layers), 'embed_size_mf':FLAGS.colab_embed_mf, 
                'embed_size_mlp':FLAGS.colab_embed_mlp, 'dropout_rate_mf':FLAGS.colab_dropout_mf, 
                'dropout_rate_mlp':FLAGS.colab_dropout_mlp}
    colab_config = {'lr': FLAGS.colab_lr, 'weight_decay': FLAGS.colab_lambda}
    
    model = NHR(input_size=input_size, nb_users=int(nb_users), nb_items=int(nb_items), 
                encoder_arch=encoder_arch, encoder_config=encoder_config, 
                colab_arch=colab_arch, colab_config=colab_config)
    
    if FLAGS.resume:
        start_iter, start_epoch = model.load(model_path)
        training_dataset.resume(start_iter)
    else:
        start_iter, start_epoch = 1, 0
    
    print('start iter',start_iter)
    losses = trainIters(file_path, model, training_dataset, valid_dataset, finetune=FLAGS.finetune, 
                        iter=start_iter, start_epoch=start_epoch, n_epochs=epochs, batch_size=FLAGS.batch_size,
                        print_every=print_every, save_every=save_every, plot_every=plot_every)
