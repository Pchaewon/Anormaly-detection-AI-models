from dataloader import Dataloader
import model.CNN as CNN
import model.autoencoder as autoencoder
from model.UNet import UNet
import trainer
from model.diffusion import DiffusionModel
from model.core.solver import get_beta_schedule

import copy
from torch import optim
import torch
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F
import torch


# main fun
def main(params):
    # 모델 정의
    MODEL=params.model
    
    # GPU 설정
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"]= "3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 로더 
    dloader = Dataloader(path_data='data/resized/')
    
    train_loader, test_loader=dloader.total_dataset()
    AE_train_loader, AE_test_loader=dloader.AE_total_dataset()
    
    # 모델
    if MODEL == 'CNN':
        model=CNN.CNN().to(device)
        lr = 0.0001
        num_epochs = 5
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_function = nn.CrossEntropyLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        parameter = {
        'num_epochs':num_epochs,
        'optimizer':optimizer,
        'loss_function':loss_function,
        'train_dataloader':train_loader,
        'test_dataloader': test_loader,
        'device':device}
    
        trainer.trainer(model).trainer(parameter)
    
    
    elif MODEL == 'AE':
        model=autoencoder.autoencoder().to(device)
        
        lr = 0.0001
        num_epochs = 5
        optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
        loss_function = nn.MSELoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        parameter = {
        'num_epochs':num_epochs,
        'optimizer':optimizer,
        'loss_function':loss_function,
        'train_dataloader':train_loader,
        'test_dataloader': test_loader,
        'device':device}
    
        trainer.trainer(model).AE_trainer(parameter)
    
    elif MODEL=='GAN':
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
                
        netG = AnoGAN.Generator().to(device)
        netD = AnoGAN.Discriminator().to(device)        
        netG.apply(weights_init)
        netD.apply(weights_init)
        
        lr = 0.0001
        num_epochs = 5
        optimizer_G = torch.optim.Adam(netG.parameters(),lr=lr,weight_decay=1e-5)
        optimizer_D = torch.optim.Adam(netD.parameters(),lr=lr,weight_decay=1e-5)
        loss_function = nn.BCELoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        parameter = {
        'num_epochs':num_epochs,
        'optimizer_G':optimizer_G,
        'optimizer_D':optimizer_D,
        'loss_function':loss_function,
        'train_dataloader':train_loader,
        'test_dataloader': test_loader,
        'abnormal_dataset':abnormal_dataset,
        'normal_dataset':normal_dataset,
        'device':device}
    
        trainer.trainer(model=(netG, netD)).GAN_trainer(parameter)
       
   # AnoDDPM
    elif MODEL == 'AnoDDPM':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # params
        ROOT_DIR = "./"
        IMG_SIZE = 256
        BASE_CHANNEL = 128
        DROPOUT = 0
        NUM_HEADS = 4
        NUM_HEAD_CHANNELS = 64
        T = 1000
        LEARNING_RATE = 0.0001
        EPOCHS = 5
        BETA_SCHEDULE = "linear"
        LOSS_TYPE="l2"
        NOISE_FN = "simplex" #guass
        ITER = 200
        TRAIN_START = True
        SAMPLE_DISTANCE = 800
        ARG_NUM = 1
        SAVE_IMGS = False
        SAVE_VIDS = True
        WEIGHT_DECAY = 0.0
        #betas = get_beta_schedule(T, BETA_SCHEDULE)        
                    
        model = UNet()
        model = torch.nn.DataParallel(model)
        diffusion = DiffusionModel()
        ema = copy.deepcopy(model)
        model.to(device)
        ema.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9,0.999), weight_decay=WEIGHT_DECAY)
        
    
        # params
        parameter = {
            'EPOCHS':EPOCHS, #num_epochs,
            'LOSS_TYPE':LOSS_TYPE,#loss_function,
            'train_dataloader':train_loader,
            'test_dataloader': test_loader,
            'device':device,
            'BETA_SCHEDULE': BETA_SCHEDULE,
            'NOISE_FN':NOISE_FN,
            'NUM_HEADS':NUM_HEADS,
            'NUM_HEADS_CHANNELS':NUM_HEAD_CHANNELS,
            'DROPOUT':DROPOUT,
            'T':T,
            'WEIGHT_DECAY':WEIGHT_DECAY,
            'BASE_CHANNEL':BASE_CHANNEL,
            'IMG_SIZE':IMG_SIZE,
            'ROOT_DIR':ROOT_DIR,
            'ITER':ITER,
            'TRAIN_START':TRAIN_START,
            'SAMPLE_DISTANCE':SAMPLE_DISTANCE,
            'ARG_NUM':ARG_NUM,
            'SAVE_IMGS':SAVE_IMGS,
            'SAVE_VIDS':SAVE_VIDS,
            'model':model,
            'diffusion':diffusion,
            'ema':ema,
            'optimizer':optimizer}        
        
        for i in ['./model/', "./diffusion-videos/", './diffusion-training-images/']:
            try:
                os.makedirs(i)
            except OSError:
                pass


        # make arg specific directories
        for i in [f'./model/diff-params-ARGS={ARG_NUM}',
                  f'./model/diff-params-ARGS={ARG_NUM}/checkpoint',
                  f'./diffusion-videos/ARGS={ARG_NUM}',
                f'./diffusion-training-images/ARGS={ARG_NUM}']:
            try:
                os.makedirs(i)
            except OSError:
                pass
        
        trainer.trainer(model).anoddpm_trainer(parameter)

    # remove checkpoints after final_param is saved (due to storage requirements)
    for file_remove in os.listdir(f'./model/diff-params-ARGS={ARG_NUM}/checkpoint'):
        os.remove(os.path.join(f'./model/diff-params-ARGS={ARG_NUM}/checkpoint', file_remove))
    os.removedirs(f'./model/diff-params-ARGS={ARG_NUM}/checkpoint')
       
# main.py 실행시만 작동     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CNN')
    params = parser.parse_args()
    
    main(params)