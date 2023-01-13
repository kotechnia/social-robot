import os
import glob
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from robust_optimization import RobustOptimizer
from emotion_dataset import EmotionDataset, emotion_transforms
import timm
import copy
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def label_smooth(target, n_classes: int, label_smoothing=0.1):
    # convert to one-hot
    batch_size = target.size(0)
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return soft_target

def cross_entropy_loss_with_soft_target(pred, soft_target):
    #logsoftmax = nn.LogSoftmax(dim=-1)
    return torch.mean(torch.sum(- weights*soft_target * torch.nn.functional.log_softmax(pred, -1), 1))

def cross_entropy_with_label_smoothing(pred, target):
    soft_target = label_smooth(target, pred.size(1)) #num_classes) #
    return cross_entropy_loss_with_soft_target(pred, soft_target)

def train(model, train_loader, valid_loader=None, n_epochs=3, learningrate=0.001, robust=False, device='cuda', criterion=None):

    if robust:
        optimizer = RobustOptimizer(filter(lambda p: p.requires_grad, model.parameters()), optim.Adam, lr=learningrate)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learningrate)
    
    
    best_acc=0
    best_model=None
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        model.train()
        for _, data, label, _, _ in tqdm(train_loader, desc='Training'):
            data = data.to(device)
            label = label.to(device)
            
            with torch.cuda.amp.autocast(enabled=True):
                output = model(data)
                
            loss = criterion(output, label)

            if robust:
                #optimizer.zero_grad()
                loss.backward()
                optimizer.first_step(zero_grad=True)
  
                # second forward-backward pass
                output = model(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = (output.argmax(dim=1) == label).float().sum()

            epoch_accuracy += acc
            epoch_loss += loss
        epoch_accuracy /= len(train_loader.sampler)
        epoch_loss /= len(train_loader.sampler)
        
        if valid_loader is not None:
            model.eval()
            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for _, data, label, _, _ in tqdm(valid_loader, desc='Validation'):
                    data = data.to(device)
                    label = label.to(device)

                    val_output = model(data)
                    val_loss = criterion(val_output, label)

                    acc = (val_output.argmax(dim=1) == label).float().sum()
                    epoch_val_accuracy += acc
                    epoch_val_loss += val_loss
            epoch_val_accuracy /= len(valid_loader.sampler)
            epoch_val_loss /= len(valid_loader.sampler)
            print(
                f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}"
            )
            if best_acc<epoch_val_accuracy:
                best_acc=epoch_val_accuracy
                best_model=copy.deepcopy(model.state_dict())

                torch.save(model, 'models/best_model.pt')
            #scheduler.step()
            
    return model


def main(args):

    emotaion_dataset_path = args.dataset
    model_save_path = args.model
    batch_size = int(args.batch_size)
    fine_epochs = int(args.fine_epochs)
    epochs = int(args.epochs)
    num_workers = int(args.num_workers)
    device = 'cuda'
    use_cuda = True
    pin_memory=True

    df_face_dataset = pd.read_csv(emotaion_dataset_path)
    df_face_dataset['image_path'] = df_face_dataset['image_path'].map(lambda x : x.replace('/mnt/hdd18t/', '/mnt/ssd4t/'))
    df_face_trainset = df_face_dataset[df_face_dataset['train_val_test'] == 'train'].reset_index(drop=True)
    df_face_validset = df_face_dataset[df_face_dataset['train_val_test'] == 'validation'].reset_index(drop=True)

    kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory} if use_cuda else {}

    train_dataset = EmotionDataset(data=df_face_trainset, transforms=emotion_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    valid_dataset = EmotionDataset(data=df_face_validset, transforms=emotion_transforms)
    valid_loader  = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, **kwargs) 

    (unique, counts) = np.unique(train_dataset.targets, return_counts=True)
    cw=1/counts
    cw/=cw.min()
    class_weights = {i:1. for i,cwi in zip(unique, cw)}
    num_classes=len(train_dataset.classes)
    global weights
    weights = torch.FloatTensor(list(class_weights.values())).cuda()

    criterion=cross_entropy_with_label_smoothing

    model=timm.create_model('tf_efficientnet_b0_ns', pretrained=False)
    model.classifier=torch.nn.Identity()
    model.load_state_dict(torch.load('models/pretrained_faces/state_vggface2_enet0_new.pt'))
    model.classifier=nn.Sequential(nn.Linear(in_features=1280, out_features=num_classes))
    model=model.to(device)

    #model = torch.load('models/emotion_model.pt')
    #model=model.to(device)

    #model=timm.create_model('tf_efficientnet_b2_ns', pretrained=False)
    #model.classifier=torch.nn.Identity()
    #model.load_state_dict(torch.load('models/pretrained_faces/state_vggface2_enet2.pt'))
    #model.classifier=nn.Sequential(nn.Linear(in_features=1408, out_features=num_classes))
    #model.to(device)

    set_parameter_requires_grad(model, requires_grad=False)
    set_parameter_requires_grad(model.classifier, requires_grad=True)
    model = train(model, train_loader, valid_loader, fine_epochs, 0.001, robust=True, device=device, criterion=criterion)

    set_parameter_requires_grad(model, requires_grad=True)
    model = train(model, train_loader, valid_loader, epochs, 1e-4, robust=True, device=device, criterion=criterion)

    torch.save(model, model_save_path)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--fine_epochs", default=3)
    parser.add_argument("--epochs", default=6)
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--num_workers", default=4)
    args = parser.parse_args()

    main(args)
