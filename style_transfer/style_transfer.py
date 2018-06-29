from __future__ import print_function, division
#  these imports are the same as pytorch tutorial imports 
import torch 
import torch.nn as nn
import torch.optim as optim
# lr_scheduler: learning rate scheduler
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os 
import copy
import PIL

image_size = 512 if torch.cuda.is_available() else 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = models.vgg16(pretrained=True).features.to(device).eval()
# vggleft = list(vgg16.features)
# vggmid = list(vgg16.features)
# vggright = list(vgg16.features)


loader = transforms.Compose(
    [transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor]
)
def image_loader(image):
    image = PIL.Image.open(image)
    image = loader(image).unsqueeze(0)


# normalize vgg 
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
class Normalization(nn.Module):
    def __init__(self,mean,std):
        super(Normalization,self).__init__()
        self.mean = torch.tensor(mean).view(-1,1,1)
        self.std = torch.tensor(std).view(-1,1,1)

    def forward(self,img):
        return (img-self.mean())/self.std

class Contentloss(nn.Module):
    def __init__(self,target):
        self.target = target.detach()
    
    def forward(self,input):
        self.loss = nn.MSELoss(input,self.target)
        
        return input

def sytle_loss_gram_matrix(input):
    a,b,c,d = input.size()
    features = input.view(a*b,c*d)
    G = torch.mm(features,features.t())
    return G.div(a*b*c*d)

class Styleloss(nn.Module):
    def __init__(self,target_feature):
        super(Styleloss,self).__init__()
        self.target =sytle_loss_gram_matrix(target_feature).detach()
        
style = image_loader('./style.jpg')
content = image_loader('./content.jpg')
# midimage = loader('./content.jpg')
content_layer_labels = ['conv_4']
style_layer_labels = ['conv_1','conv_2','conv_3','conv_4','conv_5']

def get_style_loss(vgg16,normalization_mean,normalization_std,style_img,content_img,content_layers = content_layer_labels,style_layers = style_layer_labels):
    vgg16 = copy.deepcopy(vgg16)
    normalization = Normalization(normalization_mean,normalization_std)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    i = 0 
    for layer in vgg16.children():
        if isinstance(layer,nn.Conv2d):
            i+=1
            name = 'conv_{}'.format(i)
        elif isinstance(layer,nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer,nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer,nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError("undefined layer in the VGG16")
        model.add_module(name,layer)
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = Contentloss(target)
            model.add_module('content_loss_{}'.format(i),content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = Styleloss(target_feature)
            model.add_module("style_loss_{}".format(i),style_loss)
            style_losses.append(style_loss)
    for i in range(len(model)-1,-1,-1):
        if isinstance(model[i],Contentloss) or isinstance(model[i],style_loss):
            break
    # trim the model until last content_loss or style_loss layer 
    
    model = model[:(i+1)]
    
    return model,style_losses,content_losses

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.require_grad()])
    return optimizer

def train(vgg,normalization_mean,normalization_std,style_img,content_img,input_image,num_steps= 200,style_weight =5,content_weight= 5):
    model,style_losses,content_losses = get_style_loss(vgg,normalization_mean,normalization_std,style_img,content_img)
    optimizer = get_input_optimizer(input_image)
    run = [0]
    while run[0]<=num_steps:
        def closure():
            input_image.data.clump_(0,1)
            optimizer.zero_grad()
            style_loss = 0
            content_loss = 0
            for sl in style_losses:
                style_loss += sl
            for cl in content_losses:
                content_loss += cl 
            style_loss *= style_weight
            content_loss *= content_weight
            loss = style_loss + content_loss
            loss.backward()
            run[0]+=1
            if run[0]%50==0:
                print('curr style and content losses are',style_loss.item(),content_loss.item())
            return style_loss+content_loss
        optimizer.step(closure)
    input_image.data.clump_(0,1)
    return input_image



loss = nn.MSELoss(size_average=True)


# print (device)
# Input = nn.Conv2d(in_channels=(1,3,224,224),out_channels= (1,3,224,224),kernel_size=(1,1),stride=1)
