# -*- coding: utf-8 -*-

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import time


model = models.vgg19(pretrained=True).features

device = torch.device("cuda")


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.req_features = ['0', '5', '10', '19', '28'] 
        self.model = models.vgg19(pretrained=True).features[:29]
    
   
    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if (str(layer_num) in self.req_features):
                features.append(x)
        return features


def image_loader(path):
    image = Image.open(path)
    check_img = np.array(image)
    if check_img.shape[2] == 4:
        temp = np.zeros((check_img.shape[0], check_img.shape[1], 3))
        temp[:, :, 0] = check_img[:, :, 0]
        temp[:, :, 1] = check_img[:, :, 1]
        temp[:, :, 2] = check_img[:, :, 2]
        image = Image.fromarray(np.uint8(temp))
    loader = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

    image = loader(image).unsqueeze(0)
    print(image.shape)
    return image.to(device, torch.float)


original_image = image_loader('./test/word_37.png')
style_image = image_loader('./test/ant.png')

generated_image = original_image.clone().requires_grad_(True)

def calc_content_loss(gen_feat, orig_feat):
    content_l = torch.mean((gen_feat - orig_feat) ** 2)
    return content_l

def calc_style_loss(gen, style):
    batch_size, channel, height, width = gen.shape

    G = torch.mm(gen.view(channel, height * width), gen.view(channel, height * width).t())
    A = torch.mm(style.view(channel, height * width), style.view(channel, height * width).t())

    style_l = torch.mean((G - A) ** 2)
    return style_l

def calculate_loss(gen_features, orig_feautes, style_featues):
    style_loss = content_loss = 0
    for gen, cont, style in zip(gen_features, orig_feautes, style_featues):
        content_loss += calc_content_loss(gen, cont)
        style_loss += calc_style_loss(gen, style)
    
    total_loss = alpha * content_loss + beta * style_loss 
    return total_loss


model = VGG().to(device).eval() 

epoch = 15000
lr = 0.004
alpha = 8
beta = 70

optimizer = optim.Adam([generated_image], lr=lr)

start = time.time()
for e in range (epoch):
    gen_features = model(generated_image)
    orig_feautes = model(original_image)
    style_featues = model(style_image)
    
    total_loss = calculate_loss(gen_features, orig_feautes, style_featues)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if(not(e % 100)):
        print(total_loss)
        
    if(not(e % 1000)):
        file_name = './gan_ant/gen_' + str(e) + '.png'
        save_image(generated_image, file_name)

end = time.time()

print("The time of execution of above program is :", (end-start) * 10**3, "ms")