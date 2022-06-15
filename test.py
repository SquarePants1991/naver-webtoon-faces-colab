import os
import cv2
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from tqdm import tqdm

def load_image(path, size):
    image = image2tensor(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

    w, h = image.shape[-2:]
    if w != h:
        crop_size = min(w, h)
        left = (w - crop_size)//2
        right = left + crop_size
        top = (h - crop_size)//2
        bottom = top + crop_size
        image = image[:,:,left:right, top:bottom]

    if image.shape[-1] != size:
        image = torch.nn.functional.interpolate(image, (size, size), mode="bilinear", align_corners=True)
    
    return image

def image2tensor(image):
    image = torch.FloatTensor(image).permute(2,0,1).unsqueeze(0)/255.
    return (image-0.5)/0.5

def tensor2image(tensor):
    tensor = tensor.clamp(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy()
    return tensor*0.5 + 0.5

def imshow(img, size=5, cmap='jet'):
    plt.figure(figsize=(size,size))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

def horizontal_concat(imgs):
    return torch.cat([img.unsqueeze(0) for img in imgs], 3) 

device = 'cuda:0'
image_size = 256
torch.set_grad_enabled(False)


from model import Encoder, Generator

ae_model_path = '/content/drive/MyDrive/train_models/001800.pt'
        
encoder = Encoder(32).to(device)
generator = Generator(32).to(device)

ckpt = torch.load(ae_model_path, map_location=device)
encoder.load_state_dict(ckpt["e_ema"])
generator.load_state_dict(ckpt["g_ema"])

encoder.eval()
generator.eval()

print(f'[SwapAE model loaded] {ae_model_path}')

from stylegan2.model import Generator as StyleGAN

stylegan_model_path = '/content/drive/MyDrive/train_models/800000.pt'
stylegan_ckpt = torch.load(stylegan_model_path, map_location=device)

latent_dim = stylegan_ckpt['args'].latent

stylegan = StyleGAN(image_size, latent_dim, 8).to(device)
stylegan.load_state_dict(stylegan_ckpt["g_ema"], strict=False)
stylegan.eval()
print(f'[StyleGAN2 generator loaded] {stylegan_model_path}\n')

truncation = 0.7
trunc = stylegan.mean_latent(4096).detach().clone()

num_samples = 8

latent = stylegan.get_latent(torch.randn(num_samples, latent_dim, device=device))
imgs_gen, _ = stylegan([latent],
                        truncation=truncation,
                        truncation_latent=trunc,
                        input_is_latent=True,
                        randomize_noise=True)

print("StyleGAN2 generated images:")
imshow(tensor2image(horizontal_concat(imgs_gen)), size=20)

structures, textures = encoder(imgs_gen)
recon_results = generator(structures, textures)

print("SwapAE reconstructions:")    
imshow(tensor2image(horizontal_concat(recon_results)), size=20)

print("Swapping results:")    
swap_results = generator(structures, textures[0].unsqueeze(0).repeat(num_samples,1))
imshow(tensor2image(horizontal_concat(swap_results)), size=20)



