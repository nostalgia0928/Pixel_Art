import collections
import datetime
import json
import math
import os
import random
import copy
import time
import tempfile
import subprocess
import torch
import torch.utils.data
import torchvision.transforms
import numpy as np
import visdom
import scipy
import einops
import json
from geomloss import SamplesLoss
from torch import nn, optim
from collections import defaultdict
from torch.optim.optimizer import Optimizer, required
from math import sqrt
from functools import partial, lru_cache
from torch.nn import functional as F
from torch.nn import Parameter
from PIL import Image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
vis = visdom.Visdom(server='http://ncc1.clients.dur.ac.uk', port=10086)

txt = ''
callback_text_window = vis.text(txt, win='quit', opts={'title': 'type here to stop training safely'})


def type_callback(event):
    if event['event_type'] == 'KeyPress':
        curr_txt = event['pane_data']['content']
        if event['key'] == 'Delete':
            curr_txt = txt
        elif len(event['key']) == 1:
            curr_txt += event['key']
        vis.text(curr_txt, win=callback_text_window)


vis.register_event_handler(type_callback, callback_text_window)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


args = {
    'width': 32,
    'dataset': 'easy_warrior',
    'n_channels': 3,   #default is 3 ; related to palettefile; will be changed when choose dataset
    'n_classes': 10,
    'batch_size': 16,  # default is 16
    'vid_batch': 16,
    'latent_dim': 8,  # lower is better modelling but worst interpolation freedom
    'lr': 0.005,
    'log_every': 2
}


def show_video(tensor, win="video", opts=None, num_channels=1):
    if num_channels == 1:
        fmt = "gray"
        b = (tensor * 255).byte().numpy().tobytes()
    else:
        fmt = "rgb24"
        b = tensor.permute(0, 3, 1, 2).contiguous()
        b = b.view(b.size(0), b.size(1) * b.size(2), b.size(3)).contiguous()
        b = b.permute(0, 2, 1).contiguous()
        b = (b * 255).byte().numpy().tobytes()

    videofile_fd = tempfile.NamedTemporaryFile(suffix='.mp4')
    videofile = videofile_fd.name

    subprocess.run(['/usr/bin/ffmpeg', '-s', '%dx%d' %
                    (tensor.shape[3], tensor.shape[2]),
                    '-f', 'rawvideo',
                    '-pix_fmt', fmt,
                    '-y', '-i', '-',
                    '-vcodec', 'h264',
                    '-pix_fmt', 'yuv420p',
                    '-c:v', 'libx264',
                    '-loglevel', 'panic',
                    videofile], input=b)

    return vis.video(videofile=videofile, win=win, opts=opts)

#Functions defined by myself
def getDatasetPalette(dataset, folder_path):
    _, folder_name = os.path.split(folder_path)
    parent_folder, _ = os.path.split(folder_path)  # save palette in parent folder
    color_folder_path = os.path.join(parent_folder, "palette_tensors")
    os.makedirs(color_folder_path, exist_ok=True)
    filename = f"{folder_name}.pt"

    file_path = os.path.join(color_folder_path, filename)
    if os.path.exists(file_path):
        # Load if exist
        unique_colors_tensor = torch.load(file_path)
        print("Loaded unique colors from file.")
    else:
        # if not exist, generate palette
        all_colors_list = []
        for rgb_image in dataset:
            reshaped_image = rgb_image.view(3, -1)
            all_colors_list.append(reshaped_image)

        all_colors_tensor = torch.cat(all_colors_list, dim=1)  # Concatenate the tensors
        unique_colors_tensor = torch.unique(all_colors_tensor, dim=1)  # Find unique colors

        torch.save(unique_colors_tensor, file_path)
        print("Saved unique colors to file.")

    print(unique_colors_tensor.size())
    num_colors = unique_colors_tensor.shape[1]
    print("Number of unique colors:", num_colors)
    return unique_colors_tensor


def rgb_to_palette(rgb_tensor, unique_colors):  # CHW
    num_classes = unique_colors.shape[1]

    rgb_tensor = rgb_tensor.to(device)
    unique_colors = unique_colors.to(device)

    reshaped_tensor = rgb_tensor.permute(1, 2, 0).contiguous().view(-1, rgb_tensor.shape[0])
    unique_colors = unique_colors.t()

    onehot = (reshaped_tensor[:, None, :] == unique_colors[None, :, :]).all(dim=2)
    onehot = onehot.view(rgb_tensor.shape[1], rgb_tensor.shape[2], num_classes).permute(2, 0, 1).float()

    return onehot

def palette_to_rgb(palette_tensor, unique_colors):  # CHW
    num_classes = unique_colors.shape[1]

    palette_tensor = palette_tensor.to(device)
    unique_colors = unique_colors.to(device)

    reshaped_palette = palette_tensor.permute(1, 2, 0).contiguous().view(-1, num_classes)
    reshaped_unique_colors = unique_colors.t()

    rgb_tensor = torch.matmul(reshaped_palette, reshaped_unique_colors)
    rgb_tensor = rgb_tensor.view(palette_tensor.shape[1], palette_tensor.shape[2], 3).permute(2, 0, 1)

    return rgb_tensor


def palette_to_softmax(palette_tensor):  # palette_tensor CHW
    channels, height, width = palette_tensor.shape

    reshaped_palette = palette_tensor.permute(1, 2, 0).contiguous().view(-1, channels)
    reshaped_palette = torch.softmax(reshaped_palette, dim=1)
    palette_tensor = reshaped_palette.view(height, width, channels).permute(2, 0, 1)

    # Test the softmax code
    print(palette_tensor[:, 30, 22])
    print(torch.sum(palette_tensor[:, 30, 22]))  # Inspect the probability mass function (PMF) for a specific pixel

    return palette_tensor


def softmax_to_palette(softmax_tensor):  # CHW
    channels, height, width = softmax_tensor.shape

    _, indices = torch.max(softmax_tensor, dim=0)
    palette_tensor = torch.zeros(channels, height, width, device=softmax_tensor.device)  # Ensure both on GPU
    palette_tensor.scatter_(0, indices.unsqueeze(0), 1)  # scatter(dim, index ,src)

    print(indices.shape)
    print(palette_tensor.shape)

    return palette_tensor

def tensor_to_image(input_tensor):
    # Clone the tensor and detach it from the current computation graph
    input_tensor = input_tensor.clone().detach()
    # Change the range of the tensor from [0, 1] to [0, 255]
    input_tensor = input_tensor * 255
    # Convert the tensor to uint8 type
    input_tensor = input_tensor.to(torch.uint8)
    # Move the tensor to CPU and convert it to numpy array
    input_numpy = input_tensor.cpu().numpy()
    # Convert the shape of the numpy array from (C, H, W) to (H, W, C)
    input_numpy = input_numpy.transpose((1, 2, 0))
    # Create PIL image from numpy array
    pil_image = Image.fromarray(input_numpy)
    return pil_image


class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
               'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
               'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
               'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard',
               'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle',
               'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
               'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon',
               'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
               'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
               'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
               'willow_tree', 'wolf', 'woman', 'worm', ]

if args['dataset'] == 'cifar100':
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100('data', train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))
        ])),
        shuffle=True, batch_size=1, drop_last=True)
    train_iterator = iter(cycle(train_loader))

if args['dataset'] == 'mnist':
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(3),
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: 1 - x)
        ])),
        shuffle=True, batch_size=args['batch_size'], drop_last=True)
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    train_iterator = iter(cycle(train_loader))

if args['dataset'] == 'fashion':
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST('data', train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(3),
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: 1 - x)
        ])),
        shuffle=True, batch_size=args['batch_size'], drop_last=True)
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                   "Ankle boot"]
    train_iterator = iter(cycle(train_loader))

if args['dataset'] == 'celeba':
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CelebA('data', download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize([args['width'], args['width']]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))
        ])),
        shuffle=True, batch_size=args['batch_size'], drop_last=True)
    train_iterator = iter(cycle(train_loader))

if args['dataset'] == 'cifar':
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))
        ])),
        shuffle=True, batch_size=args['batch_size'], drop_last=True)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']
    train_iterator = iter(cycle(train_loader))

if args['dataset'] == 'churches':
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.LSUN(
        "/home2/projects/cgw/lsun", classes=["church_outdoor_train"], transform=torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(args['width']),
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.Resize(args['width']),
            torchvision.transforms.ToTensor()])),
        shuffle=True, batch_size=args['batch_size'], drop_last=True)
    train_iterator = iter(cycle(train_loader))

if args['dataset'] == 'ffhq':
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        "/home2/projects/cgw/FFHQ-256", transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(args['width']),
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.ToTensor()])),
        shuffle=True, batch_size=args['batch_size'], drop_last=True)
    train_iterator = iter(cycle(train_loader))

if args['dataset'] == 'noise':
    def inf_dataset(batch_size):
        while True:
            yield torch.rand(batch_size, 1, 32, 32).to(device), torch.zeros(batch_size, 40).long()


    train_iterator = iter(cycle(inf_dataset(args['batch_size'])))


# dataset defined by my self
class WarriorDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_names = os.listdir(folder_path)
        if transform:
            self.transform = torchvision.transforms.Compose([
                #torchvision.transforms.Resize((32, 32)),    #default is 32
                torchvision.transforms.ToTensor(),
                #torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.folder_path, image_name)
        image = Image.open(image_path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

class PokemonDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_names = os.listdir(folder_path)
        if transform:
            self.transform = torchvision.transforms.Compose([
                #torchvision.transforms.Resize((32, 32)),    #default is 32
                torchvision.transforms.ToTensor(),
                #torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.folder_path, image_name)
        image = Image.open(image_path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

class TreesDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_names = os.listdir(folder_path)
        if transform:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((32, 32)),    #default is 32
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.folder_path, image_name)
        image = Image.open(image_path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image
if args['dataset'] == 'easy_warrior':
    folder_path = os.getcwd() + "/Pictures/Warrior"
    dataset = WarriorDataset(folder_path, transform=True)
    palettefile = getDatasetPalette(dataset, folder_path)
    args['n_channels'] = palettefile.shape[1]
    train_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True, batch_size=args['batch_size'], drop_last=True)
    train_iterator = iter(cycle(train_loader))

if args['dataset'] == 'intermediate_pokemon':
    folder_path = os.getcwd() + "/Pictures/Pokemon"
    dataset = PokemonDataset(folder_path, transform=True)
    palettefile = getDatasetPalette(dataset, folder_path)
    args['n_channels'] = palettefile.shape[1]
    train_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True, batch_size=args['batch_size'], drop_last=True)
    train_iterator = iter(cycle(train_loader))

if args['dataset'] == 'hard_trees':
    folder_path = os.getcwd() + "/Pictures/Trees"
    dataset = TreesDataset(folder_path, transform=True)
    palettefile = getDatasetPalette(dataset, folder_path)
    args['n_channels'] = palettefile.shape[1]
    train_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True, batch_size=args['batch_size'], drop_last=True)
    train_iterator = iter(cycle(train_loader))

print(f'> Size of training dataset {len( train_loader.dataset)}')


def ot_loss(x, y):
    return ot_loss_fn(x.contiguous().view(x.size(0), -1), y.contiguous().view(y.size(0), -1))


def lerp(a, b, t):
    return (1 - t) * a + t * b


def slerp(a, b, t):
    omega = torch.acos(
        (a / torch.norm(a, dim=1, keepdim=True) * b / torch.norm(b, dim=1, keepdim=True)).sum(1)).unsqueeze(1)
    res = (torch.sin((1.0 - t) * omega) / torch.sin(omega)) * a + (torch.sin(t * omega) / torch.sin(omega)) * b
    return res



class Decoder(nn.Module):
    def __init__(self, latent_dim, n_channels):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.LazyConvTranspose2d(512, 4, stride=1, padding=0),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),  # Output: [512, 4, 4]

            nn.LazyConvTranspose2d(256, 4, stride=2, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),  # Output: [256, 8, 8]

            nn.LazyConvTranspose2d(128, 4, stride=2, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),  # Output: [128, 16, 16]

            nn.LazyConvTranspose2d(64, 4, stride=2, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),  # Output: [64, 32, 32]

            nn.LazyConvTranspose2d(32, 4, stride=2, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),  # Output: [32, 64, 64]

            nn.LazyConvTranspose2d(n_channels, 4, stride=2, padding=1)
            #nn.Sigmoid()  # Output: [3, 128, 128]
        )

    def forward(self, z):
        x = self.decoder(z)
        # Crop from [3, 128, 128] to [3, 44, 69]
        x = x[:, :, :44, :69]
        return x

net = Decoder(args['latent_dim'], args['n_channels']).to(device)

opt = torch.optim.Adam(net.parameters(), lr=args['lr'])
ot_loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.001)

epoch = 0
start_time = time.time()
global_step = 0

vis.line(
    X=[0],
    Y=[[0, 0, 0]],
    win='losses',
    opts={'legend': ['loss1', 'loss2', 'loss3'], 'ytype': 'log'}
)


def sample_prior():
    p_z = torch.randn(args['batch_size'], args['latent_dim'], 1, 1).to(device)
    return p_z


def cost(x, y):
    return ((x - y) ** 2).sum(1).mean()


# grabs a batch of data from the dataset
#xb= next(train_iterator)
#xb= xb.to(device)

while (True):

    # # grabs a batch of data from the dataset
    if args['dataset'] == 'easy_warrior' or args['dataset'] == 'intermediate_pokemon' or args['dataset'] == 'hard_trees': # Test case
        xb = next(train_iterator)
        xb = xb.to(device)
        ##RGB to palette
        xb_processed_images = []
        for i in range(xb.shape[0]):     #The function receives CHW but xb is BCHW
            single_image = xb[i, :, :, :]
            processed_image = rgb_to_palette(single_image, palettefile)
            xb_processed_images.append(processed_image)
        xb = torch.stack(xb_processed_images, dim=0)   #Use stack to get BCHW; Now xb is a palette tensor
    else:
        xb,cb = next(train_iterator)
        xb,cb = xb.to(device), cb.to(device)

    # arrays for metrics
    logs = {}
    logs['loss1'] = logs['loss2'] = logs['loss3'] = 0
    logs['num_stats'] = 0

    opt.zero_grad()

    # p(x | z)
    # p(x | z, p)

    p_z = torch.randn(args['batch_size'], args['latent_dim'], 1, 1).to(device)
    g = net(p_z)
    g = torch.softmax(g, dim=1)
    # transform section:
    #g = batch_rgb_to_palette(g, unique_colors)
    #print(g.requires_grad)

    loss = ot_loss(g, xb)  # ((g-xb)**2).mean()
    #loss.requires_grad = True   #Idk why I have to set up this after using palette
    #print(loss.requires_grad)
    loss.backward()
    opt.step()
    #g back to rgb
    g_processed_images = []
    for i in range(g.shape[0]):  # The function receives CHW but g is BCHW
        single_image = g[i, :, :, :]
        processed_image = palette_to_rgb(single_image, palettefile)
        g_processed_images.append(processed_image)
    g= torch.stack(g_processed_images, dim=0)  # Use stack to get BCHW; Now g is a rgb tensor

    # accumulate statistics
    logs['loss1'] += loss.item()
    logs['loss2'] += loss.item()
    logs['loss3'] += loss.item()
    logs['num_stats'] += 1

    # update global step counter
    global_step += 1

    # log the loss value
    if global_step % args['log_every'] == 0:
        logs['loss1'] /= logs['num_stats']
        logs['loss2'] /= logs['num_stats']
        logs['loss3'] /= logs['num_stats']

        vis.line(
            X=[global_step],
            Y=[[logs['loss1'], logs['loss2'], logs['loss3']]],
            win='losses',
            update='append'
        )

        logs['loss1'] = logs['loss2'] = logs['loss3'] = logs['num_stats'] = 0

        print(
            f"Memory: {(torch.cuda.max_memory_allocated() / 1000000.0)} mb, step = {global_step + 1}: loss = {loss.item():.4f}")

        # safely exit the loop
        if len(json.loads(vis.get_window_data('quit'))['content']) > 0:
            vis.text('', win='quit')
            print("Exiting safely...")
            break

    if (global_step) % 50 == 49:
        vid_batch = args['vid_batch']
        vis.image(
            torchvision.utils.make_grid(torch.clamp(g.data[:vid_batch], 0, 1), padding=0, nrow=int(np.sqrt(vid_batch))),
            win='p_xg0', opts={'title': 'p_xg0 reconstructions', 'jpgquality': 50})

    if (global_step) % 100 == 99:
        with torch.no_grad():

            vid_batch = args['vid_batch']

            # Show latent interpolations SLERP video
            z1 = sample_prior()[:vid_batch]
            z2 = sample_prior()[:vid_batch]

            frames = 64

            ts = torch.linspace(0, 1, frames)
            #vsx = args['width'] * int(np.sqrt(vid_batch))
            #vid = torch.zeros(frames, 3, vsx, vsx)

            #Warrior test case
            vid = torch.zeros(frames, 3, 176, 276)

            for j in range(frames):
                zs = lerp(z1, z2, ts[j])
                with torch.no_grad():
                    v = net(zs)
                    v = torch.softmax(v, dim=1)
                    v_processed_images = []
                    for i in range(v.shape[0]):  # The function receives CHW but g is BCHW
                        single_image = v[i, :, :, :]
                        processed_image = palette_to_rgb(single_image, palettefile)
                        v_processed_images.append(processed_image)
                    v = torch.stack(v_processed_images, dim=0)  # Use stack to get BCHW; Now g is a rgb tensor

                vid[j] = torchvision.utils.make_grid(torch.clamp(v, 0, 1), nrow=int(np.sqrt(vid_batch)), padding=0)

            show_video(vid, num_channels=3)

