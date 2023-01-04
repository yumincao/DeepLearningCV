import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import dataloader, random_split

# 1. 准备数据
ROOT = '.data'

train_data = datasets.MNIST(root=ROOT,
                            train=True,
                            download=True)
# 1.1 数据白化（归一化）
means = train_data.data.mean(axis=(0,1,2)) / 255
stds = train_data.data.std(axis=(0,1,2)) / 255

test_data = datasets.MNIST(root=ROOT,
                            train=False,
                            download=True)

# 1.2 设计噪声函数
"""
0-1正态分布 最后显现为白点, clip到0-1
"""
import numpy as np
class Noise(object):
    def __init__(self, factor=0.5):
        self.factor = factor
    def __call__(self,img):
        img_noisy = img + self.factor * np.random.normal(loc=0.0, scale=1.0, size=img.shape)
        img_noisy = np.clip(img_noisy,0.,1.)
        return img_noisy

# 1.3 数据增强
train_transforms = transforms.Compose([
                            Noise(),
                            transforms.RandomRotation(90),
                            transforms.RandomHorizontalFlip(0.3),
                            transforms.RandomCrop(20, padding=2),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = means, std=stds)
])
test_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=means, std=stds)
])

train_data.transform = train_transforms
test_data.transform = test_transforms

# cross-validation
m = len(train_data)
train_dataset, val_dataset = random_split(train_data, [int(m-m*0.2),int(m*0.2)])

# 读取数据并做shuffle
train_loader = dataloader(train_dataset, batch_size=128, shuffle=True)
val_loader = dataloader(val_dataset, batch_size=128, shuffle=False)
test_loader = dataloader(test_data, batch_size=128, shuffle=False)

# 设计网络
class Encoder(nn.Module):
    def __init__(self, encoded_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,8,3,stride=2,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8,16,3,stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=0),
            nn.ReLU()
        )
        # 默认在 dim=1 flatten，形成mxn行，1列
        self.flatten = nn.Flatten()
        # linear和conv2d不同，后者在设计网络时不考虑feature map的size大小，但前者要考虑，体现为3x3
        self.lin = nn.Sequential(
            nn.Linear(3*3*32,128),
            nn.ReLU(),
            nn.Linear(128, encoded_dim),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.lin(x)
        return x

class Decoder(nn.Module):
    def __init__(self,encoded_dim):
        super(Decoder,self).__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3*3*32),
        )
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=(32,3,3))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32,16,3,stride=2,padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16,8,3,stride=2,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8,1,3,stride=2,padding=1)
        )
    def forward(self,x):
        x = self.decoder_lin(x),
        x = self.unflatten(x),
        x = self.decoder(x),
        return x

# 设计超参数
DIM = 16
LOSS = "MSEloss"
LR_RATE = 0.001
SEED = 0
WEIGHT_DECAY = 1e-05
NUM_EPOCH = 30

# 可放在主函数里初始化
encoder = Encoder(encoded_dim=DIM).to(torch.device("cuda"))
decoder = Decoder(encoded_dim=DIM).to(torch.device("cuda"))
if LOSS == "MSEloss":
    criterion = nn.MSELoss()
else:
    raise NameError
params = [
    {'params':encoder.parameters()},
    {'params':decoder.parameters()}
]
optimizer = torch.optim.Adam(params,lr=LR_RATE,weight_decay=WEIGHT_DECAY)

# 设计训练函数
def train_epoch(encoder,decoder,dataloader,criterion,optimizer,device=torch.device("cuda")):
    encoder.train()
    decoder.train()
    temp = []
    for (x,y) in dataloader:
        x = x.float().to(device)
        encoder_outputs = encoder(x)
        decoder_outputs = decoder(encoder_outputs)
        loss = criterion(decoder_outputs,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'for single epoch, the loss is: {loss.data}')
        # temp.append(loss.item())
        temp.append(loss.detach().cpu().numpy())

    return np.mean(temp)

# 设计测试函数用于test和val
def test_epoch(encoder,decoder,criterion,device=torch.device("cuda")):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        for (x,y) in dataloader:
            x = x.float().to(device)
            encoder_output = encoder(x)
            decoder_output = decoder(encoder_output)
            y_pred.append(decoder_output.cpu().numpy())
            y_true.append(y.cpu().numpy())
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)

        val_loss = criterion(y_pred,y_true)
    return val_loss.data

final_train_loss = []
final_test_loss = []
for epoch in NUM_EPOCH:
    train_loss = train_epoch(encoder=encoder,decoder=decoder,criterion=criterion,dataloader=train_loader)
    val_loss = test_epoch(encoder=encoder,decoder=decoder,criterion=criterion,dataloader=val_loader)
    final_train_loss.append(train_loss)
    final_test_loss.append(val_loss)


# 最后测试... dataloader = test_loader
test_epoch()




