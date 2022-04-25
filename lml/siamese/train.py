import torch
import torchvision
from model import SiameseModel
from utils.dataset import SiameseDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

def imgshow(img):
    npimg = img.numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.01), transforms.RandomGrayscale(p=0.01), transforms.GaussianBlur(15)])
transform = transforms.Compose(
    [transforms.ToTensor()])

train_dataset = SiameseDataset('train', './data', './data/siamese_images', (64, 64), transform)
valid_dataset = SiameseDataset('valid', './data', './data/siamese_images', (64, 64), transform)

exmp_batch = next(iter(DataLoader(dataset=train_dataset, shuffle=True, batch_size=8)))
concated = torch.cat((exmp_batch[0], exmp_batch[1]), 0)
print('Example batch labels: {}'.format(exmp_batch[2].squeeze(1).tolist()))
imgshow(torchvision.utils.make_grid(concated))


train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=64)
valid_dataloader = DataLoader(dataset=valid_dataset, shuffle=True, batch_size=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Current deviuce: {}\n'.format(device))

model = SiameseModel()
model.to(device)

criterion = nn.CosineEmbeddingLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)


def train_loop(epochs, train_data, valid_data, model, loss_function, optimizer):
    for epoch in range(epochs):

        model.train()

        print(f"\n------------------------------------> Epoch: {epoch+1}\n")
        print("\033[0;32m"+'############################### Train ###############################'+"\033[1;36m")

        epoch_loss = 0
        with tqdm(total=len(train_data), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
            for (left_img, right_img, label) in train_data:

                left_img, right_img, label = left_img.to(device), right_img.to(device), label.to(device)

                optimizer.zero_grad()
                out1,out2 = model(left_img, right_img)
                loss = loss_function(out1, out2, label.squeeze(1))
                loss.backward()
                optimizer.step()

                epoch_loss += loss
                pbar.update(1)
            pbar.close()

        print("\033[0;0m"+'\nTrian loss: {}\n'.format(epoch_loss))
        print("\033[0;32m"+'############################### Valid ###############################'+"\033[1;36m")

        model.eval()
        epoch_loss = 0

        out_mse = []
        real_mse = []

        with torch.no_grad():
            with tqdm(total=len(valid_data), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
                for (left_img, right_img, label) in valid_data:

                    left_img, right_img, label = left_img.to(device), right_img.to(device), label.to(device)

                    out1, out2 = model(left_img, right_img)
                    loss = loss_function(out1,out2, label.squeeze(1))
                    epoch_loss += loss


                    temp_out = F.cosine_similarity(out1,out2)/2+0.5 
                    out_mse += temp_out.reshape((temp_out.shape[0], )).cpu().detach().numpy().tolist()
                    real_mse += label.reshape((label.shape[0], )).cpu().detach().numpy().tolist()

                    pbar.update(1)
                pbar.close()

            print(f'\n{out_mse[-10:]}')
            print(real_mse[-10:])

            round_real = np.round(real_mse)
            round_pred = np.round(out_mse)

            accuracy = accuracy_score(round_real, round_pred)
            roc_auc = roc_auc_score(real_mse, out_mse)

            print('\nPred zeros count: {}, Real zeros count: {}\n'.format(
                len(round_pred[round_pred != 1]), len(round_real[round_real != 1])))
            print('Pred ones count: {}, Real ones count: {}'.format(
                len(round_pred[round_pred != 0]), len(round_real[round_real != 0])))
            print("\033[0;0m"+'\nValid loss: {}\nAccuracy: {}%, ROC AUC: {}'.format(epoch_loss, accuracy*100, roc_auc))

            if(not os.path.exists(f'./models')):
                os.mkdir(f'./models')
            torch.save(model.state_dict(), f'./models/last.pt')


train_loop(epochs=10, train_data=train_dataloader, valid_data=valid_dataloader,
           model=model, loss_function=criterion, optimizer=optimizer)
