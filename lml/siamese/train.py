import torch
from model import SiameseModel
from utils.dataset import SiameseDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = SiameseDataset('train', './data', './data/siamese_images', (32, 32), transform)
valid_dataset = SiameseDataset('valid', './data', './data/siamese_images', (32, 32), transform)

train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=8)
valid_dataloader = DataLoader(dataset=valid_dataset, shuffle=False, batch_size=8)

model = SiameseModel()
model.train()

criterion = nn.TripletMarginLoss()
optimizer = optim.AdamW(model.parameters())


def train_loop(epochs, train_data, valid_data, model, loss_function, optimizer):
    for epoch in range(epochs):
        print(f"\n------------------------------------> Epoch: {epoch}\n")
        print("\033[0;32m"+'############################### Train ###############################'+"\033[1;36m")

        epoch_loss = 0
        with tqdm(total=len(train_data), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
            for (anchor, positive, negative) in train_data:

                pos = model(anchor, positive)
                neg = model(anchor, negative)
                target = torch.tensor([1 for i in range(pos.shape[0])], dtype=torch.float32,
                                      requires_grad=True).reshape((pos.shape[0], 1))

                optimizer.zero_grad()
                loss = loss_function(target, pos, neg)
                loss.backward()

                epoch_loss += loss
                pbar.update(1)
            pbar.close()

        print("\033[0;0m"+'Trian loss: {}'.format(epoch_loss))
        print("\033[0;32m"+'############################### Valid ###############################'+"\033[1;36m")

        model.eval()
        epoch_loss = 0

        pos_roc_auc = []
        neg_roc_auc = []

        pos_target_roc_auc = [1 for i in range(len(train_data))]
        neg_target_roc_auc = [0 for i in range(len(train_data))]

        with torch.no_grad():
            with tqdm(total=len(train_data), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
                for (anchor, positive, negative) in valid_data:

                    pos = model(anchor, positive)
                    neg = model(anchor, negative)
                    target = torch.tensor([1 for i in range(pos.shape[0])], dtype=torch.float32,
                                          requires_grad=True).reshape((pos.shape[0], 1))

                    loss = loss_function(target, pos, neg)
                    epoch_loss += loss

                    pos_roc_auc += pos.reshape((pos.shape[0], )).detach().numpy().tolist()
                    neg_roc_auc += neg.reshape((neg.shape[0], )).detach().numpy().tolist()

                    pbar.update(1)
                pbar.close()

            roc_auc_pos = roc_auc_score(pos_target_roc_auc, pos_roc_auc)
            roc_auc_neg = roc_auc_score(neg_target_roc_auc, neg_roc_auc)

            print("\033[0;0m"+'Valid loss: {}\nPos AUC ROC: {}\nNeg AUC ROC: {}'.format(epoch_loss, roc_auc_pos, roc_auc_neg))


train_loop(epochs=10, train_data=train_dataloader, valid_data=valid_dataloader,
           model=model, loss_function=criterion, optimizer=optimizer)
