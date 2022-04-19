import numpy as np
from torch import nn, optim
import os
import matplotlib.pyplot as plt
from utils.get_data_from import get_data_from
from model import LSTM
from train import training_loop
import torch.nn.functional as F

folds = [
    # ["train_06", "train_07", "train_10", "train_14", "train_19"],
    ["train_17.json", "train_18.json", "train_20.json", "train_21.json", "train_24.json"],
    ["train_02.json", "train_03.json", "train_04.json", "train_12.json", "train_15.json"],
    ["train_01.json", "train_05.json", "train_09.json", "train_13.json", "train_22.json"],
    ["train_00.json", "train_08.json", "train_11.json", "train_16.json", "train_23.json"],
]


for (idx, fold) in enumerate(folds):
    model_lstm = LSTM()
    criterion = nn.L1Loss()
    optimiser = optim.Adam(model_lstm.parameters(), lr=1e-3)

    print(f'\n----------------------------------------------> START-FOLD-{idx+1}\n')

    train_folds = folds.copy()
    del train_folds[idx]
    train_folds = [item for sublist in train_folds for item in sublist]

    objects_train, _ = get_data_from('./data', train_folds, 'train')
    objects_valid, _ = get_data_from('./data', fold, 'valid')

    loss_train, loss_val, dx_fr = training_loop(
        n_epochs=20, model=model_lstm, optimiser=optimiser, loss_fn=criterion, train_data=objects_train,
        test_data=objects_valid, save_path=f'./models/fold_{idx+1}')

    if(not os.path.exists(f'./metricks/fold_{idx+1}')):
        os.mkdir(f'./metricks/fold_{idx+1}')

    fig, axs = plt.subplots(3, 1, figsize=(18, 18))
    fig.tight_layout()

    axs[0].plot(np.linspace(0, len(loss_train), len(loss_train)), loss_train)
    axs[0].set_title('Loss Train')
    axs[0].grid()

    axs[1].plot(np.linspace(0, len(loss_val), len(loss_val)), loss_val)
    axs[1].set_title('Loss Val')
    axs[1].grid()

    axs[2].plot(np.linspace(0, len(dx_fr), len(dx_fr)), dx_fr)
    axs[2].set_title('Dx')
    axs[2].grid()

    plt.savefig(f"./metricks/fold_{idx+1}/metricks.png", dpi=250)
    plt.close()

    print(f'\n----------------------------------------------> END-FOLD-{idx+1}\n')
