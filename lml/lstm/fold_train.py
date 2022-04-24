from torch import nn, optim
from utils.get_data_from import get_data_from
from model import LSTM
from train import training_loop

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

    objects_train = get_data_from(path='./data/signate', folds=train_folds, mod='train')
    objects_valid = get_data_from(path='./data/signate', folds=fold, mod='valid')

    training_loop(
        n_epochs=20, model=model_lstm, optimiser=optimiser, loss_fn=criterion, train_data=objects_train,
        test_data=objects_valid, save_path=f'./models/signate/fold_{idx+1}',
        metrick_path=f'./metricks/signate/fold_{idx+1}')

print(f'\n----------------------------------------------> END-FOLD-{idx+1}\n')
