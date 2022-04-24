from math import floor
import random
from torch.nn import Module
from utils.get_data_from import GetDataFrom


class SiameseDataset(Module):
    def __init__(self, mode='train', anno_path=None, images_path=None, resize_shape=None, transforms=None) -> None:

        getter = GetDataFrom(mode, anno_path, images_path, resize_shape, transforms)
        anchor_collection, positive_collection = getter.get_full_data()

        self.__anchor_collection = []
        self.__positive_collection = []
        self.__negative_collection = []

        for (idx, anchor) in enumerate(anchor_collection):
            if(len(positive_collection[idx]) >= 5):
                for k in range(floor(len(positive_collection[idx])/5)):
                    self.__anchor_collection += [anchor for p in range(5)]
                    self.__positive_collection += positive_collection[idx][5*k:(k+1)*5]
                    break
        self.__negative_collection = self.__positive_collection.copy()
        random.shuffle(self.__negative_collection)

        print('\nDataset - Anchor size: {}, Positive size: {} in {} mode\n'.format(
            len(self.__anchor_collection),
            len(self.__positive_collection),
            mode))

        # Clear memory
        anchor_collection.clear()
        positive_collection.clear()
        getter = None

    def __len__(self):
        return len(self.__anchor_collection)

    def __getitem__(self, idx):
        return self.__anchor_collection[idx], self.__positive_collection[idx], self.__negative_collection[idx]
