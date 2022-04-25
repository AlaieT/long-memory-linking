from math import ceil, floor
from torch.nn import Module
from utils.get_data_from import GetDataFrom
import numpy as np
from tqdm import tqdm


class SiameseDataset(Module):
    def __init__(self, mode='train', anno_path=None, images_path=None, reshape=None,  transforms=None) -> None:

        getter = GetDataFrom(mode, anno_path, images_path, reshape, transforms)
        positive_collection = getter.get_full_data()

        self.__left_collection = []
        self.__right_collection = []

        print('\nCreateing pairs...\n')

        with tqdm(total=len(positive_collection), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
            for collection in positive_collection:
                if(len(collection) >= 2):
                    temp_left = collection[1:]
                    temp_right = collection[:-1]

                    for k in range(len(collection)-1):
                        self.__left_collection.append(temp_left[k])
                        self.__right_collection.append(temp_right[k])
                pbar.update(1)
            pbar.close()

        size_full = len(self.__right_collection)
        size_right = ceil(len(self.__right_collection)/2)

        self.__right_collection = self.__right_collection[:size_right] + self.__right_collection[size_right+floor(
            size_right/2):] + self.__right_collection[size_right:size_right+floor(size_right/2)]

        self.__labels = np.array([1 for i in range(size_right)] +
                                 [-1 for i in range(size_full - size_right)], dtype=np.float32)
        self.__labels = self.__labels.reshape((self.__labels.shape[0], 1))

        print('\nDataset - Left size: {}, Right size: {}, Labels: {} in {} mode\n'.format(
            len(self.__left_collection),
            len(self.__right_collection),
            len(self.__labels),
            mode))

    def __len__(self):
        return len(self.__left_collection)

    def __getitem__(self, idx):
        return self.__left_collection[idx], self.__right_collection[idx], self.__labels[idx]
