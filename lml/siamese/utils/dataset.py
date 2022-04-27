from math import ceil, floor
from torch.nn import Module
from utils.get_data_from import GetDataFrom
import numpy as np
from tqdm import tqdm


class SiameseDataset(Module):
    def __init__(self, mode='train', anno_path=None, images_path=None, reshape=None,  transforms=None) -> None:

        getter = GetDataFrom(mode, anno_path, images_path, reshape, transforms)
        collection_pepole, collection_car = getter.get_full_data()

        self.__left_collection = []
        self.__right_collection = []
        self.__labels = np.array([],dtype=np.float32)

        temp_left = []
        temp_right = []
        temp_labels = []

        print('\nCreateing pairs of pedestrians...\n')

        with tqdm(total=len(collection_pepole), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
            for collection in collection_pepole:
                if(len(collection) >= 2):
                    temp_sub_left = collection[1:]
                    temp_sub_right = collection[:-1]

                    for k in range(len(collection)-1):
                        temp_left.append(temp_sub_left[k])
                        temp_right.append(temp_sub_right[k])
                pbar.update(1)
            pbar.close()

        size_full = len(temp_right)
        size_right = ceil(len(temp_right)/2)

        temp_right = temp_right[:size_right] + temp_right[size_right+floor(size_right/2):] + temp_right[size_right:size_right+floor(size_right/2)]

        temp_labels = [1 for i in range(size_right)] + [0 for i in range(size_full - size_right)]

        self.__left_collection += temp_left
        self.__right_collection += temp_right
        self.__labels = np.append(self.__labels,np.array(temp_labels,dtype=np.float32)) 

        temp_left.clear()
        temp_right.clear()
        temp_labels.clear()

        print('\nCreateing pairs of cars...\n')

        with tqdm(total=len(collection_car), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
            for collection in collection_car:
                if(len(collection) >= 2):
                    temp_sub_left = collection[1:]
                    temp_sub_right = collection[:-1]

                    for k in range(len(collection)-1):
                        temp_left.append(temp_sub_left[k])
                        temp_right.append(temp_sub_right[k])
                pbar.update(1)
            pbar.close()

        size_full = len(temp_right)
        size_right = ceil(len(temp_right)/2)

        temp_right = temp_right[:size_right] + temp_right[size_right+floor(size_right/2):] + temp_right[size_right:size_right+floor(size_right/2)]

        temp_labels = [1 for i in range(size_right)] + [0 for i in range(size_full - size_right)]

        self.__left_collection += temp_left
        self.__right_collection += temp_right
        self.__labels = np.append(self.__labels,np.array(temp_labels,dtype=np.float32)) 
        self.__labels = self.__labels.reshape((self.__labels.shape[0], 1))

        temp_left.clear()
        temp_right.clear()
        temp_labels.clear()

        print('\nDataset - Left size: {}, Right size: {}, Labels: {} in {} mode\n'.format(
            len(self.__left_collection),
            len(self.__right_collection),
            len(self.__labels),
            mode))

    def __len__(self):
        return len(self.__left_collection)

    def __getitem__(self, idx):
        return self.__left_collection[idx], self.__right_collection[idx], self.__labels[idx]
