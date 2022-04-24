from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

class GetDataFrom:
    def __init__(self, mode='train', anno_path=None, images_path=None, resize_shape=None, transforms=None):
        self.__mode = mode
        self.__valid = ['train_06', 'train_07', 'train_10', 'train_14', 'train_19']
        self.__data_annotation_path = anno_path
        self.__data_images_path = images_path
        self.__resize_shape = resize_shape
        self.__transforms = transforms

    def __get_anchor_data(self):
        if(self.__data_annotation_path is not None and self.__data_images_path is not None):
            annotations = pd.read_csv(self.__data_annotation_path + '/siamese_training.csv')
            # Get unique objects
            unique_image_paths = annotations.drop_duplicates(subset=['object_id']).values

            image_collection = []
            id_collection = []

            size = unique_image_paths.shape[0]

            with tqdm(total=size, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
                for image_path in unique_image_paths:
                    if(((image_path[0][:8] not in self.__valid) and (self.__mode == 'train'))
                       or ((image_path[0][:8] in self.__valid) and (self.__mode == 'valid'))):

                        image = Image.open(self.__data_images_path + '/{}/{}'.format(image_path[0][:8], image_path[0]))
                        # Resize and grayscale
                        image = image.resize(self.__resize_shape)

                        # Transform image
                        if(self.__transforms is not None):
                            image = self.__transforms(image)
                        else:
                            image = np.array(image)

                        image_collection.append(image)
                        id_collection.append(image_path[1])

                    pbar.update(1)
                pbar.close()

            return image_collection, id_collection

    def __get_positive_data(self, id_collection: list):
        annotations = pd.read_csv(self.__data_annotation_path + '/siamese_training.csv')
        image_collection = []

        size = len(id_collection)

        print('\nCreating positive data!\nSize of data: {}\n'.format(size))

        with tqdm(total=size, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
            for id in id_collection:
                appearances = annotations.loc[annotations['object_id'] == id].values
                image_collection.append([])

                for image_path in appearances:
                    image = Image.open(self.__data_images_path + '/{}/{}'.format(image_path[0][:8], image_path[0]))
                    # Resize and grayscale
                    image = image.resize(self.__resize_shape)

                    # Transform image
                    if(self.__transforms is not None):
                        image = self.__transforms(image)
                    else:
                        image = np.array(image)

                    image_collection[-1].append(image)

                pbar.update(1)
            pbar.close()

        return image_collection

    def get_full_data(self):
        anchor_collection, id_collection = self.__get_anchor_data()
        positive_collection = self.__get_positive_data(id_collection)

        return anchor_collection, positive_collection
