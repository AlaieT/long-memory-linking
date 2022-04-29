from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

class GetDataFrom:
    def __init__(self, mode='train', anno_path=None, images_path=None, reshape=None, transforms=None):
        self.__mode = mode
        self.__valid = ['train_06', 'train_07', 'train_10', 'train_14', 'train_19']
        self.__data_annotation_path = anno_path
        self.__data_images_path = images_path
        self.__reshape = reshape
        self.__transforms = transforms

    def __get_unique_data(self):
        if(self.__data_annotation_path is not None and self.__data_images_path is not None):
            print('\n-------------------> Working in {} mode\n'.format(self.__mode))
            
            annotations = pd.read_csv(self.__data_annotation_path + '/siamese_training.csv')
            # Get unique objects
            unique_image_paths = annotations.drop_duplicates(subset=['object_id']).values
            id_collection_people = []
            id_collection_car = []

            print('\nGeting all unique objects...\n')

            with tqdm(total=unique_image_paths.shape[0], bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
                for image_path in unique_image_paths:
                    if(((image_path[0][:8] not in self.__valid) and (self.__mode == 'train'))
                       or ((image_path[0][:8] in self.__valid) and (self.__mode == 'valid'))):
                       if('Car' in image_path[0].split('_')):
                           id_collection_car.append(image_path[1])
                       if('Pedestrian' in image_path[0].split('_')):
                           id_collection_people.append(image_path[1])

                    pbar.update(1)
                pbar.close()

            # return image_collection, id_collection
            return id_collection_car, id_collection_people

    def __get_positive_data(self, id_collection: list):
        annotations = pd.read_csv(self.__data_annotation_path + '/siamese_training.csv')
        image_collection = []


        with tqdm(total=len(id_collection), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
            for id in id_collection:
                appearances = annotations.loc[annotations['object_id'] == id].values
                image_collection.append([])

                for image_path in appearances:
                    image = Image.open(self.__data_images_path + '/{}/{}'.format(image_path[0][:8], image_path[0])).resize(self.__reshape)

                    # Transform image
                    if(self.__transforms is not None):
                        image = self.__transforms(image)

                    image_collection[-1].append(image)

                pbar.update(1)
            pbar.close()

        return image_collection

    def get_full_data(self):
        id_collection_car,id_collection_people = self.__get_unique_data()
        print('\nCollecting full data of pedestrians...\n')
        collection_people = self.__get_positive_data(id_collection_people)
        print('\nCollecting full data of cars...\n')
        collection_car = self.__get_positive_data(id_collection_car)

        return collection_people, collection_car
