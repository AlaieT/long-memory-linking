from math import floor
import random
import re
import os
import json
import numpy as np
import torch
from tqdm import tqdm


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def to_tensor(data, device):
    exit_data = []
    for i in range(data.shape[0]-2):
        samples = np.empty((1, i+2, 4), dtype=np.float32)
        samples[0, :, :] = np.array(data[:(i+2)], dtype=np.float32)
        samples = torch.tensor(samples.reshape((1, samples.shape[1], 4))).to(device)

        exit_data.append(samples)

    return exit_data


img_width = 1936
img_height = 1216

TYPES = ['Car', 'Pedestrian']


def get_data_from(path: str, folds: list, device, mod: str):
    objects_id = []
    objects_on_frames = []
    splited_data = []

    if(os.path.exists(path)):

        if folds is not None:
            folds.sort(key=natural_keys)
        else:
            folds = os.listdir(path)

        print(f'\nPrepaing {mod} data...\n')
        for file in folds:
            with open(f'{path}/{file}') as f:
                json_file = json.load(f)

                if('annotations' in json_file):
                    ann = json_file['annotations']
                    img_width = ann['imgWidth']
                    img_height = ann['imgHeight']

                true_json = json_file['sequence']

                for (i, frame) in enumerate(true_json):
                    for type in TYPES:
                        if(type in frame):
                            for obj in frame[type]:
                                box2d = obj['box2d']
                                h = abs(box2d[1] - box2d[3])/img_height
                                w = abs(box2d[0] - box2d[2])/img_width
                                x_c = (box2d[0] + abs(box2d[0] - box2d[2])/2)/img_width
                                y_c = (box2d[1] + abs(box2d[1] - box2d[3])/2)/img_height
                                if(obj['id'] in objects_id):
                                    obj_idx = objects_id.index(obj['id'])
                                    objects_on_frames[obj_idx] = np.append(
                                        objects_on_frames[obj_idx],
                                        np.array([x_c, y_c, w, h]).reshape(1, 4),
                                        axis=0)
                                else:
                                    objects_id.append(obj['id'])
                                    coords = np.array([x_c, y_c, w, h]).reshape(1, 4)
                                    objects_on_frames.append(coords)

        with tqdm(total=len(objects_on_frames), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
            for i in range(len(objects_on_frames)):
                if(len(objects_on_frames[i]) >= 11):
                    for k in range(0, floor(len(objects_on_frames[i])/11), 1):
                        splited_data.append(to_tensor(objects_on_frames[i][k*11:(k+1)*11, :], device))
                    if(len(objects_on_frames[i]) - (k+1)*11 > 2):
                        splited_data.append(to_tensor(objects_on_frames[i][(k+1)*11:, :], device))
                else:
                    splited_data.append(to_tensor(objects_on_frames[i], device))

                pbar.update(1)
            pbar.close()

    random.shuffle(splited_data)
    return splited_data
