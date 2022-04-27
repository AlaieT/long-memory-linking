from math import floor
import random
import re
import os
import json
import numpy as np
import torch

# -----------------------------------------------------------------------------
# ----------------------------Load and parse data------------------------------
# -----------------------------------------------------------------------------


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def to_tensor(data):
    data = data
    samples = np.empty((1, len(data), 4), dtype=np.float32)
    samples[0, :, :] = np.array(data, dtype=np.float32)
    samples = torch.tensor(samples.reshape((1, samples.shape[1], 4))).cuda()

    return samples


img_width = 1936
img_height = 1216

TYPES = ['Car', 'Pedestrian']


def get_data_from(path: str, folds: list, mod: str):
    objects_id = []
    objects_on_frames = []
    splited_data = []

    if(os.path.exists(path)):

        if folds is not None:
            folds.sort(key=natural_keys)
        else:
            folds = os.listdir(path)

        for file in folds:
            with open(f'{path}/{file}') as f:
                json_file = json.load(f)

                # if('annotations' in json_file):
                #     ann = json_file['annotations']
                #     img_width = ann['imgWidth']
                #     img_height = ann['imgHeight']

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
                                    objects_on_frames[obj_idx].append([x_c, y_c, w, h])
                                else:
                                    objects_id.append(obj['id'])
                                    coords = [[x_c, y_c, w, h]]
                                    objects_on_frames.append(coords)

        if(mod == 'train'):
            for i in range(len(objects_on_frames)):
                # splited_data.append(to_tensor(objects_on_frames[i]))
                if(len(objects_on_frames[i]) > 6 and len(objects_on_frames[i]) < 500):
                    for k in range(0, floor(len(objects_on_frames[i])/5), 1):
                        splited_data.append(to_tensor(objects_on_frames[i][k*2:(k+1)*5]))

                    if(len(objects_on_frames[i]) - (k+1)*5 > 2):
                        splited_data.append(to_tensor(objects_on_frames[i][(k+1)*5:]))

        else:
            splited_data = []
            for i in range(len(objects_on_frames)):
                splited_data.append(to_tensor(objects_on_frames[i]))

    random.shuffle(splited_data)
    return splited_data
