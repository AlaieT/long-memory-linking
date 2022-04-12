from math import floor
import random
import re
import os
import json
import numpy as np

# -----------------------------------------------------------------------------
# ----------------------------Load and parse data------------------------------
# -----------------------------------------------------------------------------


img_width = 1936
img_height = 1216


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_data_from(path: str, mod: str):
    objects_id = []
    objects_on_frames = []
    frames_count = 0

    if(os.path.exists(path)):
        files = os.listdir(path)
        files.sort(key=natural_keys)

        for file in files:
            with open(f'{path}/{file}') as f:
                true_json = json.load(f)['sequence']
                frames_count = len(true_json)
                # Separete all objects, collect its frames. Each missed past frame marked as -1
                for (i, frame) in enumerate(true_json):
                    if('Car' in frame):
                        for obj in frame['Car']:
                            box2d = obj['box2d']
                            h = abs(box2d[1] - box2d[3])/img_height
                            a = abs(box2d[0] - box2d[2])/abs(box2d[1] - box2d[3])
                            x_c = (box2d[0] + abs(box2d[0] - box2d[2])/2)/img_width
                            y_c = (box2d[1] + abs(box2d[1] - box2d[3])/2)/img_height
                            if(obj['id'] in objects_id):
                                obj_idx = objects_id.index(obj['id'])
                                objects_on_frames[obj_idx].append([x_c, y_c, a, h])
                            else:
                                objects_id.append(obj['id'])
                                coords = [[x_c, y_c, a, h]]
                                objects_on_frames.append(coords)
                    if('Pedestrian' in frame):
                        for obj in frame['Pedestrian']:
                            box2d = obj['box2d']
                            h = abs(box2d[1] - box2d[3])/img_height
                            a = abs(box2d[0] - box2d[2])/abs(box2d[1] - box2d[3])
                            x_c = (box2d[0] + abs(box2d[0] - box2d[2])/2)/img_width
                            y_c = (box2d[1] + abs(box2d[1] - box2d[3])/2)/img_height
                            if(obj['id'] in objects_id):
                                obj_idx = objects_id.index(obj['id'])
                                objects_on_frames[obj_idx].append([x_c, y_c, a, h])
                            else:
                                objects_id.append(obj['id'])
                                coords = [[x_c, y_c, a, h]]
                                objects_on_frames.append(coords)

        # split data by length
        if(mod == 'train'):
            splited_data = []

            for i in range(len(objects_on_frames)):
                if(len(objects_on_frames[i]) > 2):
                    if(len(objects_on_frames[i]) > 7):
                        for k in range(0, floor(len(objects_on_frames[i])/5), 1):
                            splited_data.append(objects_on_frames[i][k*3:(k+1)*5])
                        if(len(objects_on_frames[i]) - (k+1)*5 > 0):
                            splited_data.append(objects_on_frames[i][(k+1)*5:])
                    else:
                        splited_data.append(objects_on_frames[i])

            random.shuffle(splited_data)
            return splited_data, frames_count

    random.shuffle(objects_on_frames)
    return objects_on_frames, frames_count
