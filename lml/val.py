import os
import re
import json


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


annotation_files = []  # files for traning and vlidation

if(os.path.exists('./validation/true')):
    annotation_files = os.listdir('./validation/true')
    annotation_files.sort(key=natural_keys)

    in_frame_cars = []
    in_frame_pedestrians = []

    with open(f'./validation/true/{annotation_files[0]}') as f:
        true_json = json.load(f)['sequence']
        for frame in true_json:
            in_frame_cars += frame['Car']
            in_frame_pedestrians += frame['Pedestrians']
