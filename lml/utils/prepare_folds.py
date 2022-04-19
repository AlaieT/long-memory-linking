import json
import pandas as pd

folds = [
    ["train_17.json", "train_18.json", "train_20.json", "train_21.json", "train_24.json"],
    ["train_02.json", "train_03.json", "train_04.json", "train_12.json", "train_15.json"],
    ["train_01.json", "train_05.json", "train_09.json", "train_13.json", "train_22.json"],
    ["train_00.json", "train_08.json", "train_11.json", "train_16.json", "train_23.json"],
]

IMAGE_WIDTH = 1936
IMAGE_HEIGH = 1216


for (idx, fold) in enumerate(folds):
    objects_on_frames = []
    for file in fold:
        with open(f'../lstm/data/{file}') as f:
            true_json = json.load(f)['sequence']
            for (i, frame) in enumerate(true_json):
                if(i < 10):
                    frame_name = f'{file[:-5]}-00{i}'
                elif(i < 100):
                    frame_name = f'{file[:-5]}-0{i}'
                elif(i < 1000):
                    frame_name = f'{file[:-5]}-{i}'

                if('Car' in frame):
                    for obj in frame['Car']:
                        box2d = obj['box2d']
                        h = abs(box2d[1] - box2d[3])/IMAGE_HEIGH
                        w = abs(box2d[0] - box2d[2])/IMAGE_WIDTH
                        x_c = (box2d[0] + abs(box2d[0] - box2d[2])/2)/IMAGE_WIDTH
                        y_c = (box2d[1] + abs(box2d[1] - box2d[3])/2)/IMAGE_HEIGH
                        objects_on_frames.append([frame_name, x_c, y_c, w, h, 1, 1])
                if('Pedestrian' in frame):
                    for obj in frame['Pedestrian']:
                        box2d = obj['box2d']
                        h = abs(box2d[1] - box2d[3])/IMAGE_HEIGH
                        w = abs(box2d[0] - box2d[2])/IMAGE_WIDTH
                        x_c = (box2d[0] + abs(box2d[0] - box2d[2])/2)/IMAGE_WIDTH
                        y_c = (box2d[1] + abs(box2d[1] - box2d[3])/2)/IMAGE_HEIGH
                        objects_on_frames.append([frame_name, x_c, y_c, w, h, 0, 1])

    df = pd.DataFrame(data=objects_on_frames, columns=['filename', 'x_c', 'y_c', 'w', 'h', 'class_label', 'confidence'])
    df.to_csv(f'../data/folds/fold_{idx+1}.csv')
