import pandas as pd
import numpy as np
from ensemble_boxes import *  # https://github.com/ZFTurbo/Weighted-Boxes-Fusion

DATA_PATH = '../data/mot/MOT15.csv'
videos = ['ETH-Bahnhof', 'PETS09-S2L1', 'Venice-2']
# videos = ['MOT17-05-SDP', 'MOT17-02-SDP', 'MOT17-05-FRCNN', 'MOT17-02-FRCNN']

df = pd.read_csv(DATA_PATH)
fixed_data = []

for video in videos:
    frames = df.loc[df['filename'].str.contains(video)]
    start_frame = int(frames.values[0][0].split('-')[-1])
    video_length = int(frames.values[-1][0].split('-')[-1])

    print(f'Video: {video}, Length: {video_length}, Start Frame: {start_frame}')
    
    for i in range(video_length):
        frame = frames.loc[frames['filename'].str.contains(f'{"0"*(6 - len(str(i+start_frame)))}{i+start_frame}')].values
        _boxes = frame[:, 1:5].tolist()

        _labels = frame[:, 5]
        _labels = _labels.reshape((1, _labels.shape[0])).tolist()

        _scores = frame[:, 6]
        _scores = _scores.reshape((1, _scores.shape[0])).tolist()

        re_boxes = [[]]

        for (idx, obj) in enumerate(_boxes):
            left = obj[0]-obj[2]/2
            right = obj[0]+obj[2]/2
            top = obj[1]-obj[3]/2
            bottom = obj[1]+obj[3]/2

            if(left < 0):
                left = 0
            if(top < 0):
                top = 0

            if(right > 1):
                right = 1
            if(bottom > 1):
                bottom = 1

            re_boxes[0].append([left, top, right, bottom])

        fixed_boxes, fixed_scores, fixed_labels = nms_method(re_boxes, _scores, _labels, iou_thr=.55)

        for k in range(len(fixed_boxes)):
            w = fixed_boxes[k][2]-fixed_boxes[k][0]
            h = fixed_boxes[k][3]-fixed_boxes[k][1]
            x_c = fixed_boxes[k][0]+w/2
            y_c = fixed_boxes[k][1]+h/2

            correct_box = [x_c, y_c, w, h]
            new_line = [f'{video}-{"0"*(6 - len(str(i+start_frame)))}{i+start_frame}'] + np.append(correct_box,
                                                                                       [fixed_labels[k], fixed_scores[k]]).tolist()
            fixed_data.append(new_line)

df = pd.DataFrame(data=fixed_data, columns=['filename', 'x_c', 'y_c', 'w', 'h', 'class_label', 'confidence'])
df.to_csv('../data/mot/MOT15_fixed.csv', index=False)
