from tqdm import tqdm
import os
import numpy as np
from utils.frame_object import Frame_Object
from utils.video_worker import Video_Worker
# from PIL import Image, ImageDraw, ImageFont
import json
from lstm.predict import predict
from lstm.model import LSTM
import torch
from scipy.optimize import linear_sum_assignment
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

video_worker = Video_Worker()
# video_worker.split_into(source_path='./video/train/train_10.mp4', save_path='./data/images')

# # Detect on frames
# os.system('python3 ../yolov5/detect.py --weights ../yolov5/yolov5x.pt --source ../lml/data/images --data ../yolov5/data/coco128.yaml --classes 0 2 --project ../lml/data --nosave --save-txt')

save_tracking = False

# define lstm model for predicion of next position
model_lstm = LSTM()
model_lstm.load_state_dict(torch.load('./lstm/models/best.pt'))
model_lstm.eval()


videos = ['train_06', 'train_07', 'train_10', 'train_14', 'train_19']
res_json = {'train_06.mp4': [], 'train_07.mp4': [], 'train_10.mp4': [], 'train_14.mp4': [], 'train_19.mp4': []}


for video in videos:
    # Linked between frames objects + next frame predtion(s)
    linked_data = np.array([])
    frame_idx = 0  # current frame number
    files = []  # labels for each frame that cames from yolo
    images = []  # images aka frames from splited video
    frames_count = 0

    # For each class of ojbect(car, men and e.t.c) ammount of detection(like 'car'=100, 'men'=23)
    detected_object_numeration = np.array([0]*80)

    if(os.path.exists('./data/labels')):
        df = pd.read_csv('./data/labels/submission_fold_0_1936.csv')
        for i in range(600):
            if(i < 10):
                temp_csv = df.loc[df['filename'] == f'{video}-00{i}']
            elif(i < 100):
                temp_csv = df.loc[df['filename'] == f'{video}-0{i}']
            elif (i < 999):
                temp_csv = df.loc[df['filename'] == f'{video}-{i}']

            temp_csv = temp_csv[temp_csv['confidence'] >= 0.75]
            temp_csv_new = temp_csv.drop(['filename', 'confidence'], 1)
            files.append(temp_csv_new.values.tolist())

        # if(save_tracking):
        #     images = os.listdir('./data/images')
        #     images.sort(key=video_worker.natural_keys)

        frames_count = len(files)

    # For now tracking not live, 'cos beta
    with tqdm(total=frames_count, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
        for file in files:
            detected_classes = np.empty((0, 5), dtype=np.float32)

            # Get from detecte object only class, Xc, Yc
            for obj in file:
                # line_data = obj.split()
                _class = int(obj[4])
                _x = float(obj[0])
                _y = float(obj[1])
                _w = float(obj[2])
                _h = float(obj[3])

                if(_w*_h*1936*1216 >= 1024):
                    new_data = np.empty((1, 5))
                    new_data[0:] = np.array([_class, _x, _y, _w, _h])
                    if(detected_classes.shape == 0):
                        detected_classes = np.expand_dims(detected_classes, axis=0)
                    detected_classes = np.append(detected_classes, new_data, axis=0)

            # Mahalanobis distance, link new frame objects to prev frame objects
            all_dx_0 = np.empty((len([cl for cl in linked_data if cl._class == 0 and cl._tracking]), len(
                [cl for cl in detected_classes if cl[0] == 0])), dtype=np.float32)
            all_dx_2 = np.empty((len([cl for cl in linked_data if cl._class == 1 and cl._tracking]), len(
                [cl for cl in detected_classes if cl[0] == 1])), dtype=np.float32)

            am_0 = 0
            am_2 = 0

            for in_frame_obj in linked_data:
                if(in_frame_obj._tracking):
                    obj_dx = in_frame_obj.mahalanobis_distance(
                        detected_classes[:, 0],
                        detected_classes[:, 1],
                        detected_classes[:, 2],
                        detected_classes[:, 3],
                        detected_classes[:, 4])

                    if(in_frame_obj._class == 0):
                        all_dx_0[am_0, :] = np.array(obj_dx)
                        am_0 += 1
                    if(in_frame_obj._class == 1):
                        all_dx_2[am_2, :] = np.array(obj_dx)
                        am_2 += 1

            if(all_dx_0.shape[0] > 0):
                row_ind, col_ind = linear_sum_assignment(all_dx_0)
                row_ind = row_ind.tolist()
                col_ind = col_ind.tolist()
                temp_linked = [cl for cl in linked_data if cl._class == 0 and cl._tracking]
                temp_detected = np.array([cl for cl in detected_classes if cl[0] == 0])

                for (i, row) in enumerate(row_ind):
                    # linking object
                    _x = temp_detected[col_ind[i], 1]
                    _y = temp_detected[col_ind[i], 2]
                    _w = temp_detected[col_ind[i], 3]
                    _h = temp_detected[col_ind[i], 4]

                    temp_linked[row].update_state(_x, _y, _w, _h)

                for i in range(len(temp_linked)):
                    if(not (i in row_ind)):
                        temp_linked[i].lost()

                delete_idx = []
                am_0 = 0
                for col in range(detected_classes.shape[0]):
                    if(detected_classes[col, 0] == 0):
                        if(am_0 in col_ind):
                            delete_idx.append(col)
                        am_0 += 1
                detected_classes = np.delete(detected_classes, delete_idx, axis=0)

            if(all_dx_2.shape[0] > 0):
                row_ind, col_ind = linear_sum_assignment(all_dx_2)
                row_ind = row_ind.tolist()
                col_ind = col_ind.tolist()
                temp_linked = [cl for cl in linked_data if cl._class == 1 and cl._tracking]
                temp_detected = np.array([cl for cl in detected_classes if cl[0] == 1])

                for (i, row) in enumerate(row_ind):
                    # linking object
                    _x = temp_detected[col_ind[i], 1]
                    _y = temp_detected[col_ind[i], 2]
                    _w = temp_detected[col_ind[i], 3]
                    _h = temp_detected[col_ind[i], 4]

                    temp_linked[row].update_state(_x, _y, _w, _h)

                for i in range(len(temp_linked)):
                    if(not (i in row_ind)):
                        temp_linked[i].lost()
                # clear detection array

                delete_idx = []
                am_2 = 0
                for col in range(detected_classes.shape[0]):
                    if(detected_classes[col, 0] == 1):
                        if(am_2 in col_ind):
                            delete_idx.append(col)
                        am_2 += 1
                detected_classes = np.delete(detected_classes, delete_idx, axis=0)

            # Add object that was detected in first time during all time of detection
            for k in range(detected_classes.shape[0]):
                _class = int(detected_classes[k, 0])
                detected_object_numeration[_class] += 1
                new_object = Frame_Object(
                    int(f"{_class+1}{detected_object_numeration[_class]}", base=10),
                    detected_classes[k, 0],
                    detected_classes[k, 1],
                    detected_classes[k, 2],
                    detected_classes[k, 3],
                    detected_classes[k, 4],
                    frame_idx)

                linked_data = np.append(linked_data, new_object)

            # END OF CURRENT FRAME -------> extract tracking data from here

            # Predict nex frame
            if(frame_idx != frames_count-1):
                for in_frame_obj in linked_data:
                    if(in_frame_obj._tracking):
                        cpp_x = in_frame_obj._x_c.copy()
                        cpp_y = in_frame_obj._y_c.copy()
                        cpp_w = in_frame_obj._b_w.copy()
                        cpp_h = in_frame_obj._b_h.copy()
                        preds = predict(cpp_x, cpp_y, cpp_w, cpp_h, model_lstm)
                        in_frame_obj.add_state(preds[0], preds[1], preds[2], preds[3])
            # Draw boxes
            # if(save_tracking):
            #     # Draw boxes
            #     if(not os.path.exists('./tracking')):
            #         os.mkdir('./tracking')

            #     img = Image.open(f'./data/images/{images[frame_idx]}')
            #     draw = ImageDraw.Draw(img)
            #     font = ImageFont.truetype("./font/arial_bold.ttf", 20)

            #     for i in range(len(linked_data)):
            #         if(linked_data[i]._tracking):
            #             color = (0, 255, 0)
            #             if(linked_data[i]._class == 2.0):
            #                 color = (255, 0, 255)
            #             if(linked_data[i]._class == 7.0):
            #                 color = (0, 255, 255)
            #             if(linked_data[i]._lost):
            #                 color = (255, 0, 0)

            #             x_t = (linked_data[i]._x_c[frame_idx] - linked_data[i]._b_w[frame_idx]/2)*1936
            #             x_b = (linked_data[i]._x_c[frame_idx] + linked_data[i]._b_w[frame_idx]/2)*1936

            #             y_t = (linked_data[i]._y_c[frame_idx] - linked_data[i]._b_h[frame_idx]/2)*1216
            #             y_b = (linked_data[i]._y_c[frame_idx] + linked_data[i]._b_h[frame_idx]/2)*1216
            #             draw.rectangle((x_t, y_t, x_b, y_b), outline=color, width=2)

            #             w_txt, h_txt = font.getsize(f'{linked_data[i]._name}')
            #             draw.rectangle((x_t, y_t-h_txt, x_t+w_txt, y_t), fill=color)
            #             draw.text((x_t, y_t-h_txt,), f'{linked_data[i]._name}', font=font, fill='black')

            #             if(linked_data[i]._x_c[linked_data[i]._x_c != -1].shape[0] > 1):
            #                 cpp_x = linked_data[i]._x_c[linked_data[i]._x_c != -1]
            #                 cpp_y = linked_data[i]._y_c[linked_data[i]._y_c != -1]

            #                 first_x = cpp_x[:-1]
            #                 second_x = cpp_x[1:]

            #                 first_y = cpp_y[:-1]
            #                 second_y = cpp_y[1:]

            #                 for k in range(cpp_x.shape[0]-1):
            #                     draw.line((first_x[k]*1936, first_y[k]*1216, second_x[k]*1936, second_y[k]*1216), fill=color, width=3)

            #         img.save(f'./tracking/{frame_idx}.jpg')

            frame_idx += 1
            pbar.update(1)

    # Delet predicted frames for losted objects
    for in_frame_obj in linked_data:
        if(in_frame_obj._lost and in_frame_obj._lost_frames > 0):
            in_frame_obj._x_c = in_frame_obj._x_c[:-1*in_frame_obj._lost_frames]
            in_frame_obj._y_c = in_frame_obj._y_c[:-1*in_frame_obj._lost_frames]
            in_frame_obj._b_w = in_frame_obj._b_w[:-1*in_frame_obj._lost_frames]
            in_frame_obj._b_h = in_frame_obj._b_h[:-1*in_frame_obj._lost_frames]

    # Save frames data into csv file fro futher validation
    for i in range(frames_count):
        frame_car = []
        frame_pedestrian = []
        for in_frame_obj in linked_data:
            if(in_frame_obj._x_c[in_frame_obj._x_c != -1].shape[0] >= 3 and in_frame_obj._x_c.shape[0] >= i + 1):
                if(in_frame_obj._x_c[i] != -1):
                    left = (in_frame_obj._x_c[i] - in_frame_obj._b_w[i]/2)*1936
                    top = (in_frame_obj._y_c[i] - in_frame_obj._b_h[i]/2)*1216

                    if left < 0:
                        left = 0
                    if top < 0:
                        top = 0

                    right = (in_frame_obj._x_c[i] + in_frame_obj._b_w[i]/2)*1936
                    bottom = (in_frame_obj._y_c[i] + in_frame_obj._b_h[i]/2)*1216

                    if bottom > 1216:
                        bottom = 1216
                    if right > 1936:
                        right = 1936

                    box2d = [left, top, right, bottom]

                    if(int(in_frame_obj._class) == 1):
                        frame_pedestrian.append({'id': int(in_frame_obj._name), 'box2d': box2d})
                    if(int(in_frame_obj._class) == 0):
                        frame_car.append({'id': int(in_frame_obj._name), 'box2d': box2d})

        res_json[f'{video}.mp4'].append({'Car': frame_car, 'Pedestrian': frame_pedestrian})

with open('./validation/val.json', 'w') as f:
    f.write(json.dumps(res_json))


# video_worker.collect_all('./tracking', './video', 'result.mp4')
