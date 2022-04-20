from tqdm import tqdm
import os
import numpy as np
from utils.frame_object import Frame_Object
import json
from lstm.predict import predict
from lstm.model import LSTM
import torch
from scipy.optimize import linear_sum_assignment
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

IMG_WIDTH = 1936
IMG_HEIGHT = 1216


def detected_object_name(obj):
    left = (obj[1]-obj[3]/2)*IMG_WIDTH
    right = (obj[1]+obj[3]/2)*IMG_WIDTH
    top = (obj[2]-obj[4]/2)*IMG_HEIGHT
    bottom = (obj[2]+obj[4]/2)*IMG_HEIGHT

    return f'{left:.6f}_{top:.6f}_{right:.6f}_{bottom:.6f}'


'''
This is main function
It can be called in loop for different data
'''


def lml(model_path: str, data_path: str, videos: list, res_json: list, val_save: bool, xgb_save: bool):

    model_lstm = LSTM()
    model_lstm.load_state_dict(torch.load(f'{model_path}'))
    model_lstm.eval()

    fixed_data = []  # xgboost formated data

    for video in videos:
        print(f'\n----------------------> Linking objects in video: {video}')
        # Linked between frames objects + next frame predtion(s)
        linked_data = np.array([])
        frame_idx = 0  # current frame number
        files = []  # labels for each frame that cames from yolo
        frames_count = 0

        out_res = []  # store liking between prev and current frames

        '''
        For each class of ojbect(car, men and e.t.c) ammount of detection(like 'car'=100, 'men'=23)
        '''
        detected_object_numeration = np.array([0]*80)

        '''
        Get data from current video,
        th = 0.75
        '''

        if(os.path.exists(data_path)):
            df = pd.read_csv(data_path)
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

            frames_count = len(files)

        '''
        Live traking
        '''
        with tqdm(total=frames_count, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
            for (idx, file) in enumerate(files):
                detected_classes = np.empty((0, 5), dtype=np.float32)
                detected_img = np.empty((0, 3, 108, 108), dtype=np.float32)

                '''
                Get objects from new(current) frame
                '''
                for obj in file:
                    _class = int(obj[4])
                    _x = float(obj[0])
                    _y = float(obj[1])
                    _w = float(obj[2])
                    _h = float(obj[3])

                    if(_w*_h*1936*1216 >= 1024):
                        new_data = np.empty((1, 5))
                        new_data[0:] = np.array([_class, _x, _y, _w, _h], dtype=np.float32)

                        if(detected_classes.shape == 0):
                            detected_classes = np.expand_dims(detected_classes, axis=0)
                            detected_img = np.expand_dims(detected_img, axis=0)
                        detected_classes = np.append(detected_classes, new_data, axis=0)

                '''
                Caluletae distance between prev-frame object and new objects, and link them
                '''

                all_dx_0 = np.empty((len([cl for cl in linked_data if cl._class == 0 and cl._tracking]), len(
                    [cl for cl in detected_classes if cl[0] == 0])), dtype=np.float32)

                all_dx_2 = np.empty((len([cl for cl in linked_data if cl._class == 1 and cl._tracking]), len(
                    [cl for cl in detected_classes if cl[0] == 1])), dtype=np.float32)

                am_0 = 0
                am_2 = 0

                '''
                Calculate distance
                '''
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

                '''
                Linking objects of pedestrian class 
                '''
                if(all_dx_0.shape[0] > 0):
                    row_ind, col_ind = linear_sum_assignment(all_dx_0)
                    row_ind = row_ind.tolist()
                    col_ind = col_ind.tolist()
                    temp_linked = [cl for cl in linked_data if cl._class == 0 and cl._tracking]
                    temp_detected = np.array([cl for cl in detected_classes if cl[0] == 0])

                    for (i, row) in enumerate(row_ind):
                        if(xgb_save):
                            for k in range(detected_classes.shape[0]):
                                if(np.array_equal(detected_classes[k, :],temp_detected[col_ind[i], :])):
                                    prediction = 1
                                else:
                                    prediction = 0

                                out_res.append([video, frame_idx-1, frame_idx, temp_linked[row]._name,
                                               detected_object_name(detected_classes[k, :]), prediction])

                        _x = temp_detected[col_ind[i], 1]
                        _y = temp_detected[col_ind[i], 2]
                        _w = temp_detected[col_ind[i], 3]
                        _h = temp_detected[col_ind[i], 4]

                        temp_linked[row].update_state(_x, _y, _w, _h,)

                    '''
                    Losing objects
                    '''
                    for i in range(len(temp_linked)):
                        if(not (i in row_ind)):
                            if(xgb_save):
                                for k in range(detected_classes.shape[0]):
                                    out_res.append([video, frame_idx-1, frame_idx, temp_linked[i]._name,
                                                   detected_object_name(detected_classes[k, :]), 0])
                            temp_linked[i].lost()

                    delete_idx = []
                    am_0 = 0
                    '''
                    Left unliked new objects as new deteced objects
                    '''
                    for col in range(detected_classes.shape[0]):
                        if(detected_classes[col, 0] == 0):
                            if(am_0 in col_ind):
                                delete_idx.append(col)
                            am_0 += 1
                    detected_classes = np.delete(detected_classes, delete_idx, axis=0)

                '''
                Linking objects of car class
                '''
                if(all_dx_2.shape[0] > 0):
                    row_ind, col_ind = linear_sum_assignment(all_dx_2)
                    row_ind = row_ind.tolist()
                    col_ind = col_ind.tolist()
                    temp_linked = [cl for cl in linked_data if cl._class == 1 and cl._tracking]
                    temp_detected = np.array([cl for cl in detected_classes if cl[0] == 1])

                    for (i, row) in enumerate(row_ind):
                        if(xgb_save):
                            for k in range(detected_classes.shape[0]):
                                if(np.array_equal(detected_classes[k, :], temp_detected[col_ind[i], :])):
                                    prediction = 1
                                else:
                                    prediction = 0

                                out_res.append([video, frame_idx-1, frame_idx, temp_linked[row]._name,
                                               detected_object_name(detected_classes[k, :]), prediction])

                        _x = temp_detected[col_ind[i], 1]
                        _y = temp_detected[col_ind[i], 2]
                        _w = temp_detected[col_ind[i], 3]
                        _h = temp_detected[col_ind[i], 4]

                        temp_linked[row].update_state(_x, _y, _w, _h)
                    '''
                    Losing objects
                    '''
                    for i in range(len(temp_linked)):
                        if(not (i in row_ind)):
                            if(xgb_save):
                                for k in range(detected_classes.shape[0]):
                                    out_res.append([video, frame_idx-1, frame_idx, temp_linked[i]._name,
                                                   detected_object_name(detected_classes[k, :]), 0])
                            temp_linked[i].lost()

                    delete_idx = []
                    am_2 = 0
                    '''
                    Left unliked new objects as new deteced objects
                    '''
                    for col in range(detected_classes.shape[0]):
                        if(detected_classes[col, 0] == 1):
                            if(am_2 in col_ind):
                                delete_idx.append(col)
                            am_2 += 1
                    detected_classes = np.delete(detected_classes, delete_idx, axis=0)

                '''
                Add new detect objects
                '''
                for k in range(detected_classes.shape[0]):
                    _class = int(detected_classes[k, 0])
                    detected_object_numeration[_class] += 1
                    new_object = Frame_Object(
                        f"{_class+1}_{detected_object_numeration[_class]}",
                        detected_classes[k, 0],
                        detected_classes[k, 1],
                        detected_classes[k, 2],
                        detected_classes[k, 3],
                        detected_classes[k, 4],
                        frame_idx)

                    linked_data = np.append(linked_data, new_object)

                '''
                Predict nex frame for objects
                that is still tracking
                '''
                if(frame_idx != frames_count-1):
                    for in_frame_obj in linked_data:
                        if(in_frame_obj._tracking):
                            cpp_x = in_frame_obj._x_c.copy()
                            cpp_y = in_frame_obj._y_c.copy()
                            cpp_w = in_frame_obj._b_w.copy()
                            cpp_h = in_frame_obj._b_h.copy()
                            preds = predict(cpp_x, cpp_y, cpp_w, cpp_h, model_lstm)
                            in_frame_obj.add_state(preds[0], preds[1], preds[2], preds[3])

                frame_idx += 1
                pbar.update(1)

        '''
        After end of video, remove predictions for objects,
        that wasn't found
        '''
        for (idx, in_frame_obj) in enumerate(linked_data):
            if(in_frame_obj._lost and in_frame_obj._lost_frames > 0):
                in_frame_obj._x_c = in_frame_obj._x_c[:-1*in_frame_obj._lost_frames]
                in_frame_obj._y_c = in_frame_obj._y_c[:-1*in_frame_obj._lost_frames]
                in_frame_obj._b_w = in_frame_obj._b_w[:-1*in_frame_obj._lost_frames]
                in_frame_obj._b_h = in_frame_obj._b_h[:-1*in_frame_obj._lost_frames]

        '''
        Add data fom current frame to combine validation json file
        '''
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
                            frame_pedestrian.append({'id': int(in_frame_obj._name, base=10), 'box2d': box2d})
                        if(int(in_frame_obj._class) == 0):
                            frame_car.append({'id': int(in_frame_obj._name, base=10), 'box2d': box2d})

            if(len(frame_pedestrian) == 0):
                res_json[f'{video}.mp4'].append({'Car': frame_car})
            if(len(frame_car) == 0):
                res_json[f'{video}.mp4'].append({'Pedestrian': frame_pedestrian})
            if(len(frame_car) != 0 and len(frame_pedestrian) != 0):
                res_json[f'{video}.mp4'].append({'Car': frame_car, 'Pedestrian': frame_pedestrian})

        '''
        Saving data for xgboost faetures
        '''
        if(xgb_save):
            print("\nFormating video's data to xgboost format...")
            # Save data for xgboost
            df = pd.DataFrame(data=out_res, columns=['video_name', 'frame1', 'frame2', 'obj_1', 'obj_2', 'predict'])

            with tqdm(total=frames_count, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
                for i in range(frames_count):
                    frame_objects = df[df['frame1'] == i]['obj_1'].to_numpy()

                    for k in range(i+10):
                        sub_frame = df[df['frame2'] == k+1]
                        sub_frame = sub_frame.to_numpy()

                        for line in sub_frame:
                            if line[3] in frame_objects:
                              
                                obj_1 = [obj for obj in linked_data if obj._name == line[3]][0]
                                
                                if(obj_1._x_c.shape[0]-1>=i):
                                    left = (obj_1._x_c[i]-obj_1._b_w[i]/2)*IMG_WIDTH
                                    right = (obj_1._x_c[i]+obj_1._b_w[i]/2)*IMG_WIDTH
                                    top = (obj_1._y_c[i]-obj_1._b_h[i]/2)*IMG_HEIGHT
                                    bottom = (obj_1._y_c[i]+obj_1._b_h[i]/2)*IMG_HEIGHT

                                    name_1 = f"{left:.6f}_{top:.6f}_{right:.6f}_{bottom:.6f}"
                                    fixed_data.append([line[0], i, k, name_1, line[4], line[5]])
                    pbar.update(1)
                pbar.close()

    if(val_save and xgb_save):
        return res_json, fixed_data
    elif(val_save):
        return res_json
    elif(xgb_save):
        return fixed_data


if __name__ == '__main__':

    folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4']

    videos = [
        ["train_06", "train_07", "train_10", "train_14", "train_19"],
        ["train_17", "train_18", "train_20", "train_21", "train_24"],
        ["train_02", "train_03", "train_04", "train_12", "train_15"],
        ["train_01", "train_05", "train_09", "train_13", "train_22"],
        ["train_00", "train_08", "train_11", "train_16", "train_23"],
    ]

    res_json = [
        {'train_06.mp4': [], 'train_07.mp4': [],'train_10.mp4': [],'train_14.mp4': [],'train_19.mp4': []},
        {'train_17.mp4': [], 'train_18.mp4': [], 'train_20.mp4': [], 'train_21.mp4': [], 'train_24.mp4': []},
        {'train_02.mp4': [], 'train_03.mp4': [], 'train_04.mp4': [], 'train_12.mp4': [], 'train_15.mp4': []},
        {'train_01.mp4': [], 'train_05.mp4': [], 'train_09.mp4': [], 'train_13.mp4': [], 'train_22.mp4': []},
        {'train_00.mp4': [], 'train_08.mp4': [], 'train_11.mp4': [], 'train_16.mp4': [], 'train_23.mp4': []}
    ]

    all_xgb_data = []

    for (idx,fold) in enumerate(folds):
        val_data, xgb_data = lml(model_path=f'./lstm/models/{fold}/last.pt',
                                 data_path=f'./data/folds/0.csv',
                                 videos=videos[0],
                                 res_json=res_json[0],
                                 val_save=True,
                                 xgb_save=False)

        all_xgb_data += xgb_data

        '''
        Save validation data for every fold
        '''
        if(not os.path.exists('./validation')):
            os.mkdir('./validation')

        if(not os.path.exists(f'./validation/{fold}')):
            os.mkdir(f'./validation/{fold}')

        with open(f'./validation/{fold}/val.json', 'w') as f:
            f.write(json.dumps(val_data))

    '''
    Save xgboost data fromat
    '''
    df = pd.DataFrame(data=all_xgb_data, columns=['video_name', 'frame1', 'frame2', 'obj_1', 'obj_2', 'predict'])
    df = df.sort_values(by=['video_name'])
    df.to_csv('./last_xgboost.csv')
