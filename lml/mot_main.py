from tqdm import tqdm
import os
import numpy as np
from utils.frame_object import Frame_Object
from lstm.predict import predict
from lstm.model import LSTM
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import manhattan_distances
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

IMG_DATA = {'ETH-Bahnhof': [640, 480, 14], 'PETS09-S2L1': [768, 576, 7], 'Venice-2': [1920, 1080, 15]}

# IMG_DATA = {
#     'Venice-1': [1920, 1080, 26],
#     'KITTI-19': [1238, 374, 10],
#     'KITTI-16': [1224, 370, 10],
#     'ADL-Rundle-3': [1920, 1080, 30],
#     'ADL-Rundle-1': [1920, 1080, 30],
#     'AVG-TownCentre': [1920, 1080, 2],
#     'ETH-Crossing': [640, 480, 14],
#     'ETH-Linthescher': [640, 480, 14],
#     'ETH-Jelmoli': [640, 480, 14],
#     'PETS09-S2L2': [768, 576, 7],
#     'TUD-Crossing': [640, 480, 25]}

# IMG_DATA = {
#     'Venice-2': [1920, 1080, 30],
#     'KITTI-17': [1224, 370, 10],
#     'KITTI-13': [1242, 375, 10],
#     'ADL-Rundle-8': [1920, 1080, 30],
#     'ADL-Rundle-6': [1920, 1080, 30],
#     'ETH-Pedcross2': [640, 480, 14],
#     'ETH-Sunnyday': [640, 480, 14],
#     'ETH-Bahnhof': [640, 480, 14],
#     'PETS09-S2L1': [768, 576, 7],
#     'TUD-Campus': [640, 480, 25],
#     'TUD-Stadtmitte': [640, 480, 26]}


def fix_lin_assigm(cost_matrix, rows, cols):
    new_rows = []
    new_cols = []

    for (i, row) in enumerate(rows):
        if(cost_matrix[row, cols[i]] != 10):
            new_rows.append(row)
            new_cols.append(cols[i])
    return new_rows, new_cols


'''
This is main function
It can be called in loop for different data
'''


def lml(model_path: str, data_path: str, video: str):

    model_lstm = LSTM()
    model_lstm.load_state_dict(torch.load(f'{model_path}'))
    model_lstm.eval()

    # set frame rate
    Frame_Object.set_params(max_pred_frames=IMG_DATA[video][2])

    print(f'\n----------------------> Linking objects in video: {video}')
    # Linked between frames objects + next frame predtion(s)
    linked_data = np.array([])
    files = []  # labels for each frame that cames from yolo
    frames_count = 0
    res = []

    detected_object_numeration = 0

    if(os.path.exists(data_path)):
        df = pd.read_csv(data_path)
        df = df.loc[df['filename'].str.contains(video)]
        start_frame = int(df['filename'].values[0].split('-')[-1], base=10)
        frames_count = df['filename'].unique().size

        for i in range(frames_count):
            temp_csv = df.loc[df['filename'] == f'{video}-{"0"*(6 - len(str(i+start_frame)))}{i+start_frame}']
            temp_csv = temp_csv[((temp_csv['confidence'] >= 0.35) & (temp_csv['class_label'] == 1)) |
                                ((temp_csv['confidence'] >= 0.35) & (temp_csv['class_label'] == 0))]
            temp_csv_new = temp_csv.drop(['confidence'], 1)
            files.append(temp_csv_new.values.tolist())

    '''
    Live traking
    '''
    with tqdm(total=frames_count, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
        for (idx, file) in enumerate(files):
            detected_classes = np.empty((0, 5), dtype=np.float32)

            '''
            Get objects from new(current) frame
            '''
            for obj in file:
                _class = int(obj[5])
                _x = float(obj[1])
                _y = float(obj[2])
                _w = float(obj[3])
                _h = float(obj[4])

                if(_w*_h >= 0.02/IMG_DATA[video][2]):
                    new_data = np.empty((1, 5))
                    new_data[0:] = np.array([_class, _x, _y, _w, _h], dtype=np.float32)

                    if(detected_classes.shape == 0):
                        detected_classes = np.expand_dims(detected_classes, axis=0)
                    detected_classes = np.append(detected_classes, new_data, axis=0)

            '''
            Caluletae distance between prev-frame object and new objects, and link them
            '''

            all_dx_0 = np.empty((len([cl for cl in linked_data if cl._class == 0 and cl._tracking]), len(
                [cl for cl in detected_classes if cl[0] == 0])), dtype=np.float32)

            am_0 = 0

            '''
            Calculate distance
            '''
            for in_frame_obj in linked_data:
                if(in_frame_obj._tracking):
                    obj_dx = in_frame_obj.metrick_distance(
                        detected_classes[:, 1],
                        detected_classes[:, 2],
                        detected_classes[:, 3],
                        detected_classes[:, 4])

                    if(in_frame_obj._class == 0):
                        all_dx_0[am_0, :] = np.array(obj_dx)
                        am_0 += 1
            '''
            Linking objects of pedestrian class
            '''
            if(all_dx_0.shape[0] > 0):
                row_ind, col_ind = linear_sum_assignment(all_dx_0)
                row_ind, col_ind = fix_lin_assigm(all_dx_0, row_ind, col_ind)

                temp_linked = [cl for cl in linked_data if cl._class == 0 and cl._tracking]
                temp_detected = np.array([cl for cl in detected_classes if cl[0] == 0])

                for (i, row) in enumerate(row_ind):
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
            Add new detect objects
            '''
            for k in range(detected_classes.shape[0]):
                _class = int(detected_classes[k, 0])
                detected_object_numeration += 1
                new_object = Frame_Object(
                    detected_object_numeration,
                    detected_classes[k, 0],
                    detected_classes[k, 1],
                    detected_classes[k, 2],
                    detected_classes[k, 3],
                    detected_classes[k, 4],
                    idx)
                linked_data = np.append(linked_data, new_object)

            '''
            Predict nex frame for objects
            that is still tracking
            '''
            if(idx != frames_count-1):
                for in_frame_obj in linked_data:
                    if(in_frame_obj._tracking):
                        cpp_x = in_frame_obj._x_c.copy()
                        cpp_y = in_frame_obj._y_c.copy()
                        cpp_w = in_frame_obj._b_w.copy()
                        cpp_h = in_frame_obj._b_h.copy()
                        preds = predict(cpp_x, cpp_y, cpp_w, cpp_h, model_lstm, IMG_DATA[video][2])
                        in_frame_obj.add_state(preds[0], preds[1], preds[2], preds[3])
            pbar.update(1)

    '''
    After end of video, remove predictions for objects,
    that wasn't found
    '''
    for in_frame_obj in linked_data:
        if(in_frame_obj._lost and in_frame_obj._lost_frames > 0):
            in_frame_obj._x_c = in_frame_obj._x_c[:-1*in_frame_obj._lost_frames]
            in_frame_obj._y_c = in_frame_obj._y_c[:-1*in_frame_obj._lost_frames]
            in_frame_obj._b_w = in_frame_obj._b_w[:-1*in_frame_obj._lost_frames]
            in_frame_obj._b_h = in_frame_obj._b_h[:-1*in_frame_obj._lost_frames]

    '''
    Add data fom current frame to combine validation json file
    '''
    for i in range(frames_count):
        for in_frame_obj in linked_data:
            if(in_frame_obj._x_c.shape[0] >= i+1):
                if(in_frame_obj._x_c[i] != -1):
                    left = int((in_frame_obj._x_c[i] - in_frame_obj._b_w[i]/2)*IMG_DATA[video][0])
                    top = int((in_frame_obj._y_c[i] - in_frame_obj._b_h[i]/2)*IMG_DATA[video][1])

                    if left < 0:
                        left = 0
                    if top < 0:
                        top = 0

                    width = int(in_frame_obj._b_w[i]*IMG_DATA[video][0])
                    height = int(in_frame_obj._b_h[i]*IMG_DATA[video][1])

                    res.append([i+start_frame, in_frame_obj._name, left, top, width, height, -1, -1, -1, -1])

    res = pd.DataFrame(data=res, columns=['frame', 'id', 'left', 'top', 'width', 'height', 'conf', 'x', 'y', 'z'])
    res = res.sort_values(['id', 'frame'])

    return res


videos = ['ETH-Bahnhof', 'PETS09-S2L1', 'Venice-2']

# videos = ['Venice-1',
#           'KITTI-19',
#           'KITTI-16',
#           'ADL-Rundle-3',
#           'ADL-Rundle-1',
#           'AVG-TownCentre',
#           'ETH-Crossing',
#           'ETH-Linthescher',
#           'ETH-Jelmoli',
#           'PETS09-S2L2',
#           'TUD-Crossing']

# videos = ['Venice-2',
#           'KITTI-17',
#           'KITTI-13',
#           'ADL-Rundle-8',
#           'ADL-Rundle-6',
#           'ETH-Pedcross2',
#           'ETH-Sunnyday',
#           'ETH-Bahnhof',
#           'PETS09-S2L1',
#           'TUD-Campus',
#           'TUD-Stadtmitte']

if __name__ == '__main__':
    for video in videos:
        res_df = lml(model_path=f'./lstm/models/mot/last.pt', data_path=f'./data/mot/MOT15_fixed.csv', video=video)

        if(not os.path.exists('./validation')):
            os.mkdir('./validation')

        if(not os.path.exists(f'./validation/mot')):
            os.mkdir(f'./validation/mot')

        res_df.to_csv(f'./validation/mot/{video}.txt', index=False, header=None)
