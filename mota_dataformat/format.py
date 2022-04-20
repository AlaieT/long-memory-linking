import os
import json
import re
from tqdm import tqdm
'''
Folder tree
data
|----MOT15
|   |-----test
|   |-----train
|   |     |----det
|   |     |    |---det.txt
|   |     |----gt
|   |     |    |---gt.txt
|   |     |----img1
|   |     |    |---0000001.jpg
|   |     |----seqinfo.ini
'''

mots = ['MOT15', 'MOT16', 'MOT17', 'MOT20']

for mot in mots:
    if(not os.path.exists(f'./data/{mot}')):
        print(f"Path: ./data/{mot} doesn't exist")
    else:
        if(not os.path.exists(f'./data/{mot}/train')):
            print(f"Path: ./data/{mot}/train doesn't exist")
        else:
            train_folders = os.listdir(f'./data/{mot}/train')

            for train_folder in train_folders:
                if(not os.path.exists(f'./data/{mot}/train/{train_folder}/gt')):
                    print(f"Path or file: data/{mot}/train/{train_folder}/gt/gt.txt doesn't exist")
                else:
                    if(not os.path.exists(f'./data/{mot}/train/{train_folder}/seqinfo.ini')):
                        print(f"Path or file: data/{mot}/train/{train_folder}/seqinfo.ini doesn't exist")
                    else:
                        with open(f'./data/{mot}/train/{train_folder}/seqinfo.ini') as info_f:
                            lines = info_f.readlines()
                            frame_rate = int(re.findall(r'\d+', lines[3])[0], base=10)
                            seq_length = int(re.findall(r'\d+', lines[4])[0], base=10)
                            img_width = int(re.findall(r'\d+', lines[5])[0], base=10)
                            img_height = int(re.findall(r'\d+', lines[6])[0], base=10)

                        with open(f'./data/{mot}/train/{train_folder}/gt/gt.txt') as gt_f:
                            start_frame_idx = 1
                            video_data = {'sequence': [], 'annotations': {
                                'length': seq_length, 'frameRate': frame_rate, 'start_frame': 1, 'imgWidth': img_width, "imgHeight": img_height}}

                            for i in range(seq_length):
                                video_data['sequence'].append({'Pedestrian': [], "Car": []})

                            lines = gt_f.readlines()
                            print(f"MOT:{mot} - {train_folder}")

                            with tqdm(total=len(lines), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
                                for (idx, line) in enumerate(lines):
                                    data_in_line = line.split(',')

                                    if(idx == 0):
                                        start_frame_idx = int(data_in_line[0], base=10)

                                    frame_id = int(data_in_line[0], base=10)-start_frame_idx
                                    object_id = int(data_in_line[1], base=10)
                                    left = float(data_in_line[2])
                                    top = float(data_in_line[3])
                                    right = left + float(data_in_line[4])
                                    bottom = top + float(data_in_line[5])

                                    video_data['sequence'][frame_id]['Pedestrian'].append(
                                        {'id': object_id, 'box2d': [left, top, right, bottom]})

                                    pbar.update(1)
                                pbar.close()

                            if(not os.path.exists('./formated_data_train/')):
                                os.mkdir('./formated_data_train')

                            with open(f'./formated_data_train/{train_folder}.json', 'w') as video_json:
                                if(start_frame_idx != 1):
                                    video_data['annotations']['start_frame'] = start_frame_idx
                                json.dump(video_data, video_json)
