from cmath import inf
import os
import json
import re
from torch import save
from tqdm import tqdm
from argparse import ArgumentParser
import pandas as pd

'''
Use: $ python format.py --data data_folder --save_path save_folder --mode train/test

Folder tree
data
|----MOT15
|   |-----ADL-Rundle-6
|   |     |----det
|   |     |    |---det.txt
|   |     |----gt
|   |     |    |---gt.txt
|   |     |----img1
|   |     |    |---0000001.jpg
|   |     |----seqinfo.ini
'''


def train(data_path: str, save_path: str, mod: str):
    if(not os.path.exists(data_path)):
        print(f"Path: {data_path} doesn't exist")
    else:
        mots = os.listdir(data_path)
        for mot in mots:
            if(not os.path.exists(f'{data_path}/{mot}/{mod}')):
                print(f"Path: {data_path}/{mot}/{mod} doesn't exist")
            else:
                train_folders = os.listdir(f'{data_path}/{mot}/{mod}')

                for train_folder in train_folders:
                    if(not os.path.exists(f'{data_path}/{mot}/{mod}/{train_folder}/gt')):
                        print(f"Path or file: {data_path}/{mot}/{mod}/{train_folder}/gt/gt.txt doesn't exist")
                    else:
                        if(not os.path.exists(f'{data_path}/{mot}/{mod}/{train_folder}/seqinfo.ini')):
                            print(f"Path or file: {data_path}/{mot}/{mod}/{train_folder}/seqinfo.ini doesn't exist")
                        else:
                            with open(f'{data_path}/{mot}/{mod}/{train_folder}/seqinfo.ini') as info_f:
                                lines = info_f.readlines()
                                frame_rate = int(re.findall(r'\d+', lines[3])[0], base=10)
                                seq_length = int(re.findall(r'\d+', lines[4])[0], base=10)
                                img_width = int(re.findall(r'\d+', lines[5])[0], base=10)
                                img_height = int(re.findall(r'\d+', lines[6])[0], base=10)

                            with open(f'{data_path}/{mot}/{mod}/{train_folder}/gt/gt.txt') as gt_f:
                                start_frame_idx = 1
                                video_data = {'sequence': [], 'annotations': {'length': seq_length, 'frameRate': frame_rate,
                                                                              'start_frame': 1, 'imgWidth': img_width, "imgHeight": img_height}}

                                for i in range(seq_length):
                                    video_data['sequence'].append({'Pedestrian': []})

                                lines = gt_f.readlines()

                                for i in lines:
                                    if(int(i[0], base=10) < start_frame_idx):
                                        start_frame_idx = int(i[0], base=10)

                                print(f"MOT:{mot} - {train_folder}, Seq length: {seq_length}, Start frame: {start_frame_idx}")

                                with tqdm(total=len(lines), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
                                    for (idx, line) in enumerate(lines):
                                        data_in_line = line.split(',')

                                        frame_id = int(data_in_line[0], base=10)-start_frame_idx
                                        object_id = int(data_in_line[1], base=10)
                                        left = float(data_in_line[2])
                                        top = float(data_in_line[3])

                                        if(left < 0):
                                            left = 0
                                        if(top < 0):
                                            top = 0

                                        right = left + float(data_in_line[4])
                                        bottom = top + float(data_in_line[5])

                                        if(right > img_width):
                                            right = img_width
                                        if(bottom > img_height):
                                            bottom = img_height

                                        video_data['sequence'][frame_id]['Pedestrian'].append(
                                            {'id': object_id, 'box2d': [left, top, right, bottom]})

                                        pbar.update(1)
                                    pbar.close()

                                if(not os.path.exists(save_path)):
                                    os.mkdir(save_path)

                                with open(f'{save_path}/{train_folder}.json', 'w', encoding='utf8') as video_json:
                                    if(start_frame_idx != 1):
                                        video_data['annotations']['start_frame'] = start_frame_idx
                                    json.dump(video_data, video_json, ensure_ascii=False)


def test(data_path: str, save_path: str, mod: str):
    if(not os.path.exists(data_path)):
        print(f"Path: {data_path} doesn't exist")
    else:
        mots = os.listdir(data_path)
        for mot in mots:
            if(not os.path.exists(f'{data_path}/{mot}/{mod}')):
                print(f"Path: {data_path}/{mot}/{mod} doesn't exist")
            else:
                test_folders = os.listdir(f'{data_path}/{mot}/{mod}')
                video_data = []

                for test_folder in test_folders:
                    if(not os.path.exists(f'{data_path}/{mot}/{mod}/{test_folder}/det')):
                        print(f"Path or file: {data_path}/{mot}/{mod}/{test_folder}/det/det.txt doesn't exist")
                    else:
                        if(not os.path.exists(f'{data_path}/{mot}/{mod}/{test_folder}/seqinfo.ini')):
                            print(f"Path or file: {data_path}/{mot}/{mod}/{test_folder}/seqinfo.ini doesn't exist")
                        else:
                            with open(f'{data_path}/{mot}/{mod}/{test_folder}/seqinfo.ini') as info_f:
                                lines = info_f.readlines()
                                img_width = int(re.findall(r'\d+', lines[5])[0], base=10)
                                img_height = int(re.findall(r'\d+', lines[6])[0], base=10)
                                seq_length = int(re.findall(r'\d+', lines[4])[0], base=10)

                            with open(f'{data_path}/{mot}/{mod}/{test_folder}/det/det.txt') as det_f:
                                lines = det_f.readlines()

                                print(f"MOT:{mot} - {test_folder}, Seq length: {seq_length}")

                                with tqdm(total=len(lines), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
                                    for (idx, line) in enumerate(lines):
                                        data_in_line = line.split(',')

                                        frame_id = int(data_in_line[0], base=10)
                                        w = float(data_in_line[4])/img_width
                                        h = float(data_in_line[5])/img_height

                                        if(w > 1):
                                            w = 1
                                        if(h > 1):
                                            h = 1

                                        if(float(data_in_line[2])/img_width >= 0):
                                            x_c = float(data_in_line[2])/img_width + w/2
                                        else:
                                            x_c = 0 + w/2

                                        if(float(data_in_line[3])/img_height >= 0):
                                            y_c = float(data_in_line[3])/img_height + h/2
                                        else:
                                            y_c = 0 + h/2

                                        prob = float(data_in_line[6])

                                        video_name = f'{test_folder}-{"0"*(abs(4 - len(str(frame_id))))}{abs(frame_id)}'
                                        video_data.append([video_name, x_c, y_c, w, h, 1, prob/100])

                                        pbar.update(1)
                                    pbar.close()

                if(not os.path.exists(save_path)):
                    os.mkdir(save_path)

                df = pd.DataFrame(data=video_data, columns=['filename', 'x_c',
                                  'y_c', 'w', 'h', 'class_label', 'confidence'])
                df.to_csv(f'{save_path}/{mot}.csv', index=False, encoding='utf-8')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-d", "--data", default='./data', dest='data_path', help="path to mot data", metavar="PATH")
    parser.add_argument("-s", "--save_path", default='./formated', dest='save_path',
                        help="path to save formated data", metavar="PATH")
    parser.add_argument("-m", "--mode", default='train', dest='mode',
                        help="mode of formatin train/test", metavar="MODE")

    args = parser.parse_args()

    if(args.mode == 'train'):
        train(args.data_path, args.save_path, args.mode)
    elif(args.mode == 'test'):
        test(args.data_path, args.save_path, args.mode)
    else:
        print(f'No such mode as {args.mode} exists!')
