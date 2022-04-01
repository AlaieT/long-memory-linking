# formate ofad dataset to yolo look like format

from PIL import Image
import json
import os

if(not os.path.exists('../ofad')):
    os.mkdir('../ofad')
    os.mkdir('../ofad/images')
    os.mkdir('../ofad/images/train')
    os.mkdir('../ofad/labels')
    os.mkdir('../ofad/labels/train')

sequences = ['000076', '000080', '000092', '000104', '000113', '000121']

max_seq_data = 260

for seq in sequences:

    f_json = open(f'./json/{seq}/{seq}.json')

    data = json.load(f_json)

    f_json.close()

    frame_width = data['meta_info']['image_size'][0]
    frame_height = data['meta_info']['image_size'][1]

    # parse frames data
    ofad_frames = data['frames']
    yolo_frames = []

    ofad_frames = ofad_frames[0:max_seq_data]

    for frame in ofad_frames:
        frame_id = frame['frame_id']
        fixed_frame = []

        # if on frame exists some objects
        if 'annos' in frame:
            names = frame['annos']['names']  # all detected object by all 7 cameras
            boxes_2d_cam0 = frame['annos']['boxes_2d']['cam01']  # chose boxes for only one camera

            # store all detected objects on one frame
            for i in range(len(names)):
                if boxes_2d_cam0[i] != [-1, -1, -1, -1] and (names[i] == 'Car' or names[i] == 'Pedestrian'):
                    # class id for yolo format, car == 0, person == 1
                    class_id = 0 if names[i] == 'Car' else 1

                    # calulcate center, width and height of object's 2d box

                    o_width = (boxes_2d_cam0[i][2] - boxes_2d_cam0[i][0])/2/frame_width
                    o_height = (boxes_2d_cam0[i][3] - boxes_2d_cam0[i][1])/2/frame_height

                    o_center_x = (boxes_2d_cam0[i][2] - (boxes_2d_cam0[i][2] - boxes_2d_cam0[i][0])/2)/frame_width
                    o_center_y = (boxes_2d_cam0[i][3] - (boxes_2d_cam0[i][3] - boxes_2d_cam0[i][1])/2)/frame_height

                    # formate to yolo string data
                    yolo_label = f'{class_id} {o_center_x} {o_center_y} {o_width} {o_height}'
                    if(o_width*2 < 0.7 and o_height*2 < 0.7):
                        fixed_frame.append(yolo_label)

        if len(fixed_frame) != 0:
            # save images with 2d boxes
            img = Image.open(f'./images/{seq}/cam01/{frame_id}.jpg')
            img = img.resize((640, 640))
            img.save(f'../ofad/images/train/{seq}{frame_id}.jpg')
            yolo_frames.append({'frame_id': frame_id, 'objects': fixed_frame})

    # save each objects labels in separate frame_id.txt file
    for frame in yolo_frames:
        frame_id = frame['frame_id']
        objects = frame['objects']

        with open(f'../ofad/labels/train/{seq}{frame_id}.txt', 'w', encoding='utf-8') as f:
            for obj in objects:
                f.write(f"{obj}\n")
        f.close()
