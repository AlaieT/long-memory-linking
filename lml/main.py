import os
import numpy as np
from utils.frame_object import Frame_Object
from utils.video_worker import Video_Worker
from PIL import Image, ImageDraw, ImageFont
import json

video_worker = Video_Worker()
# video_worker.split_into(source_path='./video/train/train_07.mp4', save_path='./data/images')

# # Detect on frames
# os.system('python3 ../yolov5/detect.py --weights ../yolov5/yolov5x.pt --source ../lml/data/images --data ../yolov5/data/coco128.yaml --classes 0 2 --project ../lml/data --nosave --save-txt')

# Linked between frames objects + next frame predtion(s)
linked_data = np.array([])
frame_idx = 0  # current frame number
files = []  # labels for each frame that cames from yolo
images = []  # images aka frames from splited video
frames_count = 0

save_tracking = True

# For each class of ojbect(car, men and e.t.c) ammount of detection(like 'car'=100, 'men'=23)
detected_object_numeration = np.array([0]*80)

if(os.path.exists('./data/exp/labels') and os.path.exists('./data/images')):
    files = os.listdir('./data/exp/labels')
    files.sort(key=video_worker.natural_keys)

    if(save_tracking):
        images = os.listdir('./data/images')
        images.sort(key=video_worker.natural_keys)

    frames_count = len(files)

# For now tracking not live, 'cos beta
for file in files:
    detected_obejcts_x_c = np.array([])
    detected_obejcts_y_c = np.array([])
    detected_classes = np.array([])

    with open(f'./data/exp/labels/{file}') as f:
        # Get from detecte object only class, Xc, Yc
        lines = f.readlines()
        detected_classes = np.array([(obj.split())[0] for obj in lines], dtype=np.int_)
        detected_obejcts_x_c = np.array([(obj.split())[1] for obj in lines], dtype=np.float32)
        detected_obejcts_y_c = np.array([(obj.split())[2] for obj in lines], dtype=np.float32)
        detected_obejcts_b_w = np.array([(obj.split())[3] for obj in lines], dtype=np.float32)
        detected_obejcts_b_h = np.array([(obj.split())[4] for obj in lines], dtype=np.float32)

        filter = detected_obejcts_b_w.copy()*detected_obejcts_b_h.copy()*1936*1216 >= 1024
        detected_classes = detected_classes[filter]
        detected_obejcts_x_c = detected_obejcts_x_c[filter]
        detected_obejcts_y_c = detected_obejcts_y_c[filter]
        detected_obejcts_b_w = detected_obejcts_b_w[filter]
        detected_obejcts_b_h = detected_obejcts_b_h[filter]

    # Mahalanobis distance, link new frame objects to prev frame objects
    for in_frame_obj in linked_data:
        if(in_frame_obj._tracking):
            link_idx = in_frame_obj.mahalanobis_distance(
                detected_classes, detected_obejcts_x_c, detected_obejcts_y_c, detected_obejcts_b_w, detected_obejcts_b_h)
            if(link_idx != -1):
                detected_classes = np.delete(detected_classes, link_idx)
                detected_obejcts_x_c = np.delete(detected_obejcts_x_c, link_idx)
                detected_obejcts_y_c = np.delete(detected_obejcts_y_c, link_idx)
                detected_obejcts_b_w = np.delete(detected_obejcts_b_w, link_idx)
                detected_obejcts_b_h = np.delete(detected_obejcts_b_h, link_idx)

    # Delete losted objects
    # linked_data = np.array([ld for ld in linked_data if not (ld._lost)])

    # Add object that was detected in first time during all time of detection

    for k in range(len(detected_classes)):
        detected_object_numeration[detected_classes[k]] += 1
        new_object = Frame_Object(
            f"{detected_classes[k]}_{detected_object_numeration[detected_classes[k]]}",
            detected_classes[k],
            detected_obejcts_x_c[k],
            detected_obejcts_y_c[k],
            detected_obejcts_b_w[k],
            detected_obejcts_b_h[k],
            frame_idx)

        linked_data = np.append(linked_data, new_object)

    # Predict nex frame
    if(frame_idx != frames_count-1):
        for in_frame_obj in linked_data:
            if(in_frame_obj._tracking):
                cpp_x = in_frame_obj._x_c[in_frame_obj._x_c != -1].copy()
                cpp_y = in_frame_obj._y_c[in_frame_obj._y_c != -1].copy()

                x_n = np.random.uniform(cpp_x[-1]*0.9, cpp_x[-1]*1.1)
                y_n = np.random.uniform(cpp_y[-1]*0.9, cpp_y[-1]*1.1)
                w_n = np.random.uniform(in_frame_obj._b_w[-1]*0.9, in_frame_obj._b_w[-1]*1.1)
                h_n = np.random.uniform(in_frame_obj._b_h[-1]*0.9, in_frame_obj._b_h[-1]*1.1)
                in_frame_obj.add_state(x_n, y_n, w_n, h_n)
            else:
                in_frame_obj.add_state(0, 0, 0, 0)
    # Draw boxes
    if(save_tracking):
        # Draw boxes
        if(not os.path.exists('./tracking')):
            os.mkdir('./tracking')

        img = Image.open(f'./data/images/{images[frame_idx]}')
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("./font/arial_bold.ttf", 20)

        for i in range(len(linked_data)):
            if(linked_data[i]._tracking):
                color = (0, 255, 0)
                if(linked_data[i]._class == 2.0):
                    color = (255, 0, 255)
                if(linked_data[i]._class == 7.0):
                    color = (0, 255, 255)
                if(linked_data[i]._lost):
                    color = (255, 0, 0)

                x_t = (linked_data[i]._x_c[frame_idx] - linked_data[i]._b_w[frame_idx]/2)*1936
                x_b = (linked_data[i]._x_c[frame_idx] + linked_data[i]._b_w[frame_idx]/2)*1936

                y_t = (linked_data[i]._y_c[frame_idx] - linked_data[i]._b_h[frame_idx]/2)*1216
                y_b = (linked_data[i]._y_c[frame_idx] + linked_data[i]._b_h[frame_idx]/2)*1216
                draw.rectangle((x_t, y_t, x_b, y_b), outline=color, width=2)

                w_txt, h_txt = font.getsize(linked_data[i]._name)
                draw.rectangle((x_t, y_t-h_txt, x_t+w_txt, y_t), fill=color)
                draw.text((x_t, y_t-h_txt,), linked_data[i]._name, font=font, fill='black')

        img.save(f'./tracking/{frame_idx}.jpg')

    frame_idx += 1

# Save frames data into csv file fro futher validation
res_json = {'sequence': []}
for i in range(frames_count):
    frame_car = []
    frame_pedestrian = []
    for in_frame_obj in linked_data:
        if(in_frame_obj._x_c[i] != -1):
            box2d = [(in_frame_obj._x_c[i] - in_frame_obj._b_w[i]/2)*1936, (in_frame_obj._y_c[i] - in_frame_obj._b_h[i]/2)*1216,
                     (in_frame_obj._x_c[i] + in_frame_obj._b_w[i]/2)*1936, (in_frame_obj._y_c[i] + in_frame_obj._b_h[i]/2)*1216]

            if(int(in_frame_obj._class) == 0):
                frame_pedestrian.append({'id': in_frame_obj._name, 'box2d': box2d})
            if(int(in_frame_obj._class) == 2):
                frame_car.append({'id': in_frame_obj._name, 'box2d': box2d})

    res_json['sequence'].append({'Car': frame_car, 'Pedestrian': frame_pedestrian})

with open('./validation/val.json', 'w') as f:
    f.write(json.dumps(res_json))


video_worker.collect_all('./tracking', './video', 'result.mp4')
