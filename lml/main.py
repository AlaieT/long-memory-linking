import os
import numpy as np
import matplotlib.pyplot as plt
from frame_object import Frame_Object
from PIL import Image, ImageDraw, ImageFont

# Linked between frames objects + next frame predtion(s)
linked_data = np.array([])
frame_idx = 0  # current frame number

# For each class of ojbect(car, men and e.t.c) ammount of detection(like 'car'=100, 'men'=23)
detected_object_numeration = np.array([0]*80)

files = os.listdir('./data/labels')
files.sort()

images = os.listdir('./data/images')
images.sort()

# For now tracking not live, 'cos beta
for file in files:
    detected_obejcts_x_c = np.array([])
    detected_obejcts_y_c = np.array([])
    detected_classes = np.array([])

    with open(f'./data/labels/{file}') as f:
        # Get from detecte object only class, Xc, Yc
        lines = f.readlines()
        detected_classes = np.array([(obj.split())[0] for obj in lines], dtype=np.int_)
        detected_obejcts_x_c = np.array([(obj.split())[1] for obj in lines], dtype=np.float32)
        detected_obejcts_y_c = np.array([(obj.split())[2] for obj in lines], dtype=np.float32)
        detected_obejcts_b_w = np.array([(obj.split())[3] for obj in lines], dtype=np.float32)
        detected_obejcts_b_h = np.array([(obj.split())[4] for obj in lines], dtype=np.float32)

    # Mahalanobis distance, link new frame objects to prev frame objects
    for in_frame_obj in linked_data:
        link_idx = in_frame_obj.mahalanobis_distance(
            detected_obejcts_x_c, detected_obejcts_y_c, detected_obejcts_b_w, detected_obejcts_b_h)
        if(link_idx != -1):
            detected_classes = np.delete(detected_classes, link_idx)
            detected_obejcts_x_c = np.delete(detected_obejcts_x_c, link_idx)
            detected_obejcts_y_c = np.delete(detected_obejcts_y_c, link_idx)
            detected_obejcts_b_w = np.delete(detected_obejcts_b_w, link_idx)
            detected_obejcts_b_h = np.delete(detected_obejcts_b_h, link_idx)

    # Delete losted objects
    linked_data = np.array([ld for ld in linked_data if not (ld._lost)])

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
    for in_frame_obj in linked_data:
        x_n = np.random.uniform(in_frame_obj._x_c[-1]*0.8, in_frame_obj._x_c[-1]*1.2)
        y_n = np.random.uniform(in_frame_obj._y_c[-1]*0.8, in_frame_obj._y_c[-1]*1.2)
        w_n = np.random.uniform(in_frame_obj._b_w[-1]*0.9, in_frame_obj._b_w[-1]*1.15)
        h_n = np.random.uniform(in_frame_obj._b_h[-1]*0.9, in_frame_obj._b_h[-1]*1.15)
        in_frame_obj.add_state(x_n, y_n, w_n, h_n)

    # Draw boxes
    if(not os.path.exists('./tracking')):
        os.mkdir('./tracking')

    img = Image.open(f'./data/images/{images[frame_idx]}')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("./font/arial_bold.ttf", 20)

    for i in range(len(linked_data)):
        color = (0, 255, 0)
        if(linked_data[i]._lost):
            color = (255, 0, 0)

        x_t = (linked_data[i]._x_c[frame_idx] - linked_data[i]._b_w[frame_idx]/2)*1920
        x_b = (linked_data[i]._x_c[frame_idx] + linked_data[i]._b_w[frame_idx]/2)*1920

        y_t = (linked_data[i]._y_c[frame_idx] - linked_data[i]._b_h[frame_idx]/2)*1020
        y_b = (linked_data[i]._y_c[frame_idx] + linked_data[i]._b_h[frame_idx]/2)*1020
        draw.rectangle((x_t, y_t, x_b, y_b), outline=color, width=2)

        w_txt, h_txt = font.getsize(linked_data[i]._name)
        draw.rectangle((x_t, y_t-h_txt, x_t+w_txt, y_t), fill=color)
        draw.text((x_t, y_t-h_txt,), linked_data[i]._name, font=font, fill='black')

    img.save(f'./tracking/{frame_idx}.jpg')
    frame_idx += 1
