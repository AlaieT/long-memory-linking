import cv2
import os
import re


class Video_Worker(object):
    def __init__(self) -> None:
        pass

    def __atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.__atoi(c) for c in re.split(r'(\d+)', text)]

    def split_into(self, source_path: str, save_path: str) -> None:
        # Split video
        if(not os.path.exists(save_path)):
            os.mkdir(save_path)

        vidcap = cv2.VideoCapture(source_path)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(save_path + f"/frame_{count}.jpg", image)
            success, image = vidcap.read()
            count += 1

    def collect_all(self, source_path: str, save_path: str, file_name: str) -> None:
        # Save results to video
        tracking_images = os.listdir(source_path)
        tracking_images.sort(key=self.natural_keys)

        images = [img for img in tracking_images if img.endswith(".jpg")]
        frame = cv2.imread(os.path.join(source_path, images[0]))
        height, width, _ = frame.shape

        video = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MP4V'), 1, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(source_path, image)))

        cv2.destroyAllWindows()
        video.release()
