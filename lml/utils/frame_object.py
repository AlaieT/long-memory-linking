import numpy as np
from numpy.linalg import norm
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances


class Frame_Object(object):
    def __init__(self, name: str, cl: int, x: float, y: float, w: float, h: float, img:np.array, n: int) -> None:
        self._name = name
        self._class = cl
        self._x_c = np.append(np.array([-1]*n), [x])
        self._y_c = np.append(np.array([-1]*n), [y])
        self._b_w = np.append(np.array([-1]*n), [w])
        self._b_h = np.append(np.array([-1]*n), [h])
        self._lost = False
        self._lost_frames = 0
        self._tracking = True
        self._img = img

    @classmethod
    def set_model(self, model):
        self._model = model

    def add_state(self, x: float, y: float, w: float, h: float) -> None:
        self._x_c = np.append(self._x_c, x)
        self._y_c = np.append(self._y_c, y)
        self._b_w = np.append(self._b_w, w)
        self._b_h = np.append(self._b_h, h)

    def update_state(self, x: float, y: float, w: float, h: float,img):
        if(self._lost):
            self._x_c = np.append(self._x_c[: -1*(self._lost_frames+1)], [-1]*self._lost_frames)
            self._y_c = np.append(self._y_c[: -1*(self._lost_frames+1)], [-1]*self._lost_frames)
            self._b_w = np.append(self._b_w[: -1*(self._lost_frames+1)], [-1]*self._lost_frames)
            self._b_h = np.append(self._b_h[: -1*(self._lost_frames+1)], [-1]*self._lost_frames)

            self._x_c = np.append(self._x_c, x)
            self._y_c = np.append(self._y_c, y)
            self._b_w = np.append(self._b_w, w)
            self._b_h = np.append(self._b_h, h)

            self._lost = False
            self._lost_frames = 0
            self._img = img
        else:
            self._x_c = np.append(self._x_c[: -1], x)
            self._y_c = np.append(self._y_c[: -1], y)
            self._b_w = np.append(self._b_w[: -1], w)
            self._b_h = np.append(self._b_h[: -1], h)
            self._img = img

    def lost(self):
        if(self._tracking):
            self._lost_frames += 1
            self._lost = True

        if(self._tracking and (self._lost_frames >= 10 or self._x_c[self._x_c != -1].shape[0] < 3)):
            self._tracking = False
            self._x_c = self._x_c[:-self._lost_frames]
            self._y_c = self._y_c[:-self._lost_frames]
            self._h_w = self._b_w[:-self._lost_frames]
            self._b_h = self._b_h[:-self._lost_frames]
            self._img = None
            self._lost_frames = 0

    def mahalanobis_distance(self, cl: np.float32, x: np.float32, y: np.float32, w: np.float32, h: np.float32, img:list) -> int:

        all_dx = []
        eps = 0.45

        _x = self._x_c[self._x_c != -1]
        _y = self._y_c[self._y_c != -1]
        _w = self._b_w[self._b_w != -1]
        _h = self._b_h[self._b_h != -1]

        for i in range(len(x)):
            if(cl[i] == self._class):

                img_distance = self._model(self._img, img[i]).cpu().detach().numpy().tolist()[0][0]

                pred = np.array([_x[-1], _y[-1], _w[-1], _h[-1]])
                real = np.array([x[i], y[i],  w[i], h[i]])

                dx = pairwise_distances(pred.reshape((1, 4)), real.reshape((1, 4)))[0][0]
                # lm = np.dot(pred, real)/(norm(pred)*norm(real))

                # d = eps*dx +(1-eps)*lm
                d = dx + 1-img_distance

                all_dx.append(d)

        return all_dx
