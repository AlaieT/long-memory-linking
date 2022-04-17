import numpy as np
from numpy.linalg import norm
import torch


class Frame_Object(object):
    def __init__(self, name: str, cl: int, x: float, y: float, w: float, h: float, n: int) -> None:
        self._name = name
        self._class = cl
        self._x_c = np.append(np.array([-1]*n), [x])
        self._y_c = np.append(np.array([-1]*n), [y])
        self._b_w = np.append(np.array([-1]*n), [w])
        self._b_h = np.append(np.array([-1]*n), [h])
        # self._img = img
        self._lost = False
        self._lost_frames = 0
        self._tracking = True

    def add_state(self, x: float, y: float, w: float, h: float) -> None:
        self._x_c = np.append(self._x_c.copy(), x)
        self._y_c = np.append(self._y_c.copy(), y)
        self._b_w = np.append(self._b_w.copy(), w)
        self._b_h = np.append(self._b_h.copy(), h)

    def update_state(self, x: float, y: float, w: float, h: float):
        # print(f'Frame:{fr}, name: {self._name}, dx: {min_dx}, all_dx: {all_dx}\n')
        if(self._lost):
            self._x_c = np.append(self._x_c[: -1*(self._lost_frames+1)].copy(), [-1]*self._lost_frames)
            self._y_c = np.append(self._y_c[: -1*(self._lost_frames+1)].copy(), [-1]*self._lost_frames)
            self._b_w = np.append(self._b_w[: -1*(self._lost_frames+1)].copy(), [-1]*self._lost_frames)
            self._b_h = np.append(self._b_h[: -1*(self._lost_frames+1)].copy(), [-1]*self._lost_frames)

            self._x_c = np.append(self._x_c.copy(), x)
            self._y_c = np.append(self._y_c.copy(), y)
            self._b_w = np.append(self._b_w.copy(), w)
            self._b_h = np.append(self._b_h.copy(), h)

            self._lost = False
            self._lost_frames = 0
        else:
            self._x_c = np.append(self._x_c[: -1].copy(), x)
            self._y_c = np.append(self._y_c[: -1].copy(), y)
            self._b_w = np.append(self._b_w[: -1].copy(), w)
            self._b_h = np.append(self._b_h[: -1].copy(), h)

        # self._img = img

    def lost(self):
        if(self._tracking and self._lost_frames >= 600):
            self._tracking = False
            self._x_c = self._x_c[:-self._lost_frames]
            self._y_c = self._y_c[:-self._lost_frames]
            self._h_w = self._b_w[:-self._lost_frames]
            self._b_h = self._b_h[:-self._lost_frames]
            self._lost_frames = 0

        if(self._tracking):
            self._lost_frames += 1
            self._lost = True

    def mahalanobis_distance(
            self, cl: np.float32, x: np.float32, y: np.float32, w: np.float32, h: np.float32) -> int:
        all_dx = []
        eps = 0.45

        _x = self._x_c[self._x_c != -1]
        _y = self._y_c[self._y_c != -1]
        _w = self._b_w[self._b_w != -1]
        _h = self._b_h[self._b_h != -1]

        if(_x.shape[0] >= 10):
            _x = _x[-10:]
            _y = _y[-10:]
            _w = _w[-10:]
            _h = _h[-10:]

        combined = np.zeros((_x.shape[0], 4))
        combined[:, 0] = _x
        combined[:, 1] = _y
        combined[:, 2] = _w
        combined[:, 3] = _h

        # prev_img = torch.from_numpy(self._img).unsqueeze(0)

        for i in range(len(x)):
            if(cl[i] == self._class):

                # new_img = torch.from_numpy(img[i, :, :, :]).unsqueeze(0)
                # apperance = model(prev_img, new_img).cpu().detach().numpy()[0, 0]

                covariance_matrix = np.cov(np.array(combined), rowvar=False, ddof=1, dtype=np.float32)
                inv_cov = np.linalg.inv(covariance_matrix + np.identity(covariance_matrix.shape[0]))

                pred = np.array([_x[-1], _y[-1], _w[-1], _h[-1]])
                real = np.array([x[i], y[i],  w[i], h[i]])
                temp_xy = pred - real
                dx = np.sqrt(np.matmul(np.matmul(temp_xy, inv_cov), temp_xy.T))
                lm = np.dot(pred, real)/(norm(pred)*norm(real))

                d = eps*dx+(1-eps)*lm

                all_dx.append(d)

        return all_dx
