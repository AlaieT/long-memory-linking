import numpy as np
from numpy.linalg import norm


class Frame_Object(object):
    def __init__(self, name: str, cl: int, x: float, y: float, w: float, h: float, n: int) -> None:
        self._name = name
        self._class = cl
        self._x_c = np.append(np.array([-1]*n), [x])
        self._y_c = np.append(np.array([-1]*n), [y])
        self._b_w = np.append(np.array([-1]*n), [w])
        self._b_h = np.append(np.array([-1]*n), [h])
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

    def lost(self):
        if(self._tracking and (self._lost_frames >= 500 or self._x_c.shape[0] < 2)):
            self._tracking = False
            self._x_c = self._x_c[:-self._lost_frames]
            self._y_c = self._y_c[:-self._lost_frames]
            self._h_w = self._b_w[:-self._lost_frames]
            self._b_h = self._b_h[:-self._lost_frames]
            self._lost_frames = 0

        if(self._tracking):
            self._lost_frames += 1
            self._lost = True

    def mahalanobis_distance(self, cl: np.float32, x: np.float32, y: np.float32, w: np.float32, h: np.float32) -> int:

        combined = np.zeros((self._x_c[self._x_c != -1].shape[0], 4))
        combined[:, 0] = self._x_c[self._x_c != -1]
        combined[:, 1] = self._y_c[self._y_c != -1]
        combined[:, 2] = self._b_w[self._b_w != -1]
        combined[:, 3] = self._b_h[self._b_h != -1]

        covariance_matrix = np.cov(np.array(combined), rowvar=False, ddof=1, dtype=np.float32)
        inv_cov = np.linalg.inv(covariance_matrix + np.identity(covariance_matrix.shape[0]))

        all_dx = []
        eps = 0.45

        for i in range(len(x)):

            if(cl[i] == self._class):
                pred = np.array(
                    [self._x_c[self._x_c != -1][-1],
                     self._y_c[self._y_c != -1][-1],
                     self._b_w[self._b_w != -1][-1],
                     self._b_h[self._b_h != -1][-1]])
                real = np.array([x[i], y[i], w[i], h[i]])
                temp_xy = pred - real
                dx = np.sqrt(np.matmul(np.matmul(temp_xy, inv_cov), temp_xy.T))
                lm = np.dot(pred, real)/(norm(pred)*norm(real))

                d = eps*dx+(1-eps)*lm

                all_dx.append(d)
        return all_dx
