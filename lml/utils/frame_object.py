from cmath import inf
import numpy as np


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
        if(self._lost == False):
            self._x_c = np.append(self._x_c.copy(), x)
            self._y_c = np.append(self._y_c.copy(), y)
            self._b_w = np.append(self._b_w.copy(), w)
            self._b_h = np.append(self._b_h.copy(), h)
        else:
            self._x_c = np.append(self._x_c.copy(), -1)
            self._y_c = np.append(self._y_c.copy(), -1)
            self._b_w = np.append(self._b_w.copy(), -1)
            self._b_h = np.append(self._b_h.copy(), -1)
            if(self._tracking and self._lost_frames <= 1):
                self._tracking = False
            self._lost_frames += 1

    def mahalanobis_distance(self, cl: np.float32, x: np.float32, y: np.float32, w: np.float32, h: np.float32) -> int:
        combined = np.zeros((self._x_c[self._x_c != -1].shape[0], 2))
        combined[:, 0] = self._x_c[self._x_c != -1]
        combined[:, 1] = self._y_c[self._y_c != -1]

        covariance_matrix = np.cov(np.array(combined), rowvar=False, ddof=1)
        inv_cov = np.linalg.inv(covariance_matrix + np.identity(covariance_matrix.shape[0]))

        min_idx = -1
        min_dx = +inf

        for i in range(len(x)):
            temp_xy = np.array([[self._x_c[self._x_c != -1][-2]-x[i], self._y_c[self._y_c != -1][-2]-y[i]]])
            dx = np.sqrt(np.matmul(np.matmul(temp_xy, inv_cov), temp_xy.T))

            if dx <= min_dx and cl[i] == self._class:
                min_dx = dx
                min_idx = i

        if(min_dx < 0.08):
            self._x_c = np.append(self._x_c[: -1].copy(), x[min_idx])
            self._y_c = np.append(self._y_c[: -1].copy(), y[min_idx])
            self._b_w = np.append(self._b_w[: -1].copy(), w[min_idx])
            self._b_h = np.append(self._b_h[: -1].copy(), h[min_idx])
            self._lost = False
            self._lost_frames = 0
        else:
            self._lost = True
            self._lost_frames += 1
            min_idx = -1

        return min_idx
