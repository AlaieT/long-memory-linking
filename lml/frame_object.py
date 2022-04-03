from cmath import inf
import numpy as np
import matplotlib.pyplot as plt


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

    def add_state(self, x: float, y: float, w: float, h: float) -> None:
        if(self._lost == False):
            self._x_c = np.append(self._x_c.copy(), x)
            self._y_c = np.append(self._y_c.copy(), y)
            self._b_w = np.append(self._b_w.copy(), w)
            self._b_h = np.append(self._b_h.copy(), h)
        else:
            self._x_c = np.append(self._x_c.copy(), self._x_c[-1])
            self._y_c = np.append(self._y_c.copy(), self._y_c[-1])
            self._b_w = np.append(self._b_w.copy(), self._b_w[-1])
            self._b_h = np.append(self._b_h.copy(), self._b_h[-1])
            self._lost_frames += 1

    def mahalanobis_distance(self, x: np.float32, y: np.float32, w: np.float32, h: np.float32) -> int:
        expected_value_x = np.mean(np.unique(self._x_c[self._x_c != -1]), dtype=np.float32)
        expected_value_y = np.mean(np.unique(self._y_c[self._y_c != -1]), dtype=np.float32)

        expected_value_w = np.mean(np.unique(self._b_w[self._b_w != -1]), dtype=np.float32)
        expected_value_h = np.mean(np.unique(self._b_h[self._b_h != -1]), dtype=np.float32)

        min_idx = -1
        min_dx = +inf

        for i in range(len(x)):
            temp_xy = np.array([x[i]+w[i], y[i]+h[i]]) - np.array([(expected_value_w+expected_value_x)/2,
                                                                   (expected_value_h+expected_value_y)/2])
            dx = np.sqrt(temp_xy[0]**2 + temp_xy[1]**2)

            if dx < min_dx:
                min_dx=dx
                min_idx=i

        if(min_dx < 0.2):
            self._x_c=np.append(self._x_c[: -1].copy(), x[min_idx])
            self._y_c=np.append(self._y_c[: -1].copy(), y[min_idx])
            self._b_w=np.append(self._b_w[: -1].copy(), w[min_idx])
            self._b_h=np.append(self._b_h[: -1].copy(), h[min_idx])
            self._lost=False
            self._lost_frames=0
        else:
            self._x_c=np.append(self._x_c[: -1].copy(), self._x_c[-2])
            self._y_c=np.append(self._y_c[: -1].copy(), self._y_c[-2])
            self._b_w=np.append(self._b_w[: -1].copy(), self._b_w[-2])
            self._b_h=np.append(self._b_h[: -1].copy(), self._b_h[-2])
            self._lost=True
            min_idx=-1

        return min_idx
