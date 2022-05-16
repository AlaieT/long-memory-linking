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

    @classmethod
    def set_params(self, max_pred_frames):
        self._max_pred_frames = max_pred_frames

    def add_state(self, x: float, y: float, w: float, h: float) -> None:
        self._x_c = np.append(self._x_c, x)
        self._y_c = np.append(self._y_c, y)
        self._b_w = np.append(self._b_w, w)
        self._b_h = np.append(self._b_h, h)

    def update_state(self, x: float, y: float, w: float, h: float):
        if(self._lost):
            self._x_c = np.append(np.append(self._x_c[: -1*(self._lost_frames+1)], [-1]*self._lost_frames), x)
            self._y_c = np.append(np.append(self._y_c[: -1*(self._lost_frames+1)], [-1]*self._lost_frames), y)
            self._b_w = np.append(np.append(self._b_w[: -1*(self._lost_frames+1)], [-1]*self._lost_frames), w)
            self._b_h = np.append(np.append(self._b_h[: -1*(self._lost_frames+1)], [-1]*self._lost_frames), h)

            self._lost = False
            self._lost_frames = 0
        else:
            self._x_c = np.append(self._x_c[: -1], x)
            self._y_c = np.append(self._y_c[: -1], y)
            self._b_w = np.append(self._b_w[: -1], w)
            self._b_h = np.append(self._b_h[: -1], h)

    def lost(self):
        if(self._tracking):
            self._lost_frames += 1
            self._lost = True

        if(self._tracking and self._lost_frames >= self._max_pred_frames):
            self._tracking = False
            self._x_c = self._x_c[:-self._lost_frames]
            self._y_c = self._y_c[:-self._lost_frames]
            self._h_w = self._b_w[:-self._lost_frames]
            self._b_h = self._b_h[:-self._lost_frames]
            self._lost_frames = 0

    @staticmethod
    def bb_intersection(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        return interArea

    def metrick_distance(self, x: np.float32, y: np.float32, w: np.float32, h: np.float32) -> int:

        _x = (self._x_c[self._x_c != -1])[-1]
        _y = (self._y_c[self._y_c != -1])[-1]
        _w = (self._b_w[self._b_w != -1])[-1]
        _h = (self._b_h[self._b_h != -1])[-1]

        all_Mah = []
        all_IOU = []
        all_fin = []

        th_Mah = 1/self._max_pred_frames

        pred = np.array([_x, _y, _w, _h])

        for i in range(len(x)):
            real = np.array([x[i], y[i],  w[i], h[i]])

            dx = np.mean(np.abs(pred - real))
            iou = self.bb_intersection(
                [_x - _w / 2, _y - _h / 2, _x + _w / 2, _y + _h / 2],
                [x[i] - w[i] / 2, y[i] - h[i] / 2, x[i] + w[i] / 2, y[i] + h[i] / 2])

            all_IOU.append(iou)
            all_Mah.append(dx)

        for i in range(len(all_Mah)):
            if(all_Mah[i] <= th_Mah and all_IOU[i] != 0):
                all_fin.append(all_Mah[i])
            else:
                all_fin.append(10)

        return all_fin
