import numpy as np

class Frame_Object(object):
    def __init__(self, name: str, cl: int, x: float, y: float, w: float, h: float, img, n: int) -> None:
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

    def update_state(self, x: float, y: float, w: float, h: float, img):
        self._img = img

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
        else:
            self._x_c = np.append(self._x_c[: -1], x)
            self._y_c = np.append(self._y_c[: -1], y)
            self._b_w = np.append(self._b_w[: -1], w)
            self._b_h = np.append(self._b_h[: -1], h)

    def lost(self):
        if(self._tracking):
            self._lost_frames += 1
            self._lost = True

        if(self._tracking and self._lost_frames >= 15):
            self._tracking = False
            self._x_c = self._x_c[:-self._lost_frames]
            self._y_c = self._y_c[:-self._lost_frames]
            self._h_w = self._b_w[:-self._lost_frames]
            self._b_h = self._b_h[:-self._lost_frames]
            self._img = None
            self._lost_frames = 0


    @staticmethod
    def bb_intersection_over_union(boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def mahalanobis_distance(self, cl: np.float32, x: np.float32, y: np.float32, w: np.float32, h: np.float32, img) -> int:

        all_Mah = []
        all_Iou = []
        all_Img = []
        all_fin = []

        th_Img = 0.9
        th_Iou = 0.8
        th_Mah = 0.18

        _x = (self._x_c[self._x_c != -1])
        _y = (self._y_c[self._y_c != -1])
        _w = (self._b_w[self._b_w != -1])
        _h = (self._b_h[self._b_h != -1])

        if(_x.shape[0]>=10):
            combined = np.zeros((10, 4))
            combined[:, 0] = _x[-10:]
            combined[:, 1] = _y[-10:]
            combined[:, 2] = _h[-10:]
            combined[:, 3] = _w[-10:]
        else:
            combined = np.zeros((_x.shape[0], 4))
            combined[:, 0] = _x
            combined[:, 1] = _y
            combined[:, 2] = _h
            combined[:, 3] = _w

        covariance_matrix = np.cov(np.array(combined), rowvar=False, ddof=1, dtype=np.float32)
        inv_cov = np.linalg.inv(covariance_matrix + np.identity(covariance_matrix.shape[0]))

        pred = np.array([_x[-1], _y[-1], _w[-1], _h[-1]])
        last_app = np.array([_x[-1*(self._lost_frames+2)], _y[-1*(self._lost_frames+2)], _w[-1*(self._lost_frames+2)], _h[-1*(self._lost_frames+2)]])

        # calculate all distances
        for i in range(len(x)):
            if(cl[i] == self._class):
                real = np.array([x[i], y[i], w[i], h[i]])

                # img dist
                img_distance = self._model(self._img, img[i]).cpu().detach().numpy().tolist()[0][0]
                all_Img.append(img_distance)

                # iou metrick
                iou = self.bb_intersection_over_union(last_app, real)
                all_Iou.append(iou)

                # Mah dist
                temp_xy = pred - real
                mah = np.sqrt(np.matmul(np.matmul(temp_xy, inv_cov), temp_xy.T))
                all_Mah.append(mah)


        for i in range(len(all_Mah)):
            if(all_Mah[i]<=th_Mah and all_Img[i] >= th_Img and all_Iou[i] >= th_Iou):
                all_fin.append(all_Mah[i]+ (1-all_Img[i]) + (1-all_Iou[i]))
            else:
                all_fin.append(10)

        return all_fin