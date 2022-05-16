from sympy import im
import torch
import numpy as np
import torch
from sklearn.metrics import pairwise_distances


def predict(x: np.float32, y: np.float32, w: np.float32, h: np.float32, model_, frame_rate):
    with torch.no_grad():
        if(x[x != -1].shape[0] > 1):
            det = pairwise_distances([[x[-2],y[-2]]],[[x[-1],y[-1]]])
            
            if(det >= 1/(frame_rate*2)):
                if(x.shape[0] >= frame_rate):
                    _x = x[-frame_rate:]
                    _y = y[-frame_rate:]
                    _w = w[-frame_rate:]
                    _h = h[-frame_rate:]

                    _x = _x[_x != -1]
                    _y = _y[_y != -1]
                    _w = _w[_w != -1]
                    _h = _h[_h != -1]
                else:
                    _x = x[x != -1]
                    _y = y[y != -1]
                    _w = w[w != -1]
                    _h = h[h != -1]

                sample = np.empty((1, _x.shape[0], 4), dtype=np.float32)
                sample[0, :, 0] = _x
                sample[0, :, 1] = _y
                sample[0, :, 2] = _w
                sample[0, :, 3] = _h

                data = torch.from_numpy(sample)

                pred = model_(data)
                outs = pred.cpu().detach().numpy().reshape((4,))

                return outs
            else:
                return [x[-1], y[-1], w[-1], h[-1]]
        else:
            return [x[-1], y[-1], w[-1], h[-1]]
