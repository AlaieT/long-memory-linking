import torch
import numpy as np
import torch


def predict(x: np.float32, y: np.float32, w: np.float32, h: np.float32, model_: any):
    with torch.no_grad():
        if(x.shape[0] < 10):
            sample = np.empty((1, x.shape[0], 4), dtype=np.float32)
            sample[0, :, 0] = x
            sample[0, :, 1] = y
            sample[0, :, 2] = w
            sample[0, :, 3] = h
        else:
            sample = np.empty((1, 10, 4), dtype=np.float32)
            sample[0, :, 0] = x[-10:]
            sample[0, :, 1] = y[-10:]
            sample[0, :, 2] = w[-10:]
            sample[0, :, 3] = h[-10:]

        data = torch.from_numpy(sample)

        pred = model_(data)
        outs = pred.detach().numpy().reshape((4,))

        return outs
