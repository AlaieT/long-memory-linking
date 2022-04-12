import matplotlib.pyplot as plt
import os
import numpy as np

# -----------------------------------------------------------------------------
# -------------------------Save ploted graphs in images------------------------
# -----------------------------------------------------------------------------


def show_in_time(pred: list, real: list, title: str):
    # example of trajectory in time
    fig, axs = plt.subplots(2, 1)
    fig.tight_layout()

    if(not os.path.exists('./val_img')):
        os.mkdir('./val_img')

    # preds for 1 prediction
    # x_pred = np.append(real[:-1, 0], pred[0])
    # y_pred = np.append(real[:-1, 1], pred[1])

    # For in progress prediction
    x_pred = [x[0] for x in pred]
    y_pred = [y[1] for y in pred]

    # real
    x_real = real[:, 0]
    y_real = real[:, 1]

    # axs[0].plot(np.linspace(0, x_pred.shape[0], x_pred.shape[0]), x_pred, linestyle='--', marker='o')
    # axs[0].plot(np.linspace(0, x_real.shape[0], x_real.shape[0]), x_real, linestyle='--', marker='o')
    axs[0].plot(np.linspace(0, len(x_pred), len(x_pred)), x_pred, linestyle='--', marker='o')
    axs[0].plot(np.linspace(0, x_real.shape[0], x_real.shape[0]), x_real, linestyle='--', marker='o')
    axs[0].set_title('X pred - real')
    axs[0].legend(['Pred', 'Real'])
    axs[0].grid()

    axs[1].plot(np.linspace(0, len(y_pred), len(y_pred)), y_pred, linestyle='--', marker='o')
    axs[1].plot(np.linspace(0, y_real.shape[0], y_real.shape[0]), y_real, linestyle='--', marker='o')
    axs[1].set_title('Y pred - real')
    axs[1].legend(['Pred', 'Real'])
    axs[1].grid()

    plt.savefig(f"./val_img/{title}.png", dpi=200)
    plt.close()
