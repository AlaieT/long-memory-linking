import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_track(track):
    print(f'Seq length: {track.shape[1]}')
    track = track.squeeze(0)

    fig, axs = plt.subplots(4, 1, figsize=(8, 8))
    fig.tight_layout()

    # axs[0].plot(track.cpu().detach().numpy()[:, 0].reshape((track.shape[0],)),
    #             track.cpu().detach().numpy()[:, 1].reshape((track.shape[0],)), '#ff5733')
    # axs[0].set_title('XY')

    # axs[1].plot(track.cpu().detach().numpy()[:, 2].reshape((track.shape[0],)),
    #             track.cpu().detach().numpy()[:, 3].reshape((track.shape[0],)), '#ff5733')
    # axs[1].set_title('WH')

    axs[0].plot(track.cpu().detach().numpy()[:, 0].reshape((track.shape[0],)), '#ff5733')
    axs[0].set_title('X')

    axs[1].plot(track.cpu().detach().numpy()[:, 1].reshape((track.shape[0],)), '#ff9f33')
    axs[1].set_title('Y')

    axs[2].plot(track.cpu().detach().numpy()[:, 2].reshape((track.shape[0],)), '#33b5ff')
    axs[2].set_title('W')

    axs[3].plot(track.cpu().detach().numpy()[:, 3].reshape((track.shape[0],)), '#ff33ed')
    axs[3].set_title('H')

    plt.show()


def plot_heatmap(tracks):
    tracks = tracks.squeeze(0).cpu().detach().numpy()

    df = pd.DataFrame(data=tracks, columns=['x', 'y', 'w', 'h'])

    fig = plt.figure(figsize=(9, 4))
    ax = sns.heatmap(df.corr(), annot=True, annot_kws={'size': 15},cmap='Blues')
    ax.tick_params(axis="x", labelsize=15, labelrotation=90)
    ax.tick_params(axis='y', labelsize=15, labelrotation=0)

    plt.show()
