import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def multi_displot(pose_dict, names, opt, coord='x', figsize=(15,5)):
    plt.rcParams["figure.figsize"] = figsize
    
    cnt_type = len(names)
    cnt_video = len(names[0])

    fig, axes = plt.subplots(cnt_type, cnt_video)

    # print(len(axes), len(axes[0]))

    for i in range(cnt_type):

        for j in range(cnt_video):

            # plt.title(name, axes=axes[i][j])
            axes[i][j].set_title(names[i][j])
            if opt != None:
                sns.histplot(pose_dict[names[i][j]][coord][:, opt], kde=False, ax=axes[i][j])
            else:
                sns.histplot(pose_dict[names[i][j]][coord], kde=False, ax=axes[i][j])
                axes[i][j].get_legend().remove()
    
    plt.tight_layout()
    plt.show()
    # for i, ax in enumerate(axes):
    #     sns.histplot(pose_dict[names[i]]['x'][:, opt], kde=False, ax=ax)

    # plt.figure(figsize=(15, 10))
    # plt.show()