import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def visualize_joints(pose, frame_index1, frame_index2):
    pose = pose[...,:34]
    xlim = pose[...,0:34:2].max()
    ylim = pose[...,1:34:2].max()
    posex, posey = pose[frame_index1:frame_index2,0::2], pose[frame_index1:frame_index2,1::2]

    plt.figure(figsize=(3 * (frame_index2 - frame_index1), 5))
    for i in range(frame_index2 - frame_index1):
        plt.subplot(1, frame_index2-frame_index1, i+1)
        plt.scatter(posex[i], posey[i])
        plt.xlim(0, xlim)
        plt.ylim(ylim, 0)
    plt.tight_layout()
    plt.show()


def draw_joint_scatter(pose, shape = (1024,1024,3), draw=False, point_num=4):
    # colors = [(255,0,0), (255,0,0), (255,0,0), (255,0,0), (255,0,0),
    #         (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0),
    #         (0,0,255), (0,0,255), (0,0,255), (0,0,255), (0,0,255), (0,0,255)
    #         ]
    colors = [(254,0,0), (253,0,0), (252,0,0), (251,0,0), (250,0,0),
        (0,254,0), (0,253,0), (0,252,0), (0,251,0), (0,250,0), (0,249,0),
        (0,0,254), (0,0,253), (0,0,252), (0,0,251), (0,0,250), (0,0,249)
        ]
    result = np.zeros(shape, dtype=np.uint8)
    # print(result.shape)
    if len(pose.shape) == 2:
        pose = pose[...,:34].reshape(-1,17,2)
    for frame in pose:
        # print(len(frame))
        for j, joint in enumerate(frame):
            # print(j)
            # if j in [1,2,3,4]:
            #     continue
            # print(joint, joint[0]*SHAPE[0], joint[1]*SHAPE[0])
            xcoord = int(joint[0] * shape[0])
            ycoord = int(joint[1] * shape[1])
            # print(xcoord, ycoord)

            color_to_paint = colors[j]

            if point_num == 1:
                result[xcoord, ycoord, :] = color_to_paint
            if point_num == 4:
                result[xcoord, ycoord+1, :] = color_to_paint
                result[xcoord+1, ycoord, :] = color_to_paint
                result[xcoord+1, ycoord+1, :] = color_to_paint         
            elif point_num == 9:
                result[:,xcoord-1, ycoord] = color_to_paint
                result[:,xcoord, ycoord-1] = color_to_paint
                result[:,xcoord-1, ycoord-1] = color_to_paint
                result[:,xcoord-1, ycoord+1] = color_to_paint
                result[:,xcoord+1, ycoord-1] = color_to_paint

    if draw:
        try:
            plt.imshow(result)
        except:
            print('unable to draw image')
    return result

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