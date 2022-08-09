import numpy as np
from scipy.stats import kurtosis, skew

def get_bbox_xmin_ymin(bbox_item):
    return bbox_item[...,0].min(), bbox_item[...,1].min()

# def get_bbox_max_width_height(bbox_item):
    
#     widths = bbox_item[:,:,2] - bbox_item[:,:,0]
#     heights = bbox_item[:,:,3] - bbox_item[:,:,1]
#     stack = np.stack([widths, heights], axis=1).reshape(-1, 2)
#     bbox_sizes = [w*h for (w,h) in stack]
#     bbox_max_idx = np.argmax(bbox_sizes)

#     return widths[bbox_max_idx][0], heights[bbox_max_idx][0]


def get_bbox_max_width_height(bbox_item):
    try:
        widths = bbox_item[:,:,2] - bbox_item[:,:,0]
        heights = bbox_item[:,:,3] - bbox_item[:,:,1]
        # print(widths.shape, heights.shape)
    except:
        bbox_item = np.concatenate(list(bbox_item), axis=0).reshape(-1, 1, 4)
 
        widths = bbox_item[:,:,2] - bbox_item[:,:,0]
        heights = bbox_item[:,:,3] - bbox_item[:,:,1]
    
    stack = np.stack([widths, heights], axis=1).reshape(-1, 2)
    bbox_sizes = [w*h for (w,h) in stack]
    bbox_max_idx = np.argmax(bbox_sizes)

    return widths[bbox_max_idx][0], heights[bbox_max_idx][0]


# KEYPOINT_NUM = 17
def get_stats(coord):
    return np.mean(coord), np.std(coord), kurtosis(coord), skew(coord)

def pose_with_topk_joints(pose, joint_num=17, topk=5, how='x*y'):

    assert how in ['x*y', 'x&y', 'x<>y', 'x', 'y', ]

    result_x, result_y, result_xy = [], [], []

    for k in range(joint_num):
        kx = pose[:,k,0]
        ky = pose[:,k,1]
        kx_norm = kx / np.linalg.norm(kx)
        ky_norm = kx / np.linalg.norm(ky)

        # kx_stats = get_stats(kx)
        # ky_stats = get_stats(ky)
        
        # ssx, ssy = StandardScaler(), StandardScaler()
        # kx_ss = ssx.fit_transform(kx.reshape(-1,1)).flatten()
        # ky_ss = ssy.fit_transform(ky.reshape(-1,1)).flatten()
        # kx_ss_stats = get_stats(kx_ss)
        # ky_ss_stats = get_stats(ky_ss)

        kx_norm_stats = get_stats(kx_norm)
        ky_norm_stats = get_stats(ky_norm)

        result_xy.append(kx_norm_stats[1] * ky_norm_stats[1])
        result_x.append(kx_norm_stats[1])
        result_y.append(ky_norm_stats[1])
    
    if how == 'x*y':
        topk_index = np.argpartition(result_xy, -topk)[-topk:]
        return pose[:, topk_index, :], topk_index

    elif how == 'x&y':
        topk_index_x = np.argpartition(result_x, -topk)[-topk:]
        topk_index_y = np.argpartition(result_y, -topk)[-topk:]
        topk_index = list(set(topk_index_x).union(topk_index_y))
        return pose[:, topk_index, :], topk_index

    elif how == 'x<>y':
        result_x_topk_sum = np.sum(sorted(result_x, reverse=True)[:topk])
        result_y_topk_sum = np.sum(sorted(result_y, reverse=True)[:topk])
        if result_x_topk_sum > result_y_topk_sum:
            topk_index = np.argpartition(result_x, -topk)[-topk:]
            return pose[:, topk_index, :], topk_index
        else:
            topk_index = np.argpartition(result_y, -topk)[-topk:]
            return pose[:, topk_index, :], topk_index
    
    elif how == 'x':
        topk_index = np.argpartition(result_x, -topk)[-topk:]
        return pose[:, topk_index, :], topk_index
    
    elif how == 'y': 
        topk_index = np.argpartition(result_y, -topk)[-topk:]
        return pose[:, topk_index, :], topk_index

    else:
        return None
