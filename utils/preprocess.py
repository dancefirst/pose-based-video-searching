import numpy as np
from scipy.stats import kurtosis, skew
import math
import numpy as np

def pose_preprocess_v2(pose, bbox, outlier_frame_interp=False, smooth_coef=0.0, normalize=True):

    ppose = pose.copy()
    bbox = bbox.copy()

    bbox_xmins, bbox_ymins = bbox[..., 0], bbox[..., 1]
    bbox_xmaxs, bbox_ymaxs = bbox[..., 2], bbox[..., 3]
    bbox_widths, bbox_heights = bbox[..., 2] - bbox[..., 0], bbox[..., 3] - bbox[..., 1]
    bbox_max_width = np.max(bbox_widths)
    bbox_max_height = np.max(bbox_heights)

    # bbox와 joint 좌표 간의 불일치 발생 -> 흐린 이미지 등 비정상 데이터일 가능성 높음
    if outlier_frame_interp:
        ppose_x = ppose[..., 0]
        ppose_y = ppose[..., 1]
        ppose_x_above_box_index = np.stack(np.where(ppose_x > bbox_xmaxs), -1)[...,0]
        ppose_y_above_box_index = np.stack(np.where(ppose_y > bbox_ymaxs), -1)[...,0]
        ppose_x_below_box_index = np.stack(np.where(ppose_x < bbox_xmins), -1)[...,0]
        ppose_y_below_box_index = np.stack(np.where(ppose_y < bbox_ymins), -1)[...,0]
        ppose_xy_outlier_index = sorted(list(set().union(*[ppose_x_above_box_index,ppose_y_above_box_index,ppose_x_below_box_index,ppose_y_below_box_index])))
        ppose = apply_outlier_interp(ppose, ppose_xy_outlier_index)
    
    if smooth_coef:
        ppose = apply_one_euro_filter(ppose, smooth_coef=3.0)

    ppose_out_x = ppose[...,0]
    ppose_out_y = ppose[...,1]

    # create new bbox coordinates
    ppose_xmins, ppose_xmaxs = np.min(ppose_out_x, axis=1), np.max(ppose_out_x, axis=1)
    ppose_ymins, ppose_ymaxs = np.min(ppose_out_y, axis=1), np.max(ppose_out_y, axis=1)
    ppose_widths, ppose_heights = ppose_xmaxs - ppose_xmins, ppose_ymaxs - ppose_ymins

    ppose_xpaddings = ppose_widths * .2
    ppose_ypaddings = ppose_heights * .2

    new_bbox = np.stack([(ppose_xmins - ppose_xpaddings), 
                        (ppose_ymins - ppose_ypaddings), 
                        (ppose_xmaxs + ppose_xpaddings), 
                        (ppose_ymaxs + ppose_ypaddings)],
                        axis=1)
    new_bbox_width_height = np.stack([
                                        new_bbox[...,2] - new_bbox[...,0],
                                        new_bbox[...,3] - new_bbox[...,1]
                                    ], axis=-1)

    ppose_out_x -= new_bbox[:, 0].reshape(-1,1) # set origin to 0
    ppose_out_y -= new_bbox[:, 1].reshape(-1,1) # set origin to 0

    ppose_out_x_scale, ppose_out_y_scale = rescale_by_bbox(ppose_out_x, 
                                                            ppose_out_y, 
                                                            new_bbox_width_height, 
                                                            verbose=0)
    ppose_xy = np.stack([ppose_out_x_scale, ppose_out_y_scale], axis=-1).reshape(-1,34) # (num_frame, 34)

    if normalize:
        ppose_xy = np.apply_along_axis(l2_norm, 1, ppose_xy)

    return ppose_xy


def pose_preprocess(pose, bbox, outlier_frame_interp=False, smooth_coef=0.0, normalize=True):

    ppose = pose.copy()
    bbox = bbox.copy()

    bbox_xmins, bbox_ymins = bbox[..., 0], bbox[..., 1]
    bbox_xmaxs, bbox_ymaxs = bbox[..., 2], bbox[..., 3]
    bbox_widths, bbox_heights = bbox[..., 2] - bbox[..., 0], bbox[..., 3] - bbox[..., 1]
    bbox_max_width = np.max(bbox_widths)
    bbox_max_height = np.max(bbox_heights)

    ppose_x = ppose[..., 0]
    ppose_y = ppose[..., 1]

    if outlier_frame_interp:
        ppose_x_above_box_index = np.stack(np.where(ppose_x > bbox_xmaxs), -1)[...,0]
        ppose_y_above_box_index = np.stack(np.where(ppose_y > bbox_ymaxs), -1)[...,0]
        ppose_x_below_box_index = np.stack(np.where(ppose_x < bbox_xmins), -1)[...,0]
        ppose_y_below_box_index = np.stack(np.where(ppose_y < bbox_ymins), -1)[...,0]
        ppose_xy_outlier_index = sorted(list(set().union(*[ppose_x_above_box_index,ppose_y_above_box_index,ppose_x_below_box_index,ppose_y_below_box_index])))
        ppose = apply_outlier_interp(ppose, ppose_xy_outlier_index)
    
    if smooth_coef:
        ppose = apply_one_euro_filter(ppose, smooth_coef=3.0)

    ppose_out_x = ppose[...,0]
    ppose_out_y = ppose[...,1]

    # create new bbox coordinates
    ppose_xmins, ppose_xmaxs = np.min(ppose_out_x, axis=1), np.max(ppose_out_x, axis=1)
    ppose_ymins, ppose_ymaxs = np.min(ppose_out_y, axis=1), np.max(ppose_out_y, axis=1)
    # print(bbox_xmins.shape, ppose_xmins.shape, ppose_ymins.shape)

    ## padding size -> must be decided in proportion to the bbox size.
    padding = np.mean(
        [
        (ppose_xmins.flatten() - bbox_xmins.flatten()).mean(),
        (ppose_ymins.flatten() - bbox_ymins.flatten()).mean(),
        (bbox_xmaxs.flatten() - ppose_xmaxs.flatten()).mean(),
        (bbox_ymaxs.flatten() - ppose_ymaxs.flatten()).mean()
        ]
    )

    new_bbox = np.stack([(ppose_xmins - padding), (ppose_ymins - padding), (ppose_xmaxs + padding), (ppose_ymaxs + padding)], axis=1)
    new_bbox_width_height = np.stack([(ppose_xmaxs-ppose_xmins+padding*2), 
                                        (ppose_ymaxs-ppose_ymins+padding*2)], axis=-1)

    ppose_out_x -= new_bbox[:, 0].reshape(-1,1) # set origin to 0
    ppose_out_y -= new_bbox[:, 1].reshape(-1,1) # set origin to 0

    ppose_out_x_scale, ppose_out_y_scale = rescale_by_bbox(ppose_out_x, 
                                                            ppose_out_y, 
                                                            new_bbox_width_height, 
                                                            verbose=0)
    ppose_xy = np.stack([ppose_out_x_scale, ppose_out_y_scale], axis=-1).reshape(-1,34) # (num_frame, 34)
    if normalize:
        ppose_xy = np.apply_along_axis(l2_norm, 1, ppose_xy)

    return ppose_xy

def l2_norm(v):
    return v / np.sqrt(np.sum(v**2))

def make_outlier_index_dict(out_index):
    result = {}
    for fnum, jnum in out_index:
        if fnum not in result:
            result[fnum] = []
        result[fnum].append(jnum)
    return result

def get_successive_groups(num_list):
    result = []
    prev_num = -1
    for num in num_list:
        if prev_num == -1:
            prev_num = num
            result.append([num])
        elif prev_num + 1 == num:
            result[-1].append(num)
            prev_num = num
        else:
            result.append([num])
            prev_num = num

    return result

def apply_outlier_interp(pose, outlier_list):

    ppose_copy = pose.copy()
    outlier_group = get_successive_groups(outlier_list)

    for out_group in outlier_group:
        start_index = out_group[0] - 1
        end_index = out_group[-1] + 1

        # print(start_index, end_index, len(pose)-1)

        if (start_index >= 0) & (end_index <= len(pose) - 1):
            num_to_interp = end_index - start_index + 1
            ppose_copy[start_index:end_index+1, ...] = np.linspace(pose[start_index, ...],
                                                                pose[end_index, ...],
                                                                num_to_interp)
    return ppose_copy
    # print(ppose_copy[start_index:end_index+1, ...].shape)
    # print(np.linspace(ppose[start_index, ...],ppose[end_index, ...], num_to_interp).shape)


def apply_one_euro_filter(pose, smooth_coef = 3.0):
    filter = OneEuroFilterROMP(smooth_coef, 0)
    ppose_filter = pose.copy()

    for i, framepose in enumerate(pose):
        ppose_filter[i] = filter.process(framepose)
    return ppose_filter

def rescale_by_bbox(posex, posey, bbox_width_height,
                            verbose=0):
    
    posex_scale = posex / bbox_width_height[:, 0:1]
    posey_scale = posey / bbox_width_height[:, 1:2]
    
    if verbose:
        print(posex_scale.min(), posex_scale.max())
        print(posey_scale.min(), posey_scale.max())
    return posex_scale, posey_scale


class LowPassFilter:
  def __init__(self):
    self.prev_raw_value = None
    self.prev_filtered_value = None

  def process(self, value, alpha):
    if self.prev_raw_value is None:
      s = value
    else:
      s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
    self.prev_raw_value = value
    self.prev_filtered_value = s
    return s

class OneEuroFilterROMP:
  def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
    self.freq = freq
    self.mincutoff = mincutoff
    self.beta = beta
    self.dcutoff = dcutoff
    self.x_filter = LowPassFilter()
    self.dx_filter = LowPassFilter()

  def compute_alpha(self, cutoff):
    te = 1.0 / self.freq
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / te)

  def process(self, x, print_inter=False):
    prev_x = self.x_filter.prev_raw_value
    dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
    edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
    
    cutoff = self.mincutoff + self.beta * np.abs(edx)
    if print_inter:
      print(self.compute_alpha(cutoff))
    return self.x_filter.process(x, self.compute_alpha(cutoff))


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


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
