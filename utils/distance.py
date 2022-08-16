from scipy.spatial.distance import cosine
import numpy as np
import math


def weighted_distance_matching(pose1, pose2):
    # input pose = single frame joints with confidence score, 1d array (52,)
    pose1_xy = pose1[:34]
    pose1_conf = pose1[34:51]
    pose1_conf_sum = pose1[51:52]

    # first summation
    summation1 = 1 / pose1_conf_sum

    # second summation
    summation2 = 0
    for i in range(len(pose1_xy)): # iterate for each joint
        temp_conf = math.floor(i / 2) # x,y 2개에 대해 동일 컨피던스 적용
        temp_sum = pose1_conf[temp_conf] * abs(pose1[i] - pose2[i])
        summation2 += temp_sum
    return summation1 * summation2

# def euc_cosine_dist(p1, p2):
#     cosine_sim = cosine(p1[:34], p2[:34])
#     distance = 2 * (1 - cosine_sim)
#     return np.sqrt(1 - distance)

def simple_minus(p1,p2):
    return np.sum(abs(p1[:34]-p2[:34]))

def simple_euc(p1, p2):
    return np.linalg.norm(p1[:34] - p2[:34])

def simple_cosine(p1, p2):
    return cosine(p1[:34], p2[:34])


# def cosine_dist_xy(pose1, pose2):
#     # input shape = (N, 2)
#     if len(pose1) != len(pose2):
#         which_is_bigger = 0 if len(pose1) > len(pose2) else 1
#         target_len = len([pose1, pose2][1 - which_is_bigger])
#         pose1, pose2 = pose1[:target_len], pose2[:target_len]

#     dist = []
#     for p1, p2 in zip(pose1, pose2):
#         try:
#             dist.append(cosine(p1, p2))
#         except:
#             dist.append(cosine(p1.flatten(), p2.flatten()))

#     return np.mean(dist)

def cosine_dist_xy_flatten(pose1, pose2):
    # input shape = (N, 2)
    if len(pose1) != len(pose2):
        which_is_bigger = 0 if len(pose1) > len(pose2) else 1
        target_len = len([pose1, pose2][1 - which_is_bigger])
        pose1, pose2 = pose1[:target_len], pose2[:target_len]

    return cosine(pose1.flatten(), pose2.flatten())