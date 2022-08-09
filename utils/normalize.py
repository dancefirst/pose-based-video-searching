import numpy as np
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

def rescale_pose(pose_arr, scaler):
    # ss = StandardScaler()
    return scaler.fit_transform(pose_arr.reshape(-1, pose_arr.shape[-1])).reshape(pose_arr.shape)

def rescale_pose_by_axis(pose_arr, scalers):
    # ss = StandardScaler()    
    pose_arr_x = pose_arr[:,:,:1]
    pose_arr_y = pose_arr[:,:,1:]

    pose_arr_x_scaled = scalers[0].fit_transform(pose_arr_x.reshape(-1, pose_arr_x.shape[-1])).reshape(pose_arr_x.shape)
    pose_arr_y_scaled = scalers[1].fit_transform(pose_arr_y.reshape(-1, pose_arr_y.shape[-1])).reshape(pose_arr_y.shape)

    out = np.concatenate([pose_arr_x_scaled, pose_arr_y_scaled], axis=-1)
    return out

def rescale_pose_by_joint_axis(pose_arr, scalers):
    # ss = StandardScaler()
    out = np.array([])
    for i in range(17): # num of joint = 17
        scalers_cp = copy.deepcopy(scalers)
        # print('pose_arr', pose_arr.shape)
        pose_arr_x = pose_arr[:,i,:1].reshape(pose_arr.shape[0], 1, 1)
        pose_arr_y = pose_arr[:,i,1:].reshape(pose_arr.shape[0], 1, 1)
        # print('pose_arr_x', pose_arr_x.shape)

        pose_arr_x_scaled = scalers_cp[0].fit_transform(pose_arr_x.reshape(-1, pose_arr_x.shape[-1])).reshape(pose_arr_x.shape)
        pose_arr_y_scaled = scalers_cp[1].fit_transform(pose_arr_y.reshape(-1, pose_arr_y.shape[-1])).reshape(pose_arr_y.shape)
        pose_arr_xy_scaled = np.concatenate([pose_arr_x_scaled, pose_arr_y_scaled], axis=-1)
        # print('pose_arr_xy_scaled', pose_arr_xy_scaled.shape)

        if len(out) == 0:
            out = pose_arr_xy_scaled
        else:
            out = np.concatenate([out, pose_arr_xy_scaled], axis=1)
    # print('joint axis', out.shape)
    return out

####################################################
####################################################
####################################################

def vector_normalize(pose_arr, type='l2', axis=None):
    if len(pose_arr.shape) == 3: # (N, 17, 3)
        pose_arr_x = pose_arr[:,:,:1]
        pose_arr_y = pose_arr[:,:,1:]
    elif len(pose_arr.shape) == 2: # (17, 3)
        pose_arr_x = pose_arr[:,:1]
        pose_arr_y = pose_arr[:,1:]

    if type=='l2':
        xnorm = np.linalg.norm(pose_arr_x, 2, axis=axis)
        ynorm = np.linalg.norm(pose_arr_y, 2, axis=axis)
    elif type=='l1':
        xnorm = np.linalg.norm(pose_arr_x, 1, axis=axis)
        ynorm = np.linalg.norm(pose_arr_y, 1, axis=axis)    

    pose_arr_x_norm = pose_arr_x / xnorm
    pose_arr_y_norm = pose_arr_y / ynorm
    
    out = np.concatenate([pose_arr_x_norm, pose_arr_y_norm], axis=-1)
    return out

def vector_normalize2(pose_arr, type='l2', axis=None):

    if type=='l2':
        norm = np.linalg.norm(pose_arr, 2, axis=axis)
    elif type=='l1':
        norm = np.linalg.norm(pose_arr, 1, axis=axis)
        
    pose_arr_norm = pose_arr / norm
    return pose_arr_norm

def vector_normalize_by_joint(pose_arr):
    # ss = StandardScaler()
    out = np.array([])
    for i in range(17): # num of joint = 17
        # print('pose_arr', pose_arr.shape)
        pose_arr_x = pose_arr[:,i,:1].reshape(pose_arr.shape[0], 1, 1)
        pose_arr_y = pose_arr[:,i,1:].reshape(pose_arr.shape[0], 1, 1)
        # print('pose_arr_x', pose_arr_x.shape)

        pose_arr_x_norm = pose_arr_x / np.linalg.norm(pose_arr_x.reshape(pose_arr_x.shape[0], -1))
        pose_arr_y_norm = pose_arr_y / np.linalg.norm(pose_arr_y.reshape(pose_arr_y.shape[0], -1))
        # pose_arr_y_norm = (pose_arr_y.reshape(-1, pose_arr_y.shape[-1])).reshape(pose_arr_y.shape)
        
        pose_arr_xy_norm = np.concatenate([pose_arr_x_norm, pose_arr_y_norm], axis=-1)
        # print('pose_arr_xy_scaled', pose_arr_xy_scaled.shape)

        if len(out) == 0:
            out = pose_arr_xy_norm
        else:
            out = np.concatenate([out, pose_arr_xy_norm], axis=1)
    # print('joint axis', out.shape)
    return out

def normalize_pose(pose_sample, bbox_sample, fname, norm_axis=None):
    result = []

    for pose_, bbox_ in zip(pose_sample, bbox_sample):
        pose = pose_[:,:2]
        bbox = bbox_[0]
        # print(pose)

        bbox_x_min = bbox[0]
        bbox_y_min = bbox[1]
        bbox_x_max = bbox[2]
        bbox_y_max = bbox[3]
        
        pose_alloc = pose - [bbox_x_min, bbox_y_min]

        # img_size = result_info[fname][:2]

        # noramlize by video size
        # pose_norm = pose_alloc / [(bbox_x_max - bbox_x_min), (bbox_y_max - bbox_y_min)]
        
        # result.append(pose_alloc / img_size)
        
        # vector normalize
        # pose_norm = vector_normalize2(pose_alloc)
        # result.append(pose_norm)
        
        result.append(pose)
        # result.append(pose_norm / img_size)

    result = np.array(result)
    # result = vector_normalize2(result, axis=norm_axis)

    # normalize
    result = vector_normalize_by_joint(result)
    # result = vector_normalize2(result, axis=norm_axis)

    #### standardize ####
    scaler1, scaler2 = StandardScaler(), StandardScaler()
    # result = rescale_pose_by_axis(result, (scaler1, scaler2))
    # result = rescale_pose_by_joint_axis(result, (scaler1, scaler2))

    # pose_norm = vector_normalize(result)
    
    # print(fname)

    return result

def standardize_pose(pose_sample, bbox_sample, fname, norm_axis=None):
    result = []

    for pose_, bbox_ in zip(pose_sample, bbox_sample):
        pose = pose_[:,:2]
        bbox = bbox_[0]
        # print(pose)

        bbox_x_min = bbox[0]
        bbox_y_min = bbox[1]
        bbox_x_max = bbox[2]
        bbox_y_max = bbox[3]
        
        pose_alloc = pose - [bbox_x_min, bbox_y_min]

        # img_size = result_info[fname][:2]

        # noramlize by video size
        # pose_norm = pose_alloc / [(bbox_x_max - bbox_x_min), (bbox_y_max - bbox_y_min)]
        
        # result.append(pose_alloc / img_size)
        
        # vector normalize
        # pose_norm = vector_normalize2(pose_alloc)
        # result.append(pose_norm)
        
        result.append(pose)
        # result.append(pose_norm / img_size)

    result = np.array(result)
    # result = vector_normalize2(result, axis=norm_axis)

    # normalize
    # result = vector_normalize_by_joint(result)
    # result = vector_normalize2(result, axis=norm_axis)

    #### standardize ####
    scaler1, scaler2 = StandardScaler(), StandardScaler()
    # result = rescale_pose_by_axis(result, (scaler1, scaler2))
    result = rescale_pose_by_joint_axis(result, (scaler1, scaler2))

    # pose_norm = vector_normalize(result)
    
    # print(fname)

    return result