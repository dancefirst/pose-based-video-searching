
from tqdm import tqdm
import os
import numpy as np


def box_scaling(bboxes,poses,pose_paths,bbox_paths):   #박스 좌표와 포즈 좌표 포즈 경로와 박스 경로를 넣으면 padding를 적용하고 딕셔너리 형태로 반환합니다. 
                                                        #예시가 아래 있습니다.                                              
    pose_dict = {}
    obj_count = 0

    for i, (ppath, bbox_path) in enumerate(zip(pose_paths,bbox_paths)):
        fname = os.path.basename(ppath).split('_')[0].split('-')[0]
        if fname not in pose_dict:  
            pose_dict[fname] = {'x':None, 'y':None, 'w':None,'h':None}

        if bboxes[i].dtype == 'object':                                      
            bbox = np.concatenate(list(bboxes[i]), axis=0).reshape(-1, 1, 4)                  
        else:
            bbox = bboxes[i]
            
        pose = poses[i]

        x,y = [], []
        for i,(one_bbox,one_pose) in enumerate(zip(bbox,pose)):
            bbox_xmins, bbox_ymins = one_bbox[..., 0], one_bbox[..., 1]              #몸박스의 x값과 y값
            bbox_width, bbox_height = one_bbox[..., 2] - one_bbox[..., 0], one_bbox[..., 3] - one_bbox[..., 1]
            if bbox_height > bbox_width:
                distance = bbox_height - bbox_width
                bbox_width += distance
                padding = distance // 2
                
                ppose_x = one_pose[:,0] - bbox_xmins + padding
                ppose_y = one_pose[:,1] - bbox_ymins 
                
            else:
                distance = bbox_width - bbox_height
                bbox_height += distance
                padding = distance // 2

                ppose_x = one_pose[:,0] - bbox_xmins
                ppose_y = one_pose[:,1] - bbox_ymins + padding
            
            x.append(ppose_x)
            y.append(ppose_y)
            
        pose_dict[fname]['x'], pose_dict[fname]['y']= x, y
            

    return pose_dict


#예시
        
# poseyy = box_scaling(bboxes,poses,pose_paths,bbox_paths)