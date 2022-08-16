from tqdm import tqdm
import os
import numpy as np


def box_scaling_hw(bboxes,poses,pose_paths,bbox_paths):      
    
    pose_dict = {}
    obj_count = 0

    for i, (ppath, bbox_path) in tqdm(enumerate(zip(pose_paths,bbox_paths))):   #박스 좌표와 포즈 좌표 포즈 경로와 박스 경로를 넣으면 padding를 적용하고 딕셔너리 형태로 반환합니다. 
                                                                                #예시가 아래 있습니다.


        fname = os.path.basename(ppath).split('_')[0].split('-')[0]
        if fname not in pose_dict:  
            pose_dict[fname] = {'x':None, 'y':None, 'w':None,'h':None,'xy':None}

        if bboxes[i].dtype == 'object':                                      
            bbox = np.concatenate(list(bboxes[i]), axis=0).reshape(-1, 1, 4) 
            obj_count += 1                 
        else:
            bbox = bboxes[i]

        pose = poses[i]

        x,y,h,w,xy = [], [],[],[],[]
        for i,(one_bbox,one_pose) in enumerate(zip(bbox,pose)):
            bbox_xmins, bbox_ymins = one_bbox[..., 0], one_bbox[..., 1]              #몸박스의 x값과 y값
            bbox_width, bbox_height = one_bbox[..., 2] - one_bbox[..., 0], one_bbox[..., 3] - one_bbox[..., 1]

            if bbox_height > bbox_width:
                distance = bbox_height - bbox_width
                bbox_width += distance
                padding = distance // 2
                
                bbox_h = bbox_height
                bbox_w = bbox_width
                ppose_x = one_pose[:,0] - bbox_xmins + padding
                ppose_y = one_pose[:,1] - bbox_ymins 
                ppose_x /= bbox_w
                ppose_y /= bbox_h                
                ppose_xy = np.stack([ppose_x, ppose_y], axis=-1)

            else:
                distance = bbox_width - bbox_height
                bbox_height += distance
                padding = distance // 2
                
                bbox_h = bbox_height
                bbox_w = bbox_width
                ppose_x = one_pose[:,0] - bbox_xmins
                ppose_y = one_pose[:,1] - bbox_ymins + padding
                ppose_x /= bbox_w
                ppose_y /= bbox_h
                ppose_xy = np.stack([ppose_x, ppose_y], axis=-1)

            x.append(ppose_x)
            y.append(ppose_y)
            h.append(bbox_h)
            w.append(bbox_w)
            xy.append(ppose_xy)
        
            
        pose_dict[fname]['x'], pose_dict[fname]['y'],pose_dict[fname]['h'], pose_dict[fname]['w'],pose_dict[fname]['xy']= x, y, h, w,xy
            

    return pose_dict
       
       

#예시
        
# poseyy = box_scaling(bboxes,poses,pose_paths,bbox_paths)