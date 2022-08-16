
import numpy as np

def object_type_conversion(value_obj):      ##object 타입인 bbox가 들어오면 2개의 행렬을 하나로 만들어 반환한다.
    result = []
    for count,obj in enumerate(value_obj):
      if obj.shape[0] >= 2:
        # llen = len(obj)
        # print(llen)
        sum_list = []
        for i in range(len(obj)):
          sum_list.append(obj[i].sum())
        sum_list = sum_list - result[count-1].sum()
        # print(sum_list)
        location = np.where(sum_list == sum_list.min())
        result.append(obj[location])
        # print(np.where(sum_list == sum_list.min()))
          
      else:
        result.append(obj)
    return np.array(result)


##output shpae (-1,1,4)