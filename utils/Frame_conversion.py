import numpy as np


def Frame_extraction(data, target):     ##넓게 들어온 조인트 데이터를 target 값으로 축소합니다.
    
    output_data = np.zeros((target, data.shape[1], data.shape[2]))

    Interval = (len(data) // target)

    for count,i in enumerate(range(0,len(data), Interval)): #일정한 프레임 추출을 위함.
        if count == target: break
        output_data[count] = data[i]
    return output_data



def Stretch_Frames(data, target = 256):             #이 함수를 호출하면 위 Frame_extraction 를 자동으로 호출합니다.
                                                #조인트 와 만들고 싶은 크기 를 입력합니다.
      #output 값을 생성함.

    output_data = np.zeros((target, data.shape[1], data.shape[2]))
    shapes = data.shape
    result = []
    for i in range(0,len(data)):
        if i == len(data)-1:
            break
        if i == len(data) -2 :
            data_linspace = np.linspace(data[i],data[i+1],target,endpoint=True)
        else:
            data_linspace = np.linspace(data[i],data[i+1],target,endpoint=False)
        result.append(data_linspace)
    result = np.reshape(result,(-1,shapes[1],shapes[2]))
    output_data = Frame_extraction(result,target)
    return output_data
