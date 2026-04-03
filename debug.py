import numpy as np
import os

# read npys from dataset/HumanML3D-E
data_list = [f for f in os.listdir('dataset/HumanML3D-E') if f.endswith('.npy')]

read_data = []
# read each npy file
for data_file in data_list:
    data = np.load(os.path.join('dataset/HumanML3D-E', data_file), allow_pickle=True)
    # add filename to data dict
    data_dict = {
        'filename': data_file,
        'data': data
    }
    read_data.append(data_dict)
    
print(len(read_data))