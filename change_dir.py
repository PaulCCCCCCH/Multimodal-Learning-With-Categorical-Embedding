import pickle
import os.path as osp
import re

dataroot_old = 'W:/datasets'
dataroot_new = './datasets'

for phase in ['train', 'dev', 'test']:
    filepath = osp.join(dataroot_new, 'CrisisMMD_extra',
                        'info_dict_{}.pkl'.format(phase))
    with open(filepath, 'rb') as f:
        info_dict = pickle.load(f)

    new_dict = {}
    for k, v in info_dict.items():
        newk = re.sub(dataroot_old, dataroot_new, k)
        new_dict[newk] = v

    with open(filepath, 'wb') as f:
        pickle.dump(new_dict, f)
