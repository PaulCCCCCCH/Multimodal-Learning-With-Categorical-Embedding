import pickle
import os.path as osp
import re

base_dir = 'W:/datasets/'
task1_selection = [0] + [2 + x for x in [0, 13, 12, 2, 4, 1,
                                         5, 21, 43, 3, 14, 40, 94, 140, 82, 76, 134, 98, 136, 6]]
task2_selection = [2 + x for x in [15, 14, 3, 6, 2, 43,
                                   96, 117, 4, 5, 8, 51, 143, 7, 64, 23, 131, 87, 24, 11]]

selection = {'task1': task1_selection, 'task2': task2_selection}

for task in ['task1', 'task2']:
    for phase in ['train', 'dev', 'test']:
        filepath = osp.join(base_dir, 'CrisisMMD_extra',
                            '{}_info_dict_{}.pkl'.format(task, phase))
        newfilepath = osp.join(base_dir, 'CrisisMMD_extra_selected',
                               '{}_info_dict_{}.pkl'.format(task, phase))

        with open(filepath, 'rb') as f:
            info_dict = pickle.load(f)

        new_dict = {}
        for k, v in info_dict.items():
            new_dict[k] = v[selection[task]]

        with open(newfilepath, 'wb') as f:
            pickle.dump(new_dict, f)
