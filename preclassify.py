from torch.utils import data
from categorizer import Categorizer
from crisismmd_dataset import CrisisMMDataset
from torch.utils.data import DataLoader
from args import get_preclassify_args
from paths import dataroot
import os.path as osp
import os
import pickle
import numpy as np
from tqdm import tqdm


def get_data_loader(opt, phase):
    data_set = CrisisMMDataset()
    data_set.initialize(opt, phase=phase, cat='all',
                        task=opt.task)
    data_loader = DataLoader(
        data_set, batch_size=1, shuffle=True, num_workers=opt.num_workers)  # batch size is hardcoded to be 1

    return data_loader


def get_stats(info_dict):
    """Get mean and std of vectors

    Args:
        info_dict (dict)

    Returns:
        mean and std of shape [Batch]
    """
    vecs = list(info_dict.values())
    vecs = np.array(vecs)

    mean = np.mean(vecs, axis=0)    # Mean over batch
    std = np.std(vecs, axis=0)      # Std over batch

    return mean, std


def normalize(info_dict, mean, std):
    """Normalizes info_dict inplace with scalars mean and std
    """
    for k in info_dict.keys():
        info_dict[k] = (info_dict[k] - mean) / std


def build_category_dict(data_loader, categorizer):
    info_dict = dict()
    for data in tqdm(data_loader, total=len(data_loader)):
        category = get_category(data, categorizer)
        info_dict[data['path_image'][0]] = category

    return info_dict


def get_category(data, categorizer: Categorizer):
    return categorizer.categorize(data)


def dump_file(info_dict, save_dir, file_name):
    save_path = osp.join(save_dir, file_name)

    os.makedirs(save_dir, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(info_dict, f)


if __name__ == '__main__':

    args = get_preclassify_args()
    save_dir = args.save_dir
    categorizer = Categorizer()

    train_loader = get_data_loader(args, 'train')
    info_dict_train = build_category_dict(train_loader, categorizer)
    mean, std = get_stats(info_dict_train)
    normalize(info_dict_train, mean, std)
    dump_file(info_dict_train, save_dir, '{}_info_dict_train.pkl'.format(args.task))

    for phase in ['dev', 'test']:
        loader = get_data_loader(args, phase)
        info_dict = build_category_dict(loader, categorizer)
        normalize(info_dict, mean, std)
        dump_file(info_dict, save_dir, '{}_info_dict_{}.pkl'.format(args.task, phase))

    # dump_file(info_dict)
