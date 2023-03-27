import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import copy
import random

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def load_data(partition):
#     download()
    BASE_DIR = '/home//scratch1link/MVdata'
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    if partition == 'validate':
        partition = 'train'
    if partition == 'validate_train':
        partition = 'train'
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.rand()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud

def random_scale(pointcloud, scale_low=0.8, scale_high=1.25):
    N, C = pointcloud.shape
    scale = np.random.uniform(scale_low, scale_high)
    pointcloud = pointcloud*scale
    return pointcloud



class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', data_split='labeled', perceptange = 10):
        data, label = load_data(partition)
        self.num_points = num_points
        self.partition = partition

        if self.partition == 'train':
            labeled_sample_num = int(len(label) * perceptange / 100.0)
            unlabeled_sample_num = len(label) - labeled_sample_num

            if data_split == 'labeled':
                self.data, self.label = data[unlabeled_sample_num:, :, :], label[unlabeled_sample_num:]
            else:
                self.data, self.label = data[:unlabeled_sample_num, :, :], label[:unlabeled_sample_num]
                
        elif self.partition == 'validate':
            labeled_sample_num = int(len(label) * perceptange / 100.0)
            unlabeled_sample_num = len(label) - labeled_sample_num
            self.data, self.label = data[:unlabeled_sample_num, :, :], label[:unlabeled_sample_num]
        else:
            self.data, self.label = data, label


    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pt = copy.deepcopy(pointcloud)
            non_aug_pt = pt 

            pointcloud = translate_pointcloud(pointcloud)

#             pointcloud_strongaug = random_scale(pt, scale_low=0.8, scale_high=1.2)
#             pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)
#             pointcloud_strongaug = rotate_pointcloud(pointcloud_strongaug)
#             pointcloud_strongaug = jitter_pointcloud(pointcloud_strongaug)
            
            aug_num = np.random.randint(2, high=5)
            aug_list = random.sample(range(4), aug_num)
            pointcloud_strongaug = pt
            if 0 in aug_list:
                pointcloud_strongaug = random_scale(pointcloud_strongaug, scale_low=0.8, scale_high=1.2)
            if 1 in aug_list:
                pointcloud_strongaug = translate_pointcloud(pointcloud_strongaug)
            if 2 in aug_list:
                pointcloud_strongaug = rotate_pointcloud(pointcloud_strongaug)
            if 3 in aug_list:
                pointcloud_strongaug = jitter_pointcloud(pointcloud_strongaug)
            

            np.random.shuffle(non_aug_pt)
            np.random.shuffle(pointcloud)
            np.random.shuffle(pointcloud_strongaug)


            return non_aug_pt, pointcloud, pointcloud_strongaug, label
        else:
            return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
