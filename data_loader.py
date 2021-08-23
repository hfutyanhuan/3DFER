import torch.utils.data as data
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

###
#    NJU3DFE
###
class NJU3DFE_multichannel(data.Dataset):
    def __init__(self, subset):
        """Dataset class representing NJU3DFE dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('train', 'test'):
            raise(ValueError, 'subset must be one of (train, test)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))
        self.df = self.df[~self.df['class_name']
            .isin(['11_chin_raiser.png', '16_grin.png', '17_cheek_blowing.png', '20_brow_lower.png'])]\
            .reset_index(drop=True)
        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_t_filepath = self.df.to_dict()['filepath_t']
        self.datasetid_to_c_filepath = self.df.to_dict()['filepath_c']
        self.datasetid_to_d_filepath = self.df.to_dict()['filepath_d']
        self.datasetid_to_nx_filepath = self.df.to_dict()['filepath_nx']
        self.datasetid_to_ny_filepath = self.df.to_dict()['filepath_ny']
        self.datasetid_to_nz_filepath = self.df.to_dict()['filepath_nz']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
        ])

    def __getitem__(self, item):
        imgs_t = Image.open(self.datasetid_to_t_filepath[item])
        imgs_c = Image.open(self.datasetid_to_c_filepath[item])
        imgs_d = Image.open(self.datasetid_to_d_filepath[item])
        imgs_nx = Image.open(self.datasetid_to_nx_filepath[item])
        imgs_ny = Image.open(self.datasetid_to_ny_filepath[item])
        imgs_nz = Image.open(self.datasetid_to_nz_filepath[item])
        imgs_t = np.asarray(self.transform(imgs_t))
        imgs_c = np.asarray(self.transform(imgs_c))
        imgs_d = np.asarray(self.transform(imgs_d))
        imgs_nx = np.asarray(self.transform(imgs_nx))
        imgs_ny = np.asarray(self.transform(imgs_ny))
        imgs_nz = np.asarray(self.transform(imgs_nz))


        imgs = np.append(imgs_t, imgs_c[0:1], 0)
        imgs = np.append(imgs, imgs_d[0:1], 0)
        imgs = np.append(imgs, imgs_nx[0:1], 0)
        imgs = np.append(imgs, imgs_ny[0:1], 0)
        imgs = np.append(imgs, imgs_nz[0:1], 0)
        imgs = torch.tensor(imgs)

        label = self.datasetid_to_class_id[item]
        return imgs,label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):

        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(r'E:\yanhuan\datasets\NJU-3DFE_database\NJUdatabase_split_{}\T'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        for root, folders, files in os.walk(r'E:\yanhuan\datasets\NJU-3DFE_database\NJUdatabase_split_{}\T'.format(subset)):
            if len(files) == 0:
                continue

            for f in files:
                images.append({
                    'subset': subset,

                    'class_name': f,
                    'filepath_t': os.path.join(root, f),
                    'filepath_d': os.path.join(r'E:\yanhuan\datasets\NJU-3DFE_database\NJUdatabase_split_{}\Depth_map'.format(subset),
                                               root.split('\\')[-1], f),
                    'filepath_c': os.path.join(r'E:\yanhuan\datasets\NJU-3DFE_database\NJUdatabase_split_{}\C_map'.format(subset),
                                               root.split('\\')[-1], f),
                    'filepath_nx': os.path.join(r'E:\yanhuan\datasets\NJU-3DFE_database\NJUdatabase_split_{}\Nx_map'.format(subset),
                                                root.split('\\')[-1], f),
                    'filepath_ny': os.path.join(r'E:\yanhuan\datasets\NJU-3DFE_database\NJUdatabase_split_{}\Ny_map'.format(subset),
                                                root.split('\\')[-1], f),
                    'filepath_nz': os.path.join(r'E:\yanhuan\datasets\NJU-3DFE_database\NJUdatabase_split_{}\Nz_map'.format(subset),
                                                root.split('\\')[-1], f)

                })

        return images

class NJU3DFE_singlechannel(data.Dataset):
    def __init__(self, subset):
        """Dataset class representing Omniglot dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('train', 'test'):
            raise(ValueError, 'subset must be one of (train, test)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))
        self.df = self.df[~self.df['class_name']
            .isin(['11_chin_raiser.png', '16_grin.png', '17_cheek_blowing.png', '20_brow_lower.png'])]\
            .reset_index(drop=True)
        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_t_filepath = self.df.to_dict()['filepath_t']
        self.datasetid_to_c_filepath = self.df.to_dict()['filepath_c']
        self.datasetid_to_d_filepath = self.df.to_dict()['filepath_d']
        self.datasetid_to_nx_filepath = self.df.to_dict()['filepath_nx']
        self.datasetid_to_ny_filepath = self.df.to_dict()['filepath_ny']
        self.datasetid_to_nz_filepath = self.df.to_dict()['filepath_nz']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
        ])

    def __getitem__(self, item):
        imgs = Image.open(self.datasetid_to_t_filepath[item])
        imgs = self.transform(imgs)

        label = self.datasetid_to_class_id[item]
        return imgs,label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):

        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(r'E:\yanhuan\datasets\NJU-3DFE_database\NJUdatabase_split_{}\T'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        for root, folders, files in os.walk(r'E:\yanhuan\datasets\NJU-3DFE_database\NJUdatabase_split_{}\T'.format(subset)):
            if len(files) == 0:
                continue

            for f in files:
                images.append({
                    'subset': subset,

                    'class_name': f,
                    'filepath_t': os.path.join(root, f),
                    'filepath_d': os.path.join(r'E:\yanhuan\datasets\NJU-3DFE_database\NJUdatabase_split_{}\Depth_map'.format(subset),
                                               root.split('\\')[-1], f),
                    'filepath_c': os.path.join(r'E:\yanhuan\datasets\NJU-3DFE_database\NJUdatabase_split_{}\C_map'.format(subset),
                                               root.split('\\')[-1], f),
                    'filepath_nx': os.path.join(r'E:\yanhuan\datasets\NJU-3DFE_database\NJUdatabase_split_{}\Nx_map'.format(subset),
                                                root.split('\\')[-1], f),
                    'filepath_ny': os.path.join(r'E:\yanhuan\datasets\NJU-3DFE_database\NJUdatabase_split_{}\Ny_map'.format(subset),
                                                root.split('\\')[-1], f),
                    'filepath_nz': os.path.join(r'E:\yanhuan\datasets\NJU-3DFE_database\NJUdatabase_split_{}\Nz_map'.format(subset),
                                                root.split('\\')[-1], f)
                })

        return images


class Bosphorus_multichannel(data.Dataset):
    def __init__(self, subset):
        """Dataset class representing Omniglot dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('train', 'test'):
            raise(ValueError, 'subset must be one of (train, test)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))
        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_t_filepath = self.df.to_dict()['filepath_t']
        self.datasetid_to_c_filepath = self.df.to_dict()['filepath_c']
        self.datasetid_to_d_filepath = self.df.to_dict()['filepath_d']
        self.datasetid_to_nx_filepath = self.df.to_dict()['filepath_nx']
        self.datasetid_to_ny_filepath = self.df.to_dict()['filepath_ny']
        self.datasetid_to_nz_filepath = self.df.to_dict()['filepath_nz']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize([112,112]),
            transforms.ToTensor(),
        ])

    def __getitem__(self, item):
        imgs_t = Image.open(self.datasetid_to_t_filepath[item])
        imgs_c = Image.open(self.datasetid_to_c_filepath[item])
        imgs_d = Image.open(self.datasetid_to_d_filepath[item])
        imgs_nx = Image.open(self.datasetid_to_nx_filepath[item])
        imgs_ny = Image.open(self.datasetid_to_ny_filepath[item])
        imgs_nz = Image.open(self.datasetid_to_nz_filepath[item])
        imgs_t = np.asarray(self.transform(imgs_t))
        imgs_c = np.asarray(self.transform(imgs_c))
        imgs_d = np.asarray(self.transform(imgs_d))
        imgs_nx = np.asarray(self.transform(imgs_nx))
        imgs_ny = np.asarray(self.transform(imgs_ny))
        imgs_nz = np.asarray(self.transform(imgs_nz))


        imgs = np.append(imgs_t, imgs_c[0:1], 0)
        imgs = np.append(imgs, imgs_d[0:1], 0)
        imgs = np.append(imgs, imgs_nx[0:1], 0)
        imgs = np.append(imgs, imgs_ny[0:1], 0)
        imgs = np.append(imgs, imgs_nz[0:1], 0)
        imgs = torch.tensor(imgs)

        label = self.datasetid_to_class_id[item]
        return imgs,label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):

        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(r'E:\yanhuan\datasets\Bosphorus_database\Bosphorus_{}\T'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        for root, folders, files in os.walk(r'E:\yanhuan\datasets\Bosphorus_database\Bosphorus_{}\T'.format(subset)):
            if len(files) == 0:
                continue

            for f in files:
                images.append({
                    'subset': subset,

                    'class_name': f,
                    'filepath_t': os.path.join(root, f),
                    'filepath_d': os.path.join(r'E:\yanhuan\datasets\Bosphorus_database\Bosphorus_{}\Depth_map'.format(subset),
                                               root.split('\\')[-1], f),
                    'filepath_c': os.path.join(r'E:\yanhuan\datasets\Bosphorus_database\Bosphorus_{}\C_map'.format(subset),
                                               root.split('\\')[-1], f),
                    'filepath_nx': os.path.join(r'E:\yanhuan\datasets\Bosphorus_database\Bosphorus_{}\Nx_map'.format(subset),
                                                root.split('\\')[-1], f),
                    'filepath_ny': os.path.join(r'E:\yanhuan\datasets\Bosphorus_database\Bosphorus_{}\Ny_map'.format(subset),
                                                root.split('\\')[-1], f),
                    'filepath_nz': os.path.join(r'E:\yanhuan\datasets\Bosphorus_database\Bosphorus_{}\Nz_map'.format(subset),
                                                root.split('\\')[-1], f)

                })

        return images

class Bosphorus_T(data.Dataset):
    def __init__(self, subset):
        """Dataset class representing Omniglot dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('train', 'test'):
            raise(ValueError, 'subset must be one of (train, test)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))
        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_t_filepath = self.df.to_dict()['filepath_t']
        self.datasetid_to_c_filepath = self.df.to_dict()['filepath_c']
        self.datasetid_to_d_filepath = self.df.to_dict()['filepath_d']
        self.datasetid_to_nx_filepath = self.df.to_dict()['filepath_nx']
        self.datasetid_to_ny_filepath = self.df.to_dict()['filepath_ny']
        self.datasetid_to_nz_filepath = self.df.to_dict()['filepath_nz']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
        ])

    def __getitem__(self, item):
        imgs_t = Image.open(self.datasetid_to_t_filepath[item])
        imgs_t = self.transform(imgs_t)
        label = self.datasetid_to_class_id[item]
        return imgs_t,label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):

        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(r'E:\yanhuan\datasets\Bosphorus_database\Bosphorus_{}\T'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        for root, folders, files in os.walk(r'E:\yanhuan\datasets\Bosphorus_database\Bosphorus_{}\T'.format(subset)):
            if len(files) == 0:
                continue

            for f in files:
                images.append({
                    'subset': subset,

                    'class_name': f,
                    'filepath_t': os.path.join(root, f),
                    'filepath_d': os.path.join(r'E:\yanhuan\datasets\Bosphorus_database\Bosphorus_{}\Depth_map'.format(subset),
                                               root.split('\\')[-1], f),
                    'filepath_c': os.path.join(r'E:\yanhuan\datasets\Bosphorus_database\Bosphorus_{}\C_map'.format(subset),
                                               root.split('\\')[-1], f),
                    'filepath_nx': os.path.join(r'E:\yanhuan\datasets\Bosphorus_database\Bosphorus_{}\Nx_map'.format(subset),
                                                root.split('\\')[-1], f),
                    'filepath_ny': os.path.join(r'E:\yanhuan\datasets\Bosphorus_database\Bosphorus_{}\Ny_map'.format(subset),
                                                root.split('\\')[-1], f),
                    'filepath_nz': os.path.join(r'E:\yanhuan\datasets\Bosphorus_database\Bosphorus_{}\Nz_map'.format(subset),
                                                root.split('\\')[-1], f)

                })

        return images



