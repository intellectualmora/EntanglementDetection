import json
import time
import os
import random

import h5py
import numpy as np
import torch
import torch.utils.data
from scipy.io import loadmat
from torch.utils.data import DataLoader
from modules.utils import CONFIG_PATH, get_hparams


class StaticDataset(torch.utils.data.Dataset):
    def __init__(self, hps, data_type):
        if data_type == "train":
            self.dataset_path = hps.data.train_dataset_path
            self.size = int(hps.data.train_size)
        elif data_type == "eval":
            self.dataset_path = hps.data.eval_dataset_path
            self.size = int(hps.data.eval_size)
        elif data_type == "test":
            self.dataset_path = hps.data.test_dataset_path
            self.size = int(hps.data.test_size)
        self.mode = hps.data.mode
        self.n_qubit = int(hps.data.n_qubit)
        if self.mode == "json":
            datalist = os.listdir(self.dataset_path)
            for name in datalist:
                file = open(os.path.join(self.dataset_path, name), 'r')
                content = file.read()
                sample = json.loads(content)
                self.input_dim = np.array(sample["expect_value_psis"]).shape[1]
                self.output_dim = np.array(sample["entropys"]).shape[1]
                break
            self.input = torch.zeros([self.size, self.input_dim])
            self.output = torch.zeros([self.size, self.output_dim])
            counter = 0
            assert self.size % hps.data.batch_size == 0, "batch size error!"
            for name in datalist:
                file = open(self.dataset_path + '/' + name, 'r')
                content = file.read()
                sample = json.loads(content)
                self.input[counter:counter + hps.data.batch_size] = torch.tensor(np.array(sample["expect_value_psis"]))
                self.output[counter:counter + hps.data.batch_size] = torch.tensor(np.array(sample["entropys"]))
                counter += hps.data.batch_size
                if counter == self.size:
                    break
        elif self.mode == "mat":
            try:
                sample = h5py.File(self.dataset_path, 'r')
                self.input = torch.tensor(np.array(sample['inputdata']).T,dtype=torch.float32)[:self.size]
                self.output = torch.tensor(np.array(sample['outdata']).T,dtype=torch.float32)[:self.size]
            except:
                sample = loadmat(self.dataset_path)
                self.input = torch.tensor(np.array(sample['inputdata']),dtype=torch.float32)[:self.size]
                self.output = torch.tensor(np.array(sample['outdata']),dtype=torch.float32)[:self.size]
            self.input_dim = self.input.shape[1]
            self.output_dim = self.output.shape[1]
        elif self.mode == "pth":
            datalist = os.listdir(self.dataset_path)
            xs_s = None
            xs_t = None
            self.output = None
            datalist.sort()
            for name in datalist:
                if "batch_entropy" in name:
                    if self.output is None:
                        self.output = torch.load(self.dataset_path + name)
                    else:
                        if self.output.shape[0] == self.size:
                            continue
                        self.output = torch.cat((self.output, torch.load(self.dataset_path + name)), 0)

                if "batch_ob_single_couple" in name:
                    if xs_s is None:
                        xs_s = torch.load(self.dataset_path + name).reshape(100, -1)
                    else:
                        if xs_s.shape[0] == self.size:
                            continue
                        xs_s = torch.cat((xs_s, torch.load(self.dataset_path + name).reshape(100, -1)), 0)
                if "batch_ob_two_couple" in name:
                    if xs_t is None:
                        xs_t = torch.load(self.dataset_path + name).reshape(100, -1)
                    else:
                        if xs_t.shape[0] == self.size:
                            continue
                        xs_t = torch.cat((xs_t, torch.load(self.dataset_path + name).reshape(100, -1)), 0)

            self.input = torch.cat((xs_s, xs_t), 1).type(torch.float32)
            self.output = self.output.type(torch.float32)

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]

    def __len__(self):
        return int(self.size)

class DynamicDataset(torch.utils.data.Dataset):
    def __init__(self, hps, data_type):
        if data_type == "train":
            self.dataset_path = hps.data.train_dataset_path
            self.size = int(hps.data.train_size)
        elif data_type == "eval":
            self.dataset_path = hps.data.eval_dataset_path
            self.size = int(hps.data.eval_size)
        elif data_type == "test":
            self.dataset_path = hps.data.test_dataset_path
            self.size = int(hps.data.test_size)
        self.mode = hps.data.mode
        self.n_qubit = int(hps.data.n_qubit)
        if self.mode == "mat":
            sample = h5py.File(self.dataset_path, 'r')
            self.input = np.array(sample['inputdata'])
            self.output = np.array(sample['outdata'])
            self.input = np.einsum('XYZB->BXYZ', self.input)
            self.input = torch.tensor(np.reshape(self.input, [-1, 50, hps.data.n_qubit*3]),dtype=torch.float32)[:self.size]
            self.output = torch.tensor(np.einsum('XYB->BXY', self.output),dtype=torch.float32)[:self.size]

            self.input_dim = self.input.shape[2]
            self.output_dim = self.output.shape[2]
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]

    def __len__(self):
        return int(self.size)

if __name__ == "__main__":
    hps = get_hparams("../" + CONFIG_PATH["config_path"])
    train_dataset = StaticDataset(hps,"train")
    train_loader = DataLoader(train_dataset, num_workers=10, shuffle=True,
                              batch_size=1000, pin_memory=False,
                              drop_last=False)
    for batch_idx, items in enumerate(train_loader):
        print("hello")
        (input, output) = items
        print(input.shape)
