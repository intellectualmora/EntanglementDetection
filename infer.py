import os

from modules import utils
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
import torch
import torch.multiprocessing as mp
from modules.utils import CONFIG_PATH
from modules.data_loader import StaticDataset,DynamicDataset
from torch.utils.data import DataLoader
from modules.models import BPNet, RNN_50_100
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def infer(model_dir):
    assert torch.cuda.is_available(), "CPU training is not allowed."
    num_workers = 5 if mp.cpu_count() > 4 else mp.cpu_count()
    hps = utils.get_hparams(CONFIG_PATH["config_path"])
    test_dataset = StaticDataset(hps, "test") if hps.data.type == "static" else DynamicDataset(hps,"test")
    test_loader = DataLoader(test_dataset, num_workers=num_workers, shuffle=False, pin_memory=True,
                             batch_size=hps.infer.batch_size)
    model = torch.load(model_dir)  # 读取模型
    model = DDP(model, device_ids=[0], find_unused_parameters=True)
    model.eval()
    inputs, outputs, infers = None, None, None
    for batch_idx, items in enumerate(test_loader):
        input_data, output_data = items
        input_data = input_data.cuda(0, non_blocking=True)
        output_data = output_data.cuda(0, non_blocking=True)
        if batch_idx == 0:
            infers = model(input_data)
            inputs = input_data
            outputs = output_data
        else:
            infers = torch.cat([infers, model(input_data)], dim=0)
            inputs = torch.cat([inputs, input_data], dim=0)
            outputs = torch.cat([outputs, output_data], dim=0)
    return inputs, infers, outputs

if __name__ == "__main__":
    hps = utils.get_hparams(CONFIG_PATH["config_path"])
    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = hps.train.port
    dist.init_process_group(backend= 'gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus, rank=0)
    inputs, infers, outputs = infer(hps.infer.model_dir)
    plt.scatter(outputs.cpu().detach().numpy(),infers.cpu().detach().numpy(),s=0.1,alpha=0.5)
    plt.show()