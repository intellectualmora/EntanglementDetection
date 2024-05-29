from modules import utils
import logging
import time
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from modules.data_loader import StaticDataset
from modules.utils import CONFIG_PATH
from modules.models import BPNet
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    hps = utils.get_hparams(CONFIG_PATH["config_path"])
    utils.add_model_dir(hps)
    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = hps.train.port
    run(0,n_gpus,hps) #multi GPUs: mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps))

def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    # for pytorch on win, backend use gloo
    dist.init_process_group(backend= 'gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    train_dataset = StaticDataset(hps,"train")
    num_workers = 5 if mp.cpu_count() > 4 else mp.cpu_count()
    train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=True, pin_memory=True,
                              batch_size=hps.train.batch_size)
    if rank == 0:
        eval_dataset = StaticDataset(hps,"eval")
        eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
                                 batch_size=hps.train.eval_batch_size, pin_memory=True,
                                 )

    net = BPNet(train_dataset.input_dim,hps.model.static.H,hps.model.static.H2,train_dataset.output_dim).cuda(rank)
    optim = torch.optim.Adam(
        net.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    net = DDP(net, device_ids=[rank], find_unused_parameters=True)
    epoch_str = 1
    global_step = 0
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scaler = GradScaler(enabled=hps.train.fp16_run)
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, net, optim, scaler,
                               [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, net, optim, scaler,
                               [train_loader, None], None, None)
        scheduler.step()

def train_and_evaluate(rank, epoch, hps, net, optim, scaler, loaders, logger, writers):
    train_loader, eval_loader = loaders
    LossFunc = nn.MSELoss()
    if writers is not None:
        writer, writer_eval = writers

    global global_step

    net.train()
    for batch_idx, items in enumerate(train_loader):
        inputs, outputs = items
        inputs = inputs.cuda(rank, non_blocking=True)
        outputs = outputs.cuda(rank, non_blocking=True)
        with autocast(enabled=hps.train.fp16_run):
            infers = net(inputs)
            with autocast(enabled=False):
                loss = LossFunc(infers,outputs)
        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        grad_norm = utils.clip_grad_value_(net.parameters(), None)
        scaler.step(optim)
        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim.param_groups[0]['lr']
                losses = [loss]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info(f"Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}")

                scalar_dict = {"loss/train": loss, "learning_rate": lr,
                               "grad_norm": grad_norm}
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict
                )
            if global_step % hps.train.eval_interval == 0:
                evaluate(net, eval_loader, writer_eval,hps,rank)
                utils.save_checkpoint(net, epoch, os.path.join(hps.model_dir,  "Net_"+str(hps.data.n_qubit)+"_"+str(global_step)+".pth"))
                keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
        global_step += 1
    if rank == 0:
        global start_time
        now = time.time()
        durtaion = format(now - start_time, '.2f')
        logger.info(f'====> Epoch: {epoch}, cost {durtaion} s')
        start_time = now

def evaluate(net, eval_loader, writer_eval,hps,rank):
    net.eval()
    scalars_dict = {}
    LossFunc = nn.MSELoss()
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            inputs, outputs = items
            inputs = inputs.cuda(rank, non_blocking=True)
            outputs = outputs.cuda(rank, non_blocking=True)
            infers = net(inputs)
            loss = LossFunc(infers, outputs)
            scalars_dict = {
                f"loss/eval": loss
            }
    utils.summarize(
        writer=writer_eval,
        scalars=scalars_dict,
        global_step=global_step,
    )
    net.train()


if __name__ == "__main__":
    main()
