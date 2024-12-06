import glob
import json
import logging
import os
import re
import sys
import numpy as np
import torch
from numba import njit,prange
from scipy.linalg import sqrtm
MATPLOTLIB_FLAG = False
CONFIG_PATH = {"model_dir": "./EntanglementDetection/logs/",
               "config_path": "./EntanglementDetection/configs/config.json"}
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

SIGMA_X = np.array([[0,1],[1,0]])
SIGMA_Y = np.array([[0,-1j],[1j,0]])
SIGMA_Z = np.array([[1,0],[0,-1]])
SIGMA_I = np.array([[1,0],[0,1]])
phi_x_plus = np.array([1,1])/np.sqrt(2)
phi_x_neg = np.array([1,-1])/np.sqrt(2)
phi_y_plus = np.array([-1j,1])/np.sqrt(2)
phi_y_neg = np.array([1,-1j])/np.sqrt(2)
phi_z_plus = np.array([1,0])
phi_z_neg = np.array([0,1])


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_hparams(config_path):
    config_save_path = os.path.join(config_path)
    with open(config_save_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    return hparams


def add_model_dir(hps):
    hps.model_dir = os.path.join(hps.train.model_dir, hps.data.mode, str(hps.data.n_qubit))
    if not os.path.exists(hps.model_dir):
        os.makedirs(hps.model_dir)


def progress_bar(x, progress_max):
    s = int(x * 100 / progress_max)
    sys.stdout.write('\r')
    sys.stdout.write("Generate progress: {}%: ".format(s))
    sys.stdout.write("|")
    sys.stdout.write("▋" * s)
    sys.stdout.write(" " * int(100 - s))
    sys.stdout.write("|")
    sys.stdout.flush()


def load_filepaths(dataset_root):
    listdir = os.listdir(dataset_root)
    return listdir


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def plot_wave_to_numpy(wave):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.plot(wave)
    # plt.colorbar(im, ax=ax)
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_compare_waves_to_numpy(wave_1, wave_2, ylim):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(wave_1)
    ax.plot(wave_2, alpha=0.5)
    ax.set_ylim(ylim[0], ylim[1])
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_compare_scale(real, pred):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(real, pred, alpha=0.1, linewidths = 0.1)
    plt.xlabel("gt")
    plt.ylabel("pred")
    plt.xlim(-1, 1)
    #plt.ylim(-1, 1)
   # plt.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None and not skip_optimizer and checkpoint_dict['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # assert "dec" in k or "disc" in k
            # print("load", k)
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (saved_state_dict[k].shape, v.shape)
        except:
            print("error, %s is not in the checkpoint" % k)
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    print("load ")
    logger.info("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x


def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def summarize(writer, global_step, scalars={}, histograms={}, images={}):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats='HWC')

def plot_data_to_numpy(x, y):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    plt.plot(x)
    plt.plot(y)
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def save_checkpoint(model, iteration, checkpoint_path):
    logger.info("Saving model and optimizer state at iteration {} to {}".format(
        iteration, checkpoint_path))
    torch.save(model, checkpoint_path)


def clean_checkpoints(path_to_models='logs/44k/', n_ckpts_to_keep=2, sort_by_time=True):
    """Freeing up space by deleting saved ckpts

  Arguments:
  path_to_models    --  Path to the model directory
  n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
  sort_by_time      --  True -> chronologically delete ckpts
                        False -> lexicographically delete ckpts
  """
    ckpts_files = [f for f in os.listdir(path_to_models) if os.path.isfile(os.path.join(path_to_models, f))]
    name_key = (lambda _f: int(re.compile('._(\d+)\.pth').match(_f).group(1)))
    time_key = (lambda _f: os.path.getmtime(os.path.join(path_to_models, _f)))
    sort_key = time_key if sort_by_time else name_key
    x_sorted = lambda _x: sorted([f for f in ckpts_files if f.startswith(_x) and not f.endswith('_0.pth')],
                                 key=sort_key)
    to_del = [os.path.join(path_to_models, fn) for fn in
              (x_sorted('Net')[:-n_ckpts_to_keep])]
    del_info = lambda fn: logger.info(f".. Free up space by deleting ckpt {fn}")
    del_routine = lambda x: [os.remove(x), del_info(x)]
    rs = [del_routine(fn) for fn in to_del]


def rhol_to_alpha(rhol):
    """
    alpha =
    :param rhol:
    :return:
    """
    alpha = np.zeros((1, int(1/2*(rhol.shape[0]*rhol.shape[0]+rhol.shape[0])))) + 1j*np.zeros((1, int(1/2*(rhol.shape[0]*rhol.shape[0]+rhol.shape[0]))))
    for index in range(rhol.shape[0]):
        alpha[0, int(((index ** 2) + index) / 2):int(((index ** 2) + index) / 2) + index + 1] = rhol[index, :index+1]
    alpha = np.concatenate((np.real(alpha), np.imag(alpha)), axis=1)
    return alpha


def alphas_to_rhols(alphas,dim):
    alphas_real = alphas[:, :int(alphas.shape[1] / 2)]
    alphas_imag = alphas[:, int(alphas.shape[1] / 2):]
    alphas_complex = alphas_real + 1j*alphas_imag
    rhols = np.zeros((alphas.shape[0], dim, dim)) + 1j*np.zeros((alphas.shape[0], dim, dim))
    for batch in range(rhols.shape[0]):
        for index in range(rhols.shape[1]):
            rhols[batch, index, :index + 1] = alphas_complex[batch, int(((index ** 2) + index) / 2):int(((index ** 2) + index) / 2) + index + 1]
    return rhols

# def cal_avg_fid(input,target):
#     sqrt_target = np.zeros_like(target)+1j*np.zeros_like(target)
#     for i in range(target.shape[0]):
#         sqrt_target[i] = sqrtm(target[i])
#     temp = np.einsum("bij,bjk->bik",sqrt_target,input)
#     temp = np.einsum("bij,bjk->bik",temp,sqrt_target)
#     sqrt_temp = np.zeros_like(temp)+1j*np.zeros_like(temp)
#     for i in range(temp.shape[0]):
#         sqrt_temp[i] = sqrtm(temp[i])
#     return np.sum(np.abs(np.einsum("bii->b",sqrt_temp)))/sqrt_temp.shape[0]

def cal_avg_fid(input,target):
    batch_size = input.shape[0]
    numerator = np.einsum("bij,bjk->bik",target,input)
    numerator = np.einsum("bii->b",numerator)
    denominator1 = np.einsum("bij,bkj->bik",target,np.conj(target))
    denominator1 = np.sqrt(np.einsum("bii->b",denominator1))
    denominator2 = np.einsum("bij,bkj->bik", input, np.conj(input))
    denominator2 = np.sqrt(np.einsum("bii->b", denominator2))
    denominator = denominator1*denominator2
    avg_fid = np.sum(numerator/denominator)/batch_size
    avg_fid = avg_fid.real
    return avg_fid
