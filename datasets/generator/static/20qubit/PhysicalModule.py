import numpy as np
import torch as tc
import BasicFun as bf
import copy
from scipy.sparse.linalg import eigsh


def spin_operators(spin, is_torch=False, device=None):
    op = dict()
    if spin == 'half':
        op['id'] = np.eye(2, dtype=np.complex128)
        op['sx'] = np.zeros((2, 2), dtype=np.complex128)
        op['sy'] = np.zeros((2, 2), dtype=np.complex128)
        op['sz'] = np.zeros((2, 2), dtype=np.complex128)
        op['su'] = np.zeros((2, 2), dtype=np.complex128)
        op['sd'] = np.zeros((2, 2), dtype=np.complex128)
        op['sx'][0, 1] = 0.5
        op['sx'][1, 0] = 0.5
        op['sy'][0, 1] = 0.5 * 1j
        op['sy'][1, 0] = -0.5 * 1j
        op['sz'][0, 0] = 0.5
        op['sz'][1, 1] = -0.5
        op['su'][0, 1] = 1
        op['sd'][1, 0] = 1
    elif spin == 'one':
        op['id'] = np.eye(3)
        op['sx'] = np.zeros((3, 3))
        op['sy'] = np.zeros((3, 3), dtype=np.complex128)
        op['sz'] = np.zeros((3, 3))
        op['sx'][0, 1] = 1
        op['sx'][1, 0] = 1
        op['sx'][1, 2] = 1
        op['sx'][2, 1] = 1
        op['sy'][0, 1] = -1j
        op['sy'][1, 0] = 1j
        op['sy'][1, 2] = -1j
        op['sy'][2, 1] = 1j
        op['sz'][0, 0] = 1
        op['sz'][2, 2] = -1
        op['sx'] /= 2 ** 0.5
        op['sy'] /= 2 ** 0.5
        op['su'] = np.real(op['sx'] + 1j * op['sy'])
        op['sd'] = np.real(op['sx'] - 1j * op['sy'])
    if is_torch:
        if device is None:
            device = bf.choose_device()
        for k in op:
            if k != 'sy':
                op[k] = tc.from_numpy(op[k]).to(device)
    return op


def fermionic_operators(spin):
    op = dict()
    if spin == 'zero':
        op['id'] = np.eye(2)
        op['cu'] = np.zeros((2, 2))
        op['cd'] = np.zeros((2, 2))
        op['n'] = np.zeros((2, 2))
        op['cu'][0, 1] = 1
        op['cd'][1, 0] = 1
        op['n'][1, 1] = 1
    elif spin == 'one_half':
        op['id'] = np.eye(4, 4)
        op['cr_u'] = np.zeros((4, 4))
        op['cr_d'] = np.zeros((4, 4))
        op['an_u'] = np.zeros((4, 4))
        op['an_d'] = np.zeros((4, 4))
        op['n'] = np.zeros((4, 4))
        op['n_u'] = np.zeros((4, 4))
        op['n_d'] = np.zeros((4, 4))
        op['cr_u'][1, 0] = 1
        op['cr_u'][3, 2] = 1
        op['cr_d'][2, 0] = 1
        op['cr_d'][3, 2] = -1
        op['an_u'][0, 1] = 1
        op['an_u'][2, 3] = 1
        op['an_d'][0, 2] = 1
        op['an_d'][2, 3] = -1
        op['n'][1, 1] = 1
        op['n'][2, 2] = 1
        op['n'][3, 3] = 2
        op['n_u'][3, 3] = 1
        op['n_u'][2, 2] = 1
        op['n_d'][3, 3] = 1
        op['n_d'][1, 1] = 1
    return op


def get_physical_dim(spin, is_type='spin'):
    if is_type == 'spin':
        if spin == 'half':
            return 2
        elif spin == 'one':
            return 3
    elif is_type == 'fermion':
        if spin == 'one_half':
            return 4
        elif spin == 'zero':
            return 2


def hamiltonian_heisenberg(spin, jx, jy, jz, hx1,hy1, hz1,  hx2, hy2, hz2):
    op = spin_operators(spin)
    hamilt = jx*np.kron(op['sx'], op['sx']) + jy*np.kron(op['sy'], op['sy']) + jz*np.kron(
        op['sz'], op['sz'])
    hamilt += (np.kron(op['id'], op['sx'])*hx2 + np.kron(op['sx'], op['id'])*hx1)
    hamilt += (np.kron(op['id'], op['sz'])*hz2 + np.kron(op['sz'], op['id'])*hz1)
    hamilt += (np.kron(op['id'], op['sy'])*hy2 + np.kron(op['sy'], op['id'])*hy1)
    return hamilt

def hamiltonian_heisenberg_random(spin, xx,xy,xz, yx,yy,yz,zx,zy,zz, hx1,hy1, hz1,  hx2, hy2, hz2):
    op = spin_operators(spin)
    hamilt = xx*np.kron(op['sx'], op['sx']) + xy*np.kron(op['sx'], op['sy']) + xz*np.kron(
        op['sx'], op['sz']) + yx*np.kron(op['sy'], op['sx']) + yy*np.kron(op['sy'], op['sy']) + yz*np.kron(
        op['sy'], op['sz'])+ zx*np.kron(op['sz'], op['sx']) + zy*np.kron(op['sz'], op['sy']) + zz*np.kron(
        op['sz'], op['sz'])
    hamilt += (np.kron(op['id'], op['sx'])*hx2 + np.kron(op['sx'], op['id'])*hx1)
    hamilt += (np.kron(op['id'], op['sz'])*hz2 + np.kron(op['sz'], op['id'])*hz1)
    hamilt += (np.kron(op['id'], op['sy'])*hy2 + np.kron(op['sy'], op['id'])*hy1)
    return hamilt



def positions_nearest_neighbor_square(width, height, bound_cond='open'):
    pos = list()
    for i in range(0, width-1):  # interactions inside the first row
        pos.append([i, i+1])
    for n in range(1, height):  # interactions inside the n-th row
        for i in range(0, width-1):
            pos.append([n*width + i, n*width + i + 1])
    for n in range(0, width):
        for i in range(0, height-1):
            pos.append([i*width + n, (i + 1)*width + n])
    if bound_cond.lower() in ['periodic', 'pbc']:
        for n in range(0, height):
            pos.append([n*width, (n + 1)*width - 1])
        for n in range(0, width):
            pos.append([n, (height - 1)*width + n])
    return pos


def position_kagome2n(length, bound_cond):
    pos1 = [[0, 4], [0, 5], [4, 5]]
    pos1 += [[1, 2], [1, 3], [2, 3], [3, 4]]
    pos1 += [[3, 6], [4, 6], [2, 7]]
    if bound_cond.lower() in ['pbc', 'periodic']:
        pos1 += [[2, 5], [5, 7]]
    pos = copy.deepcopy(pos1)
    for n in range(length-1):
        for m in range(len(pos1)):
            pos1[m][0] = pos1[m][0] + 6
            pos1[m][1] = pos1[m][1] + 6
        pos = pos + copy.deepcopy(pos1)
    if bound_cond.lower() in ['pbc', 'periodic']:
        pos[-1][1] = pos[-1][0]
        pos[-1][0] = 1
        pos[-3][1] = pos[-3][0]
        pos[-3][0] = 1
        pos[-4][1] = pos[-4][0]
        pos[-4][0] = 0
        pos[-5][1] = pos[-5][0]
        pos[-5][0] = 0
    else:
        pos = pos[:-3]
    return pos


def ED_ground_state(hamilt, pos, v0=None, k=1, tau=1e-4):
    """
    每个局域哈密顿量的指标顺序满足: (bra0, bra1, ..., ket0, ket1, ...)
    例：求单个三角形上定义的反铁磁海森堡模型基态：
    H2 = hamiltonian_heisenberg('half', 1, 1, 1, 0, 0, 0, 0)
    e0, gs = ED_ground_state([H2.reshape(2, 2, 2, 2)]*3, [[0, 1], [1, 2], [0, 2]])
    print(e0)

    :param hamilt: list，局域哈密顿量
    :param pos: 每个局域哈密顿量作用的自旋
    :param v0: 初态
    :param k: 求解的低能本征态个数
    :param tau: 平移量 H <- I - tau*H
    :return lm: 最大本征值
    :return v1: 最大本征向量
    """
    from scipy.sparse.linalg import LinearOperator as LinearOp

    def convert_nums_to_abc(nums, n0=0):
        s = ''
        n0 = n0 + 97
        for m in nums:
            s += chr(m + n0)
        return s

    def one_map(v, hs, pos_hs, tau_shift, v_dims, ind_v, ind_v_str):
        v = v.reshape(v_dims)
        _v = copy.deepcopy(v)
        for n, pos_now in enumerate(pos_hs):
            ind_contract = list()
            for nn in range(len(pos_now)):
                ind_contract.append(ind_v.index(pos_now[nn]))
            ind_h1 = convert_nums_to_abc(ind_contract)
            ind_h2 = convert_nums_to_abc(list(range(len(pos_now))), n0=len(ind_v))
            ind_f_str = list(copy.deepcopy(ind_v_str))
            for nn, _ind in enumerate(ind_contract):
                ind_f_str[_ind] = ind_h2[nn]
            ind_f_str = ''.join(ind_f_str)
            eq = ind_v_str + ',' + ind_h1 + ind_h2 + '->' + ind_f_str
            _v = _v - tau_shift * np.einsum(eq, v, hs[n])
        return _v.reshape(-1, )

    def one_map_tensordot(v, hs, pos_hs, tau_shift, v_dims, ind_v):
        v = v.reshape(v_dims)
        _v = copy.deepcopy(v)
        for n, pos_now in enumerate(pos_hs):
            ind_contract = list()
            ind_new = copy.deepcopy(ind_v)
            for nn in range(len(pos_now)):
                ind_contract.append(ind_v.index(pos_now[nn]))
                ind_new.remove(pos_now[nn])
            ind_new += pos_now
            ind_permute = list(np.argsort(ind_new))
            _v = _v - tau_shift * np.tensordot(
                v, hs[n], [ind_contract, list(range(len(
                    pos_now)))]).transpose(ind_permute)
        return _v.reshape(-1, )

    # 自动获取总格点数
    n_site = 0
    for x in pos:
        n_site = max([n_site] + list(x))
    n_site += 1
    # 自动获取格点的维数
    d = hamilt[0].shape[0]
    dims = [d] * n_site
    dim_tot = np.prod(dims)
    # 初始化向量
    if v0 is None:
        v0 = eval('np.random.randn' + str(tuple(dims)))
        # v0 = np.random.randn(dims)
    else:
        v0 = v0.reshape(dims)
    v0 /= np.linalg.norm(v0)
    # 初始化指标顺序
    ind = list(range(n_site))
    # ind_str = convert_nums_to_abc(ind)
    # 定义等效线性映射：I - tau*H
    # if len(dims) < 23:
    #     h_effect = LinearOp((dim_tot, dim_tot), lambda vg: one_map(
    #         vg, hamilt, pos, tau, dims, ind, ind_str))
    # else:
    #     pass
    # h_effect = LinearOp((dim_tot, dim_tot), lambda vg: one_map(
    #                     vg, hamilt, pos, tau, dims, ind, ind_str))
    h_effect = LinearOp((dim_tot, dim_tot), lambda vg: one_map_tensordot(
                        vg, hamilt, pos, tau, dims, ind))
    lm, v1 = eigsh(h_effect, k=k, which='LM', v0=v0)
    # 平移本征值
    lm = (1 - lm) / tau
    return lm, v1
