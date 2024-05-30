import ExactDiagonalizationAlgorithm as ED
import PhysicalModule as pm
# import BasicFun as bf
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
from multiprocessing import Process
matplotlib.use('Agg')


device_cpu = tc.device('cpu')                   # 声明cpu设备
device_cuda = tc.device('cuda')                  # 设备cuda设备
dtype = tc.float64                              # 全文的数据类型


initial_state = tc.zeros(2**20, )
initial_state[0] = tc.tensor(1, )


def get_random(n_num):      # 随机生成 [-1, 1]
    d = np.random.random(n_num)
    return d * 2 - 1


def quantum_renyi2_entropy(quantum_state, n_site):
    n_site = n_site + 1
    quantum_state = quantum_state.reshape(pow(2, n_site), -1)
    p_A = tc.einsum('ab,cb->ac', quantum_state, quantum_state)
    P_A2 = tc.einsum('ab,bc->ac', p_A, p_A)
    ent = renyi_entropy(P_A2)
    return ent


def create_pauli_oprator():
    identiyt2_real = tc.eye(2, dtype=dtype, device=device_cuda)
    identiyt2_imag = tc.zeros((2, 2), dtype=dtype, device=device_cuda)
    identiyt2 = tc.complex(identiyt2_real, identiyt2_imag)
    x_gate_real = tc.tensor([[0, 1], [1, 0]], dtype=dtype, device=device_cuda)
    x_gate_imag = tc.zeros((2, 2), dtype=dtype, device=device_cuda)
    x_gate = tc.complex(x_gate_real, x_gate_imag)
    y_gate_real = tc.tensor([[0, 0], [0, 0]], dtype=dtype, device=device_cuda)
    y_gate_imag = tc.tensor([[0, -1], [1, 0]], dtype=dtype, device=device_cuda)
    y_gate = tc.complex(y_gate_real, y_gate_imag)
    z_gate_real = tc.tensor([[1, 0], [0, -1]], dtype=dtype, device=device_cuda)
    z_gates_imag = tc.zeros((2, 2), dtype=dtype, device=device_cuda)
    z_gate = tc.complex(z_gate_real, z_gates_imag)
    hamiton_gate = list([x_gate, y_gate, z_gate])
    return hamiton_gate


def renyi_entropy(lm):
    # print(tc.trace(lm))
    ent = - tc.log2(tc.trace(lm))
    return ent


def compute_differ_renyi_entropy(g_state):
    renyi_list = tc.zeros([10], device=device_cuda)
    quantum_state = tc.tensor(g_state, device=device_cuda)
    for et in range(10):
        renyi_list[et] = quantum_renyi2_entropy(quantum_state, et)
    return renyi_list


def compute_single_observable(o_state):
    ob_list = tc.zeros([3, 20], dtype=tc.complex128, device=device_cuda)
    ob_state = tc.tensor(o_state, dtype=tc.complex128, device=device_cuda).reshape(-1, )
    vec_state = tc.tensor(o_state, dtype=tc.complex128, device=device_cuda).reshape([2]*20)
    p_gate = create_pauli_oprator()
    for evo_t in range(3):
        evo_0 = tc.einsum('qwertyuiopasdfghjklz,mq->mwertyuiopasdfghjklz',
                          vec_state, p_gate[evo_t]).reshape(-1, )
        evo_1 = tc.einsum('qwertyuiopasdfghjklz,mw->qmertyuiopasdfghjklz',
                          vec_state, p_gate[evo_t]).reshape(-1, )
        evo_2 = tc.einsum('qwertyuiopasdfghjklz,me->qwmrtyuiopasdfghjklz',
                          vec_state, p_gate[evo_t]).reshape(-1, )
        evo_3 = tc.einsum('qwertyuiopasdfghjklz,mr->qwemtyuiopasdfghjklz',
                          vec_state, p_gate[evo_t]).reshape(-1, )
        evo_4 = tc.einsum('qwertyuiopasdfghjklz,mt->qwermyuiopasdfghjklz',
                          vec_state, p_gate[evo_t]).reshape(-1, )
        evo_5 = tc.einsum('qwertyuiopasdfghjklz,my->qwertmuiopasdfghjklz',
                          vec_state, p_gate[evo_t]).reshape(-1, )
        evo_6 = tc.einsum('qwertyuiopasdfghjklz,mu->qwertymiopasdfghjklz',
                          vec_state, p_gate[evo_t]).reshape(-1, )
        evo_7 = tc.einsum('qwertyuiopasdfghjklz,mi->qwertyumopasdfghjklz',
                          vec_state, p_gate[evo_t]).reshape(-1, )
        evo_8 = tc.einsum('qwertyuiopasdfghjklz,mo->qwertyuimpasdfghjklz',
                          vec_state, p_gate[evo_t]).reshape(-1, )
        evo_9 = tc.einsum('qwertyuiopasdfghjklz,mp->qwertyuiomasdfghjklz',
                          vec_state, p_gate[evo_t]).reshape(-1, )
        evo_10 = tc.einsum('qwertyuiopasdfghjklz,ma->qwertyuiopmsdfghjklz',
                           vec_state, p_gate[evo_t]).reshape(-1, )
        evo_11 = tc.einsum('qwertyuiopasdfghjklz,ms->qwertyuiopamdfghjklz',
                           vec_state, p_gate[evo_t]).reshape(-1, )
        evo_12 = tc.einsum('qwertyuiopasdfghjklz,md->qwertyuiopasmfghjklz',
                           vec_state, p_gate[evo_t]).reshape(-1, )
        evo_13 = tc.einsum('qwertyuiopasdfghjklz,mf->qwertyuiopasdmghjklz',
                           vec_state, p_gate[evo_t]).reshape(-1, )
        evo_14 = tc.einsum('qwertyuiopasdfghjklz,mg->qwertyuiopasdfmhjklz',
                           vec_state, p_gate[evo_t]).reshape(-1, )
        evo_15 = tc.einsum('qwertyuiopasdfghjklz,mh->qwertyuiopasdfgmjklz',
                           vec_state, p_gate[evo_t]).reshape(-1, )
        evo_16 = tc.einsum('qwertyuiopasdfghjklz,mj->qwertyuiopasdfghmklz',
                           vec_state, p_gate[evo_t]).reshape(-1, )
        evo_17 = tc.einsum('qwertyuiopasdfghjklz,mk->qwertyuiopasdfghjmlz',
                           vec_state, p_gate[evo_t]).reshape(-1, )
        evo_18 = tc.einsum('qwertyuiopasdfghjklz,ml->qwertyuiopasdfghjkmz',
                           vec_state, p_gate[evo_t]).reshape(-1, )
        evo_19 = tc.einsum('qwertyuiopasdfghjklz,mz->qwertyuiopasdfghjklm',
                           vec_state, p_gate[evo_t]).reshape(-1, )
        ob_list[evo_t, 0] = tc.einsum('a,a->', evo_0, ob_state)
        ob_list[evo_t, 1] = tc.einsum('a,a->', evo_1, ob_state)
        ob_list[evo_t, 2] = tc.einsum('a,a->', evo_2, ob_state)
        ob_list[evo_t, 3] = tc.einsum('a,a->', evo_3, ob_state)
        ob_list[evo_t, 4] = tc.einsum('a,a->', evo_4, ob_state)
        ob_list[evo_t, 5] = tc.einsum('a,a->', evo_5, ob_state)
        ob_list[evo_t, 6] = tc.einsum('a,a->', evo_6, ob_state)
        ob_list[evo_t, 7] = tc.einsum('a,a->', evo_7, ob_state)
        ob_list[evo_t, 8] = tc.einsum('a,a->', evo_8, ob_state)
        ob_list[evo_t, 9] = tc.einsum('a,a->', evo_9, ob_state)
        ob_list[evo_t, 10] = tc.einsum('a,a->', evo_10, ob_state)
        ob_list[evo_t, 11] = tc.einsum('a,a->', evo_11, ob_state)
        ob_list[evo_t, 12] = tc.einsum('a,a->', evo_12, ob_state)
        ob_list[evo_t, 13] = tc.einsum('a,a->', evo_13, ob_state)
        ob_list[evo_t, 14] = tc.einsum('a,a->', evo_14, ob_state)
        ob_list[evo_t, 15] = tc.einsum('a,a->', evo_15, ob_state)
        ob_list[evo_t, 16] = tc.einsum('a,a->', evo_16, ob_state)
        ob_list[evo_t, 17] = tc.einsum('a,a->', evo_17, ob_state)
        ob_list[evo_t, 18] = tc.einsum('a,a->', evo_18, ob_state)
        ob_list[evo_t, 19] = tc.einsum('a,a->', evo_19, ob_state)
    return ob_list


def compute_two_observable(o_state):
    ob_list = tc.zeros([3, 3, 19], dtype=tc.complex128, device=device_cuda)
    ob_state = tc.tensor(o_state, dtype=tc.complex128, device=device_cuda).reshape(-1, )
    vec_state = tc.tensor(o_state, dtype=tc.complex128, device=device_cuda).reshape([2]*20)
    p_gate = create_pauli_oprator()
    for l_g in range(3):
        for r_g in range(3):
            evo_0 = tc.einsum('qwertyuiopasdfghjklz,mq,nw->mnertyuiopasdfghjklz', vec_state, p_gate[l_g],
                              p_gate[r_g]).reshape(-1, )
            evo_1 = tc.einsum('qwertyuiopasdfghjklz,mw,ne->qmnrtyuiopasdfghjklz', vec_state, p_gate[l_g],
                              p_gate[r_g]).reshape(-1, )
            evo_2 = tc.einsum('qwertyuiopasdfghjklz,me,nr->qwmntyuiopasdfghjklz', vec_state, p_gate[l_g],
                              p_gate[r_g]).reshape(-1, )
            evo_3 = tc.einsum('qwertyuiopasdfghjklz,mr,nt->qwemnyuiopasdfghjklz', vec_state, p_gate[l_g],
                              p_gate[r_g]).reshape(-1, )
            evo_4 = tc.einsum('qwertyuiopasdfghjklz,mt,ny->qwermnuiopasdfghjklz', vec_state, p_gate[l_g],
                              p_gate[r_g]).reshape(-1, )
            evo_5 = tc.einsum('qwertyuiopasdfghjklz,my,nu->qwertmniopasdfghjklz', vec_state, p_gate[l_g],
                              p_gate[r_g]).reshape(-1, )
            evo_6 = tc.einsum('qwertyuiopasdfghjklz,mu,ni->qwertymnopasdfghjklz', vec_state, p_gate[l_g],
                              p_gate[r_g]).reshape(-1, )
            evo_7 = tc.einsum('qwertyuiopasdfghjklz,mi,no->qwertyumnpasdfghjklz', vec_state, p_gate[l_g],
                              p_gate[r_g]).reshape(-1, )
            evo_8 = tc.einsum('qwertyuiopasdfghjklz,mo,np->qwertyuimnasdfghjklz', vec_state, p_gate[l_g],
                              p_gate[r_g]).reshape(-1, )
            evo_9 = tc.einsum('qwertyuiopasdfghjklz,mp,na->qwertyuiomnsdfghjklz', vec_state, p_gate[l_g],
                              p_gate[r_g]).reshape(-1, )
            evo_10 = tc.einsum('qwertyuiopasdfghjklz,ma,ns->qwertyuiopmndfghjklz', vec_state, p_gate[l_g],
                               p_gate[r_g]).reshape(-1, )
            evo_11 = tc.einsum('qwertyuiopasdfghjklz,ms,nd->qwertyuiopamnfghjklz', vec_state, p_gate[l_g],
                               p_gate[r_g]).reshape(-1, )
            evo_12 = tc.einsum('qwertyuiopasdfghjklz,md,nf->qwertyuiopasmnghjklz', vec_state, p_gate[l_g],
                               p_gate[r_g]).reshape(-1, )
            evo_13 = tc.einsum('qwertyuiopasdfghjklz,mf,ng->qwertyuiopasdmnhjklz', vec_state, p_gate[l_g],
                               p_gate[r_g]).reshape(-1, )
            evo_14 = tc.einsum('qwertyuiopasdfghjklz,mg,nh->qwertyuiopasdfmnjklz', vec_state, p_gate[l_g],
                               p_gate[r_g]).reshape(-1, )
            evo_15 = tc.einsum('qwertyuiopasdfghjklz,mh,nj->qwertyuiopasdfgmnklz', vec_state, p_gate[l_g],
                               p_gate[r_g]).reshape(-1, )
            evo_16 = tc.einsum('qwertyuiopasdfghjklz,mj,nk->qwertyuiopasdfghmnlz', vec_state, p_gate[l_g],
                               p_gate[r_g]).reshape(-1, )
            evo_17 = tc.einsum('qwertyuiopasdfghjklz,mk,nl->qwertyuiopasdfghjmnz', vec_state, p_gate[l_g],
                               p_gate[r_g]).reshape(-1, )
            evo_18 = tc.einsum('qwertyuiopasdfghjklz,ml,nz->qwertyuiopasdfghjkmn', vec_state, p_gate[l_g],
                               p_gate[r_g]).reshape(-1, )
            ob_list[l_g, r_g, 0] = tc.einsum('a,a->', evo_0, ob_state)
            ob_list[l_g, r_g, 1] = tc.einsum('a,a->', evo_1, ob_state)
            ob_list[l_g, r_g, 2] = tc.einsum('a,a->', evo_2, ob_state)
            ob_list[l_g, r_g, 3] = tc.einsum('a,a->', evo_3, ob_state)
            ob_list[l_g, r_g, 4] = tc.einsum('a,a->', evo_4, ob_state)
            ob_list[l_g, r_g, 5] = tc.einsum('a,a->', evo_5, ob_state)
            ob_list[l_g, r_g, 6] = tc.einsum('a,a->', evo_6, ob_state)
            ob_list[l_g, r_g, 7] = tc.einsum('a,a->', evo_7, ob_state)
            ob_list[l_g, r_g, 8] = tc.einsum('a,a->', evo_8, ob_state)
            ob_list[l_g, r_g, 9] = tc.einsum('a,a->', evo_9, ob_state)
            ob_list[l_g, r_g, 10] = tc.einsum('a,a->', evo_10, ob_state)
            ob_list[l_g, r_g, 11] = tc.einsum('a,a->', evo_11, ob_state)
            ob_list[l_g, r_g, 12] = tc.einsum('a,a->', evo_12, ob_state)
            ob_list[l_g, r_g, 13] = tc.einsum('a,a->', evo_13, ob_state)
            ob_list[l_g, r_g, 14] = tc.einsum('a,a->', evo_14, ob_state)
            ob_list[l_g, r_g, 15] = tc.einsum('a,a->', evo_15, ob_state)
            ob_list[l_g, r_g, 16] = tc.einsum('a,a->', evo_16, ob_state)
            ob_list[l_g, r_g, 17] = tc.einsum('a,a->', evo_17, ob_state)
            ob_list[l_g, r_g, 18] = tc.einsum('a,a->', evo_18, ob_state)
    return ob_list


def get_Ising_ground_density_matrix(number_epoch,batch_entropy,batch_ob_single_couple,batch_ob_two_couple,pid):
    Jz_matrix = get_random(number_epoch)
    hx_matrix = get_random(number_epoch)

    for vt in range(1, number_epoch + 1):
        para = dict()
        para['lattice'] = 'chain_hs'
        para['spin'] = 'half'
        para['bc'] = 'open'
        para['length'] = 20
        para['jx'] = 0
        para['jy'] = 0
        para['jz'] = Jz_matrix[vt - 1]
        para['hx'] = hx_matrix[vt - 1]
        para['hz'] = 0
        para['k'] = 1
        # para = ED.parameters_quickED(para)
        hamilt = [None] * 19

        hamilt[0] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                              para['hx'], 0, para['hx'], 0)
        hamilt[1] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                              0, 0, 0, 0)
        hamilt[2] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                              para['hx'], 0, para['hx'], 0)
        hamilt[3] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                              0, 0, 0, 0)
        hamilt[4] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                              para['hx'], 0, para['hx'], 0)
        hamilt[5] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                              0, 0, 0, 0)
        hamilt[6] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                              para['hx'], 0, para['hx'], 0)
        hamilt[7] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                              0, 0, 0, 0)
        hamilt[8] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                              para['hx'], 0, para['hx'], 0)
        hamilt[9] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                              0, 0, 0, 0)
        hamilt[10] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                               para['hx'], 0, para['hx'], 0)
        hamilt[11] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                               0, 0, 0, 0)
        hamilt[12] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                               para['hx'], 0, para['hx'], 0)
        hamilt[13] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                               0, 0, 0, 0)
        hamilt[14] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                               para['hx'], 0, para['hx'], 0)
        hamilt[15] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                               0, 0, 0, 0)
        hamilt[16] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                               para['hx'], 0, para['hx'], 0)
        hamilt[17] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                               0, 0, 0, 0)
        hamilt[18] = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                               para['hx'], 0, para['hx'], 0)
        pos = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
               [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19]]
        d = pm.get_physical_dim(para['spin'])
        hamilt_ = [h.reshape([d] * 4) for h in hamilt]
        eg, ground_state = pm.ED_ground_state(hamilt_, pos, k=para['k'], v0=initial_state)

        r_entropy = compute_differ_renyi_entropy(ground_state)
        batch_entropy[vt-1, :] = r_entropy

        ob_single = compute_single_observable(ground_state)
        batch_ob_single_couple[vt-1, :, :] = ob_single

        ob_two = compute_two_observable(ground_state)
        batch_ob_two_couple[vt-1, :, :, :] = ob_two

        print('epoch = ' + str(vt), 'hx = ' + str(para['hx']), 'Jz = ' + str(para['jz']))
        print('----------------------------------------------------')

        if vt == (number_epoch):
            tc.save(Jz_matrix, r'./Jz_matrix'+pid+'.pth')
            tc.save(hx_matrix, r'./hx_matrix'+pid+'.pth')
            tc.save(batch_entropy, r'./batch_entropy'+pid+'.pth')
            tc.save(batch_ob_single_couple, r'./batch_ob_single_couple'+pid+'.pth')
            tc.save(batch_ob_two_couple, r'./batch_ob_two_couple'+pid+'.pth')


def main(argv):
    print(f'子进程：{str(argv[0])}开始...')
    Hamiton_gate = create_pauli_oprator()
    batch_num = int(argv[1])
    batch_entropy = tc.zeros([batch_num, 10], device=device_cuda, dtype=tc.complex128)
    batch_ob_single_couple = tc.zeros([batch_num, 3, 20], device=device_cuda, dtype=tc.complex128)
    batch_ob_two_couple = tc.zeros([batch_num, 3, 3, 19], device=device_cuda)
    get_Ising_ground_density_matrix(batch_num,batch_entropy,batch_ob_single_couple,batch_ob_two_couple,str(argv[0]))

if __name__ == "__main__":
#   Ps = []
#   for i in range(1):
#       p = Process(target=main, args=('_'+str(i), ))
#       p.start()
#       Ps.append(p)
#   for i in range(1):
#       Ps[i].join()
    main(sys.argv[1:])
