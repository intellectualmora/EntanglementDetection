import ExactDiagonalizationAlgorithm as ED
import PhysicalModule as pm
# import BasicFun as bf
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


device_cpu = tc.device('cpu')                   # 声明cpu设备
device_cuda = tc.device('cuda')                  # 设备cuda设备
dtype = tc.float64                              # 全文的数据类型


initial_state = tc.zeros(2**4, )
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
    renyi_list = tc.zeros([2], device=device_cuda)
    quantum_state = tc.tensor(g_state, device=device_cuda)
    for et in range(2):
        renyi_list[et] = quantum_renyi2_entropy(quantum_state, et)
    return renyi_list


def compute_single_observable(o_state):
    ob_list = tc.zeros([3, 4], dtype=tc.complex128, device=device_cuda)
    ob_state = tc.tensor(o_state, dtype=tc.complex128, device=device_cuda).reshape(-1, )
    vec_state = tc.tensor(o_state, dtype=tc.complex128, device=device_cuda).reshape([2]*4)
    p_gate = create_pauli_oprator()
    for evo_t in range(3):
        evo_0 = tc.einsum('qwer,mq->mwer',
                          vec_state, p_gate[evo_t]).reshape(-1, )
        evo_1 = tc.einsum('qwer,mw->qmer',
                          vec_state, p_gate[evo_t]).reshape(-1, )
        evo_2 = tc.einsum('qwer,me->qwmr',
                          vec_state, p_gate[evo_t]).reshape(-1, )
        evo_3 = tc.einsum('qwer,mr->qwem',
                          vec_state, p_gate[evo_t]).reshape(-1, )
        ob_list[evo_t, 0] = tc.einsum('a,a->', evo_0, ob_state)
        ob_list[evo_t, 1] = tc.einsum('a,a->', evo_1, ob_state)
        ob_list[evo_t, 2] = tc.einsum('a,a->', evo_2, ob_state)
        ob_list[evo_t, 3] = tc.einsum('a,a->', evo_3, ob_state)
    return ob_list


def compute_two_observable(o_state):
    ob_list = tc.zeros([3, 3, 3], dtype=tc.complex128, device=device_cuda)
    ob_state = tc.tensor(o_state, dtype=tc.complex128, device=device_cuda).reshape(-1, )
    vec_state = tc.tensor(o_state, dtype=tc.complex128, device=device_cuda).reshape([2]*4)
    p_gate = create_pauli_oprator()
    for l_g in range(3):
        for r_g in range(3):
            evo_0 = tc.einsum('qwer,mq,nw->mner', vec_state, p_gate[l_g],
                              p_gate[r_g]).reshape(-1, )
            evo_1 = tc.einsum('qwer,mw,ne->qmnr', vec_state, p_gate[l_g],
                              p_gate[r_g]).reshape(-1, )
            evo_2 = tc.einsum('qwer,me,nr->qwmn', vec_state, p_gate[l_g],
                              p_gate[r_g]).reshape(-1, )
            ob_list[l_g, r_g, 0] = tc.einsum('a,a->', evo_0, ob_state)
            ob_list[l_g, r_g, 1] = tc.einsum('a,a->', evo_1, ob_state)
            ob_list[l_g, r_g, 2] = tc.einsum('a,a->', evo_2, ob_state)
    return ob_list


def get_Ising_ground_density_matrix(number_epoch):
    J_matrix = get_random(number_epoch*3)
    h_matrix = get_random(number_epoch*4)
    for vt in range(1, number_epoch + 1):
        para = dict()
        para['lattice'] = 'chain'
        para['spin'] = 'half'
        para['bc'] = 'open'
        para['length'] = 4
        para['j1'] =  J_matrix[3*(vt-1)]
        para['j2'] =  J_matrix[3*(vt-1)+1]
        para['j3'] =  J_matrix[3*(vt-1)+2]

        para['hz1'] = h_matrix[4*(vt-1)]
        para['hz2'] = h_matrix[4*(vt-1)+1]
        para['hz3'] = h_matrix[4*(vt-1)+2]
        para['hz4'] = h_matrix[4*(vt-1)+3]


        para['hy'] = 0
        para['k'] = 1
        hamilt = [None] * 3
        hamilt[0] = pm.hamiltonian_heisenberg(para['spin'], para['j1'],0, para['j1'],
                                              0,0 ,para['hz1'],0, 0,   para['hz2'])
        hamilt[1] = pm.hamiltonian_heisenberg(para['spin'], para['j2'],0, para['j2'],
                                              0, 0,0,0, 0, 0)
        hamilt[2] = pm.hamiltonian_heisenberg(para['spin'], para['j3'], 0, para['j3'],
                                              0,0, para['hz3'],0, 0, para['hz4'])
        pos = [[0, 1], [1, 2], [2, 3]]
        d = pm.get_physical_dim(para['spin'])
        hamilt_ = [h.reshape([d] * 4) for h in hamilt]
        eg, ground_state = pm.ED_ground_state(hamilt_, pos, k=para['k'], v0=initial_state)
        r_entropy = compute_differ_renyi_entropy(ground_state)
        batch_entropy[vt-1, :] = r_entropy

        ob_single = compute_single_observable(ground_state)
        batch_ob_single_couple[vt-1, :, :] = ob_single

        ob_two = compute_two_observable(ground_state)
        batch_ob_two_couple[vt-1, :, :, :] = ob_two

        print('epoch = ' + str(vt), 'h = ' + str(para['hz1'])+" "+str(para['hz2'])+" "+str(para['hz3'])+" "+str(para['hz4']), 'J = ' + str(para['j1'])+" "+str(para['j2'])+" "+str(para['j3']))

        print('----------------------------------------------------')

        if vt == (number_epoch):
            tc.save(J_matrix, r'./4bit_Jz_matrix.pth')
            print("J_matrix")
            print(J_matrix)
            print("h_matrix")
            print(h_matrix)
            print("batch_entropy")
            print(batch_entropy)
            print("batch_ob_single_couple")
            print(batch_ob_single_couple)
            print("batch_ob_two_couple")
            print(batch_ob_two_couple)

            tc.save(h_matrix, r'./hy_matrix'+pid+'.pth')
            tc.save(batch_entropy, r'./batch_entropy.pth')
            tc.save(batch_ob_single_couple, r'./batch_ob_single_couple.pth')
            tc.save(batch_ob_two_couple, r'./batch_ob_two_couple.pth')


Hamiton_gate = create_pauli_oprator()
batch_num = 1
batch_entropy = tc.zeros([batch_num, 2], device=device_cuda, dtype=tc.complex128)
batch_ob_single_couple = tc.zeros([batch_num, 3, 4], device=device_cuda, dtype=tc.complex128)
batch_ob_two_couple = tc.zeros([batch_num, 3, 3, 3], device=device_cuda)


get_Ising_ground_density_matrix(batch_num)