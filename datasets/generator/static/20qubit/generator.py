import ExactDiagonalizationAlgorithm as ED
import PhysicalModule as pm
# import BasicFun as bf
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import re


device_cpu = tc.device('cpu')                   # 声明cpu设备
device_cuda = tc.device('cuda')                  # 设备cuda设备
dtype = tc.float64                              # 全文的数据类型
n_qubit = 6
h_shape = "H(X)+J(X+Y+Z)"
initial_state = tc.zeros(2**n_qubit, )
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
    renyi_list = tc.zeros([int(n_qubit/2)], device=device_cuda)
    quantum_state = tc.tensor(g_state, device=device_cuda)
    for et in range(int(n_qubit/2)):
        renyi_list[et] = quantum_renyi2_entropy(quantum_state, et)
    return renyi_list


def compute_single_observable(o_state):
    ob_list = tc.zeros([3, n_qubit], dtype=tc.complex128, device=device_cuda)
    ob_state = tc.tensor(o_state, dtype=tc.complex128, device=device_cuda).reshape(-1, )
    vec_state = tc.tensor(o_state, dtype=tc.complex128, device=device_cuda).reshape([2]*n_qubit)
    p_gate = create_pauli_oprator()
    for evo_t in range(3):
        for loc in range(n_qubit):
            hamInstr ="".join(str(i) for i in [chr(ord('a') + i) for i in range(n_qubit)])
            obInstr = "".join(str(i) for i in [chr(ord('a') + n_qubit),chr(ord('a') + loc)])
            finInstr = "".join(str(i) for i in [chr(ord('a') + i) if i != loc else chr(ord('a') + n_qubit) for i in range(n_qubit)])
            instr = hamInstr +','+obInstr+'->'+finInstr
            evo = tc.einsum(instr,vec_state, p_gate[evo_t]).reshape(-1, )
            ob_list[evo_t, loc] = tc.einsum('a,a->', evo, ob_state)
    return ob_list


def compute_two_observable(o_state):
    ob_list = tc.zeros([3, 3, n_qubit-1], dtype=tc.complex128, device=device_cuda)
    ob_state = tc.tensor(o_state, dtype=tc.complex128, device=device_cuda).reshape(-1, )
    vec_state = tc.tensor(o_state, dtype=tc.complex128, device=device_cuda).reshape([2]*n_qubit)
    p_gate = create_pauli_oprator()
    for l_g in range(3):
        for r_g in range(3):
            for loc in range(n_qubit-1):
                hamInstr = "".join(str(i) for i in [chr(ord('a') + i) for i in range(n_qubit)])
                obInstr1 = "".join(str(i) for i in [chr(ord('a') + n_qubit), chr(ord('a') + loc)])
                obInstr2 = "".join(str(i) for i in [chr(ord('a') + n_qubit + 1), chr(ord('a') + loc + 1)])
                finInstr = [chr(ord('a') + i) if i != loc else chr(ord('a') + n_qubit) for i in range(n_qubit)]
                finInstr[loc+1] = chr(ord('a') + n_qubit + 1)
                finInstr = "".join(str(i) for i in finInstr)
                instr = hamInstr + ',' + obInstr1+','+obInstr2+ '->' + finInstr
                evo_0 = tc.einsum(instr, vec_state, p_gate[l_g],
                                  p_gate[r_g]).reshape(-1, )
                ob_list[l_g, r_g, loc] = tc.einsum('a,a->', evo_0, ob_state)
    return ob_list



def decoder():
    h_pattern = r"H\((.*?)\)"
    match = re.search(h_pattern, h_shape)
    h_group_ = match.group(1).split("+")
    j_pattern = r"J\((.*?)\)"
    match = re.search(j_pattern, h_shape)
    j_group_ = match.group(1).split("+")
    j_mapping = {"X":1,"Y":2,"Z":3}
    h1_mapping = {"X":4,"Y":5,"Z":6}
    h2_mapping = {"X":7,"Y":8,"Z":9}

    j_group = list(map(lambda x: j_mapping[x],j_group_))
    h_group = list(map(lambda x: h1_mapping[x],h_group_))
    h_group += list(map(lambda x: h2_mapping[x],h_group_))
    return h_group,j_group


def get_Ising_ground_density_matrix(number_epoch):
    for vt in range(1, number_epoch + 1):
        para = dict()
        para['lattice'] = 'chain'
        para['spin'] = 'half'
        para['bc'] = 'open'
        para['length'] = n_qubit
        para['k'] = 1
        hamilt = [None] * (n_qubit-1)

        for loc in range(n_qubit-1):
            config = [para['spin'],get_random(9)]
            if loc%2 != 0:
                config[4:] = [0,0,0,0,0,0]
            if get_random()
            hamilt[loc] = pm.hamiltonian_heisenberg(*config)
        pos = [[i,i+1] for i in range(n_qubit-1)]

        d = pm.get_physical_dim(para['spin'])
        hamilt_ = [h.reshape([d] * 4) for h in hamilt]
        eg, ground_state = pm.ED_ground_state(hamilt_, pos, k=para['k'], v0=initial_state)
        r_entropy = compute_differ_renyi_entropy(ground_state)
        batch_entropy[vt-1, :] = r_entropy

        ob_single = compute_single_observable(ground_state)
        batch_ob_single_couple[vt-1, :, :] = ob_single

        ob_two = compute_two_observable(ground_state)
        batch_ob_two_couple[vt-1, :, :, :] = ob_two

        print('epoch = ' + str(vt), 'h = ' + str(h_matrix), 'J = ' + str(J_matrix))

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

            tc.save(h_matrix, r'./hy_matrix.pth')
            tc.save(batch_entropy, r'./batch_entropy.pth')
            tc.save(batch_ob_single_couple, r'./batch_ob_single_couple.pth')
            tc.save(batch_ob_two_couple, r'./batch_ob_two_couple.pth')


Hamiton_gate = create_pauli_oprator()
batch_num = 1
batch_entropy = tc.zeros([batch_num, int(n_qubit/2)], device=device_cuda, dtype=tc.complex128)
batch_ob_single_couple = tc.zeros([batch_num, 3, n_qubit], device=device_cuda, dtype=tc.complex128)
batch_ob_two_couple = tc.zeros([batch_num, 3, 3, n_qubit-1], device=device_cuda)


get_Ising_ground_density_matrix(batch_num)