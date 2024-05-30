import ExactDiagonalizationAlgorithm as ED
import PhysicalModule as pm
# import BasicFun as bf
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import re
import sys

device_cpu = tc.device('cpu')                   # 声明cpu设备
device_cuda = tc.device('cuda')                  # 设备cuda设备
dtype = tc.float64                              # 全文的数据类型
n_qubit = 4
initial_state = np.zeros(2**n_qubit, dtype=np.complex128)
initial_state[0] = 1


def get_random(n_num):      # 随机生成 [-1, 1]
    d = np.random.random(n_num)
    return d * 2 - 1


def quantum_renyi2_entropy(quantum_state, n_site):
    n_site = n_site + 1
    quantum_state = quantum_state.reshape(pow(2, n_site), -1)
    p_A = tc.einsum('ab,cb->ac', quantum_state, tc.conj(quantum_state))
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
    ob_list = tc.zeros([3, n_qubit], dtype=tc.float64, device=device_cuda)
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
            ob_list[evo_t, loc] = tc.einsum('a,a->', evo, tc.conj(ob_state)).real
    return ob_list


def compute_two_observable(o_state):
    ob_list = tc.zeros([3, 3, n_qubit-1], dtype=tc.float64, device=device_cuda)
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
                evo = tc.einsum(instr, vec_state, p_gate[l_g],
                                  p_gate[r_g]).reshape(-1, )
                ob_list[l_g, r_g, loc] = tc.einsum('a,a->', evo, tc.conj(ob_state)).real
    return ob_list


def get_Ising_ground_density_matrix(number_epoch,pid,batch_entropy,batch_ob_single_couple,batch_ob_two_couple):
    J_matrix = get_random(number_epoch*(n_qubit-1)*3)
    for i in range(number_epoch*(n_qubit-1)*3):
        if np.random.rand() < 0.9:
            J_matrix[i] = 0
    h_matrix = get_random(number_epoch*n_qubit)
    for i in range(number_epoch*n_qubit):
        if np.random.rand() < 0.9:
            h_matrix[i] = 0
    j_group = [1,2,3]
    for vt in range(1, number_epoch + 1):
        para = dict()
        para['spin'] = 'half'
        para['k'] = 1
        hamilt = [None] * (n_qubit-1)
        for loc in range(n_qubit-1):
            para['h1'] = h_matrix[n_qubit * (vt - 1)+2*int(loc/2)]
            para['h2'] = h_matrix[n_qubit * (vt - 1)+2*int(loc/2)+1]
            config = [para['spin']]
            for poi in range(1,10):
                if poi in j_group:
                    config.append(J_matrix[(n_qubit-1)*(vt-1)*3+3*loc+poi-1])
                elif poi == 6 and loc % 2 == 0:
                    config.append(para['h1'])
                elif poi == 9 and loc % 2 == 0:
                    config.append(para['h2'])
                else:
                    config.append(0)
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

        print('epoch = ' + str(vt), pid)

        if vt == (number_epoch):
            tc.save(h_matrix, r'/data/phy-chely/h_matrix'+pid+'_'+str(n_qubit)+'.pth')
            tc.save(batch_entropy, r'/data/phy-chely/batch_entropy'+pid+'_'+str(n_qubit)+'.pth')
            tc.save(batch_ob_single_couple, r'/data/phy-chely/batch_ob_single_couple'+pid+'_'+str(n_qubit)+'.pth')
            tc.save(batch_ob_two_couple, r'/data/phy-chely/batch_ob_two_couple'+pid+'_'+str(n_qubit)+'.pth')


def main(argv):
    print(f'子进程：{str(argv[0])}开始...')
    batch_num = int(argv[1])
    batch_entropy = tc.zeros([batch_num, int(n_qubit/2)], device=device_cuda, dtype=tc.float64)
    batch_ob_single_couple = tc.zeros([batch_num, 3, n_qubit], device=device_cuda, dtype=tc.float64)
    batch_ob_two_couple = tc.zeros([batch_num, 3, 3, n_qubit-1], device=device_cuda,dtype=tc.float64)
    batches = [batch_entropy,batch_ob_single_couple,batch_ob_two_couple]
    get_Ising_ground_density_matrix(batch_num,str(argv[0]),*batches)



if __name__ == "__main__":
    main(sys.argv[1:])
