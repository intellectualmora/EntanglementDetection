import numpy as np
import time
import copy
import PhysicalModule as pm
import BasicFun as bf


def parameters_quickED(para=None):
    para_def = dict()
    para_def['lattice'] = 'chain'
    para_def['spin'] = 'half'
    para_def['jx'] = 0
    para_def['jy'] = 0
    para_def['jz'] = 1
    para_def['hx'] = 0
    para_def['hz'] = 0

    para_def['k'] = 1
    para_def['tau'] = 1e-4

    para_def['log_name'] = 'record.log'
    para = para_from_para_def(para_def, para)
    para = paras_lattice(para)
    return para


def paras_lattice(para):
    assert 'lattice' in para
    para_def = dict()
    if para['lattice'] == 'chain':
        para_def['length'] = 10
        para_def['BC'] = 'open'
    elif para['lattice'] == 'square':
        para_def['lattice_size'] = [4, 4]
        para_def['BC'] = 'open'
    para = para_from_para_def(para_def, para)
    if para['lattice'] == 'chain':
        para['couplings2'] = [[n, n+1] for n in range(para['length']-1)]
        if para['BC'].lower() in ['pbc', 'periodic']:
            para['couplings2'].append([0, para['length']-1])
    elif para['lattice'] == 'square':
        para['couplings2'] = pm.positions_nearest_neighbor_square(
            para['lattice_size'][0], para['lattice_size'][1], para['BC'])
        para['length'] = para['lattice_size'][0] * para['lattice_size'][1]
    elif para['lattice'] == 'kagome2n':
        para['couplings2'] = pm.position_kagome2n(para['num_columns'], para['BC'])
        para['length'] = np.max(np.array(para['couplings2'])) + 1
    para['num_couplings2'] = len(para['couplings2'])
    return para


def para_from_para_def(para_def, para):
    for k in para_def:
        if k not in para:
            para[k] = copy.deepcopy(para_def[k])
    return para


def save_exp_heisenberg(para):
    exp = ''
    if para['lattice'] == 'chain':
        exp += 'MPS' + para['lattice'] + 'L%g_J(%g,%g,%g)_h(%g,%g)' % (
            para['length'], para['jx'], para['jy'], para['jz'],
            para['hx'], para['hz'])
        exp += para['BC']
    elif para['lattice'] == 'square':
        exp += 'ED' + para['lattice'] + 'Size(%g,%g)_J(%g,%g,%g)_h(%g,%g)' % (
            para['lattice_size'][0], para['lattice_size'][1], para['jx'], para['jy'],
            para['jz'], para['hx'], para['hz'])
        exp += para['BC']
    elif para['lattice'] == 'kagome2n':
        exp += 'ED' + para['lattice'] + 'NumColumn%g_J(%g,%g,%g)_h(%g,%g)' % (
            para['num_columns'], para['jx'], para['jy'],
            para['jz'], para['hx'], para['hz'])
        exp += para['BC']
    return exp


def quickED_heisenberg(para=None):
    time0 = time.time()
    para = parameters_quickED(para)
    bf.fprint('Parameters: ', para['log_name'])
    bf.print_dict(para, log=para['log_name'])

    h2 = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                   para['hx']/2, para['hz']/2, para['hx']/2, para['hz']/2)
    # h2 = np.real(h2)
    d = pm.get_physical_dim(para['spin'])
    h2 = h2.reshape((d, d, d, d))
    if para['BC'].lower() in ['open', 'obc']:
        hl = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                       para['hx'], para['hz'], para['hx']/2, para['hz']/2)
        hl = hl.reshape(d, d, d, d)
        # hl = np.real(hl)
        hr = pm.hamiltonian_heisenberg(para['spin'], para['jx'], para['jy'], para['jz'],
                                       para['hx']/2, para['hz']/2, para['hx'], para['hz'])
        hr = hr.reshape(d, d, d, d)
        # hr = np.real(hr)
        hamilt = [hl] + [h2] * (para['num_couplings2']-2) + [hr]
    else:
        hamilt = [h2] * para['num_couplings2']
    eg, v = pm.ED_ground_state(hamilt, para['couplings2'], k=para['k'], tau=para['tau'])
    bf.fprint('Task finished in %g seconds' % (time.time()-time0), para['log_name'])
    return eg, v


