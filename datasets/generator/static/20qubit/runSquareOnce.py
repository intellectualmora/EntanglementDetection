import ExactDiagonalizationAlgorithm as ED
import BasicFun as bf
import PhysicalModule as pm
import numpy as np


para = dict()
para['lattice'] = 'square'
para['BC'] = 'OBC'
para['lattice_size'] = [3, 3]
para['jx'] = 1
para['jy'] = 1
para['jz'] = 1
para['hx'] = 0
para['hz'] = 0
para['k'] = 1


for boundC in ['PBC', 'OBC']:
    para['BC'] = boundC
    para = ED.parameters_quickED(para)

    eg, v = ED.quickED_heisenberg(para)
    d = pm.get_physical_dim(para['spin'])
    dim1 = d**round(para['length']/2)
    dim2 = round(v.size / dim1)
    lm = np.linalg.svd(v.reshape(dim1, dim2), compute_uv=False)
    ob = dict()
    ob['e0_per_site'] = eg / para['length']
    ob['ent_mid'] = bf.entanglement_entropy(lm)
    ob['lm50'] = lm[:50]

    bf.fprint('E0 per site = %.15g' % (ob['e0_per_site']))
    exp = ED.save_exp_heisenberg(para)
    bf.save_pr('./data/HeisenbergSquare/', exp, [ob], ['ob'])
