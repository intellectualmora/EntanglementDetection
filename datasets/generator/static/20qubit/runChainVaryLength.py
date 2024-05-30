import ExactDiagonalizationAlgorithm as ED
import BasicFun as bf
import PhysicalModule as pm
import numpy as np
import torch as tc

length = list(range(9, 10))

bf.fprint(bf.now(), 'record.log', append=False)
for n in range(len(length)):
    bf.fprint('length = %g' % (length[n]))
    para = dict()
    para['lattice'] = 'chain'
    para['BC'] = 'OBC'
    para['length'] = length[n]
    para['k'] = 1
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

    bf.fprint('E0 per site = %g' % (ob['e0_per_site']))
    exp = ED.save_exp_heisenberg(para)
    bf.save_pr('./data/HeisenbergChain/', exp, [ob], ['ob'])
    # bf.save('./data/Ising/', exp, [ob], ['ob'])
    print(v.shape)
    # tc.save(v, r'./MPSchainL9_J(0,0,1)_h(0,0)OBC.pth')

