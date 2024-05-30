import numpy as np
from tenpy.networks.site import Site
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import time
import os
import json
import pandas as pd
from tenpy.linalg import np_conserved as npc

class SpinSite(Site):
    r"""General Spin S site.

    There are `2S+1` local states range from ``down`` (0)  to ``up`` (2S+1),
    corresponding to ``Sz=-S, -S+1, ..., S-1, S``.
    Local operators are the spin-S operators,
    e.g. ``Sz = [[0.5, 0.], [0., -0.5]]``,
    ``Sx = 0.5*sigma_x`` for the Pauli matrix `sigma_x`.

    ==============  ================================================
    operator        description
    ==============  ================================================
    ``Id, JW``      Identity :math:`\mathbb{1}`
    ``Sx, Sy, Sz``  Spin components :math:`S^{x,y,z}`,
                    equal to half the Pauli matrices.
    ``Sp, Sm``      Spin flips :math:`S^{\pm} = S^{x} \pm i S^{y}`
    ==============  ================================================

    ============== ====  ============================
    `conserve`     qmod  *excluded* onsite operators
    ============== ====  ============================
    ``'Sz'``       [1]   ``Sx, Sy, Sigmax, Sigmay``
    ``'parity'``   [2]   --
    ``'None'``     []    --
    ============== ====  ============================

    Parameters
    ----------
    conserve : str
        Defines what is conserved, see table above.
    sort_charge : bool
        Whether :meth:`sort_charge` should be called at the end of initialization.
        This is usually a good idea to reduce potential overhead when using charge conservation.
        Note that this permutes the order of the local basis states for ``conserve='parity'``!
        For backwards compatibility with existing data, it is not (yet) enabled by default.

    Attributes
    ----------
    S : {0.5, 1, 1.5, 2, ...}
        The 2S+1 states range from m = -S, -S+1, ... +S.
    conserve : str
        Defines what is conserved, see table above.
    """
    def __init__(self, S=0.5, conserve='Sz', sort_charge=None):
        if not conserve:
            conserve = 'None'
        if conserve not in ['Sz', 'parity', 'None']:
            raise ValueError("invalid `conserve`: " + repr(conserve))
        self.S = S = float(S)
        d = 2 * S + 1
        if d <= 1:
            raise ValueError("negative S?")
        if np.rint(d) != d:
            raise ValueError("S is not half-integer or integer")
        d = int(d)
        Sz_diag = -S + np.arange(d)
        Sz = -np.diag(Sz_diag)*2
        Sp = np.zeros([d, d])
        for n in np.arange(d - 1):
            # Sp |m> =sqrt( S(S+1)-m(m+1)) |m+1>
            m = n - S
            Sp[n + 1, n] = np.sqrt(S * (S + 1) - m * (m + 1))
        Sm = np.transpose(Sp)
        # Sp = Sx + i Sy, Sm = Sx - i Sy
        Sx = (Sp + Sm) * 1
        Sy = -(Sm - Sp) * 1j
        # Note: For S=1/2, Sy might look wrong compared to the Pauli matrix or SpinHalfSite.
        # Don't worry, I'm 99.99% sure it's correct (J. Hauschild)
        # The reason it looks wrong is simply that this class orders the states as ['down', 'up'],
        # while the usual spin-1/2 convention is ['up', 'down'], as you can also see if you look
        # at the Sz entries...
        # (The commutation relations are checked explicitly in `tests/test_site.py`)
        ops = dict(Sp=Sp, Sm=Sm, Sz=Sz)
        if conserve == 'Sz':
            chinfo = npc.ChargeInfo([1], ['2*Sz'])
            leg = npc.LegCharge.from_qflat(chinfo, np.array(2 * Sz_diag, dtype=np.int64))
        else:
            ops.update(Sx=Sx, Sy=Sy)
            if conserve == 'parity':
                chinfo = npc.ChargeInfo([2], ['parity_Sz'])
                leg = npc.LegCharge.from_qflat(chinfo, np.mod(np.arange(d), 2))
            else:
                leg = npc.LegCharge.from_trivial(d)
        self.conserve = conserve
        names = [str(i) for i in np.arange(-S, S + 1, 1.)]
        Site.__init__(self, leg, names, sort_charge=sort_charge, **ops)
        self.state_labels['down'] = self.state_labels[names[0]]
        self.state_labels['up'] = self.state_labels[names[-1]]

    def __repr__(self):
        """Debug representation of self."""
        return "SpinSite(S={S!s}, {c!r})".format(S=self.S, c=self.conserve)

class SpinChainNNN3(CouplingMPOModel):
    r"""Spin-S sites coupled by next-nearest neighbour interactions.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i,j \rangle, i < j}
                \mathtt{Jx} S^x_i S^x_j + \mathtt{Jy} S^y_i S^y_j + \mathtt{Jz} S^z_i S^z_j \\
            + \sum_{\langle \langle i,j \rangle \rangle, i< j}
                \mathtt{Jxp} S^x_i S^x_j + \mathtt{Jyp} S^y_i S^y_j + \mathtt{Jzp} S^z_i S^z_j \\
            + \sum_i
              \mathtt{hx} S^x_i + \mathtt{hy} S^y_i + \mathtt{hz} S^z_i

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbors and
    :math:`\langle \langle i,j \rangle \rangle, i < j` denotes next nearest neighbors.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`SpinChainNNN2` below.

    Options
    -------
    .. cfg:config :: SpinChainNNN2
        :include: CouplingMPOModel

        S : {0.5, 1, 1.5, 2, ...}
            The 2S+1 local states range from m = -S, -S+1, ... +S.
        conserve : 'best' | 'Sz' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinSite`.
            For ``'best'``, we check the parameters what can be preserved.
        sort_charge : bool | None
            Whether to sort by charges of physical legs.
            See change comment in :class:`~tenpy.networks.site.Site`.
        Jx, Jy, Jz, Jxp, Jyp, Jzp, hx, hy, hz : float | array
            Coupling as defined for the Hamiltonian above.
    """
    def init_sites(self, model_params):
        S = model_params.get('S', 0.5)
        conserve = model_params.get('conserve', 'best')
        if conserve == 'best':
            # check how much we can conserve
            if not model_params.any_nonzero([('Jx', 'Jy'),
                                             ('Jxp', 'Jyp'), 'hx', 'hy'], "check Sz conservation"):
                conserve = 'Sz'
            elif not model_params.any_nonzero(['hx', 'hy'], "check parity conservation"):
                conserve = 'parity'
            else:
                conserve = None
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        sort_charge = model_params.get('sort_charge', None)
        site = SpinSite(S, conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):
        # 0) read out/set default parameters
        L = model_params.get('L',0)
        Jxx = model_params.get('Jxx', [0.] * (L-1))
        Jyy = model_params.get('Jyy', [0.] * (L-1))
        Jzz = model_params.get('Jzz', [0.] * (L-1))
        Jxy = model_params.get('Jxy', [0.] * (L-1))
        Jxz = model_params.get('Jxz', [0.] * (L-1))
        Jyx = model_params.get('Jyx', [0.] * (L-1))
        Jyz = model_params.get('Jyz', [0.] * (L-1))
        Jzx = model_params.get('Jzx', [0.] * (L-1))
        Jzy = model_params.get('Jzy', [0.] * (L-1))

        hx = model_params.get('hx', [0.] * L)
        hy = model_params.get('hy', [0.] * L)
        hz = model_params.get('hz', [0.] * L)

        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(hx, u, 'Sx')
            self.add_onsite(hy, u, 'Sy')
            self.add_onsite(hz, u, 'Sz')

        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(Jxx, u1, 'Sx', u2, 'Sx', dx)
            self.add_coupling(Jyy, u1, 'Sy', u2, 'Sy', dx)
            self.add_coupling(Jzz, u1, 'Sz', u2, 'Sz', dx)

            self.add_coupling(Jxy, u1, 'Sx', u2, 'Sy', dx)
            self.add_coupling(Jxz, u1, 'Sx', u2, 'Sz', dx)
            self.add_coupling(Jyx, u1, 'Sy', u2, 'Sx', dx)
            self.add_coupling(Jyz, u1, 'Sy', u2, 'Sz', dx)
            self.add_coupling(Jzx, u1, 'Sz', u2, 'Sx', dx)
            self.add_coupling(Jzy, u1, 'Sz', u2, 'Sy', dx)

def DMRG_heisenberg_spin(L,Jxx,Jyy,Jzz,Jxy,Jxz,Jyx,Jyz,Jzx,Jzy,hx,hy,hz):
    model_params = dict(
        L=L,
        S=0.5,  # spin 1/2
        Jxx=Jxx,
        Jyy=Jyy,
        Jzz=Jzz,

        Jxy=Jxy,
        Jxz=Jxz,
        Jyx=Jyx,
        Jyz=Jyz,
        Jzx=Jzx,
        Jzy=Jzy,

        hx=hx,
        hy=hy,
        hz=hz,
        bc_MPS='finite',
        )
    M = SpinChainNNN3(model_params)

    product_state = ["up"] * M.lat.N_sites  # initial Neel state
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    dmrg_params = {
        'mixer': True,  # setting this to True is essential for the 1-site algorithm to work.
        'trunc_params': {
            'chi_max': 30,
            'svd_min': 1.e-25
        },
        'combine': False,
        'active_sites': 1  # specifies single-site
    }
    info = dmrg.run(psi, M, dmrg_params)
    E = info['E']
   # print("E = {E:.13f}".format(E=E))
   # print("final bond dimensions: ", psi.chi)
    return E, psi, M

def expectation_value(mps,L):
    expectation_value_list = [None]*(12*L-9)
    flag = 0
    for num in range(L):
        expectation_value_list[flag] = mps.expectation_value_term([('Sx', num)])
        expectation_value_list[flag+1] = mps.expectation_value_term([('Sy', num)])
        expectation_value_list[flag+2] = mps.expectation_value_term([('Sz', num)])
        flag+=3
    for num in range(L-1):
        expectation_value_list[flag] = mps.expectation_value_term([('Sx', num),('Sx', num+1)])
        expectation_value_list[flag+1] = mps.expectation_value_term([('Sy', num),('Sx', num+1)])
        expectation_value_list[flag+2] = mps.expectation_value_term([('Sz', num),('Sx', num+1)])
        expectation_value_list[flag+3] = mps.expectation_value_term([('Sx', num),('Sy', num+1)])
        expectation_value_list[flag+4] = mps.expectation_value_term([('Sy', num),('Sy', num+1)])
        expectation_value_list[flag+5] = mps.expectation_value_term([('Sz', num),('Sy', num+1)])
        expectation_value_list[flag+6] = mps.expectation_value_term([('Sx', num),('Sz', num+1)])
        expectation_value_list[flag+7] = mps.expectation_value_term([('Sy', num),('Sz', num+1)])
        expectation_value_list[flag+8] = mps.expectation_value_term([('Sz', num),('Sz', num+1)])
        flag+=9
    return np.array(expectation_value_list).tolist()

def global_expectation_value(mps,global_measurment1,global_measurment2):
        global_expectation_value_list = [None]*2
        global_expectation_value_list[0] = mps.expectation_value_term(global_measurment1)
        global_expectation_value_list[1] = mps.expectation_value_term(global_measurment2)
        return np.array(global_expectation_value_list).tolist()

def generate_reference_parameter(L):
    Jxx=2*np.random.rand(L-1)-1
    Jyy=2*np.random.rand(L-1)-1
    Jzz=2*np.random.rand(L-1)-1

    Jxy=2*np.random.rand(L-1)-1
    Jxz=2*np.random.rand(L-1)-1
    Jyx=2*np.random.rand(L-1)-1
    Jyz=2*np.random.rand(L-1)-1
    Jzx=2*np.random.rand(L-1)-1
    Jzy=2*np.random.rand(L-1)-1
    #
    hx=2*np.random.rand(L)-1
    hy=2*np.random.rand(L)-1
    hz=2*np.random.rand(L)-1
    return {"Jxx":Jxx.tolist(),"Jyy":Jyy.tolist(),"Jzz":Jzz.tolist(),"Jxy":Jxy.tolist(),"Jxz":Jxz.tolist(),"Jyx":Jyx.tolist(),"Jyz":Jyz.tolist(),"Jzx":Jzx.tolist(),"Jzy":Jzy.tolist(),"hx":hx.tolist(),"hy":hy.tolist(),"hz":hz.tolist()}

def generate_randn_data(batch_num,L,J,var, entropy_20_60, entropy_30_50, entropy_40_40, entropy_50_30, entropy_60_20,entanglement_spectrums,fidelitys,expect_value_psis,flag):
    _, psi_inital, _ = DMRG_heisenberg_spin(L, Jxx=J["Jxx"], Jyy=J["Jyy"], Jzz=J["Jzz"], Jxy=J["Jxy"], Jxz=J["Jxz"], Jyz=J["Jyz"], Jyx=J["Jyx"], Jzx=J["Jzx"],
                                            Jzy=J["Jzy"], hx=J["hx"], hy=J["hy"], hz=J["hz"])
    for num in range(batch_num):
        Jxx_norm = J["Jxx"] + var*np.random.randn(L-1)
        Jyy_norm = J["Jyy"] + var*np.random.randn(L-1)
        Jzz_norm = J["Jzz"] + var*np.random.randn(L-1)
        Jxy_norm = J["Jxy"] + var*np.random.randn(L-1)
        Jxz_norm = J["Jxz"] + var*np.random.randn(L-1)
        Jyx_norm = J["Jyx"] + var*np.random.randn(L-1)
        Jyz_norm = J["Jyz"] + var*np.random.randn(L-1)
        Jzx_norm = J["Jzx"] + var*np.random.randn(L-1)
        Jzy_norm = J["Jzy"] + var*np.random.randn(L-1)
        #
        hx_norm = J["hx"] + var*np.random.randn(L)
        hy_norm = J["hy"] + var*np.random.randn(L)
        hz_norm = J["hz"] + var*np.random.randn(L)
        _, psi, _ = DMRG_heisenberg_spin(L,Jxx=Jxx_norm,Jyy=Jyy_norm,Jzz=Jzz_norm,Jxy=Jxy_norm,Jxz=Jxz_norm,Jyz=Jyz_norm,Jyx=Jyx_norm,Jzx=Jzx_norm,Jzy=Jzy_norm,hx=hx_norm,hy=hy_norm,hz=hz_norm)

        expect_value_psis[flag[0]] = expectation_value(psi, L)
        entanglement_spectrums[flag[0]] = (psi.entanglement_spectrum()[int(L/2-1)]).tolist()[:2]
        entropy_20_60[flag[0]] = psi.entanglement_entropy_segment2([i for i in range(int(L/4))], n=2)
        entropy_30_50[flag[0]] = psi.entanglement_entropy_segment2([i for i in range(int(3*L/8))], n=2)
        entropy_40_40[flag[0]] = psi.entanglement_entropy_segment2([i for i in range(int(L/2))], n=2)
        entropy_50_30[flag[0]] = psi.entanglement_entropy_segment2([i for i in range(int(5*L/8))], n=2)
        entropy_60_20[flag[0]] = psi.entanglement_entropy_segment2([i for i in range(int(3*L/4))], n=2)

        fidelitys[flag[0]] = np.abs(psi.overlap(psi_inital))**2
        flag[0] = flag[0] + 1

def generate_rand_data(batch_num,L,entropys,expect_value_psis,flag):
    for num in range(batch_num):
        Jxx_norm = 2*np.random.rand(L-1)-1
        Jyy_norm = 2*np.random.rand(L-1)-1
        Jzz_norm = 2*np.random.rand(L-1)-1
        Jxy_norm = [0.] * (L-1)
        Jxz_norm = [0.] * (L-1)
        Jyx_norm = [0.] * (L-1)
        Jyz_norm = [0.] * (L-1)
        Jzx_norm = [0.] * (L-1)
        Jzy_norm = [0.] * (L-1)
        #
        hx_norm = [2*np.random.rand()-1]*L
        hy_norm = [2*np.random.rand()-1]*L
        hz_norm = [2*np.random.rand()-1]*L
        _, psi, _ = DMRG_heisenberg_spin(L,Jxx=Jxx_norm,Jyy=Jyy_norm,Jzz=Jzz_norm,Jxy=Jxy_norm,Jxz=Jxz_norm,Jyz=Jyz_norm,Jyx=Jyx_norm,Jzx=Jzx_norm,Jzy=Jzy_norm,hx=hx_norm,hy=hy_norm,hz=hz_norm)
        expect_value_psis[flag[0]] = (expectation_value(psi, L))
        entropys[flag[0]] = [(psi.entanglement_entropy_segment2([i for i in range(k)], n=2)) for k in range(1,L)]
        flag[0] = flag[0] + 1

if __name__ == "__main__":
    for L in range(4,11):
        # J = generate_reference_parameter(L)
        # pauli = ["Sx", "Sy", "Sz"]
        #
        # js = json.dumps({"J":J})
        # file = open('80bit_reference_data.json','w')
        # file.write(js)
        # file.close()
        print("开始....",os.getpid())
        T1 = time.time()
        batch_num = 1000
        np.random.seed(os.getpid())
        entropys = [None]*batch_num
        expect_value_psis = [None]*batch_num
        flag = [0,]
        generate_rand_data(batch_num,L, entropys=entropys,expect_value_psis=expect_value_psis,flag=flag)
        js = json.dumps({"entropys":entropys,"expect_value_psis":expect_value_psis})
        file = open('/data/phy-chely/4to10bit2024521/' + str(L) + 'bit_data_' + str(os.getpid()) + '.json', 'w')
        # file = open('./' + str(L) + 'bit_data_' + str(os.getpid()) + '.json', 'w')
        file.write(js)
        file.close()
        T2 = time.time()
        print('程序运行时间:%s秒' % ((T2 - T1)))


