"""Transport routines."""

from collections import OrderedDict

import numpy as np

import ceptr.constants as cc
import ceptr.thermo as cth
import ceptr.writer as cw


def transport(fstream, mechanism, species_info):
    """Write the transport functions."""
    cw.writer(fstream, cw.comment("Transport function declarations "))
    n_species = species_info.n_species
    species_transport = analyze_transport(mechanism, species_info)
    n_lite = 0
    idx_light_specs = []
    for sp in range(n_species):
        spec = species_info.nonqssa_species[sp]
        if spec.weight < 5.0:
            n_lite += 1
            idx_light_specs.append(spec.idx)
    misc_trans_info(fstream, kk=n_species, n_lite=n_lite)
    wt(fstream, species_info)
    eps(fstream, mechanism, species_info, species_transport)
    sig(fstream, mechanism, species_info, species_transport)
    dip(fstream, mechanism, species_info, species_transport)
    pol(fstream, mechanism, species_info, species_transport)
    zrot(fstream, mechanism, species_info, species_transport)
    nlin(fstream, mechanism, species_info, species_transport)

    viscosity(fstream, mechanism, species_info, species_transport, ntfit=50)
    diffcoefs(fstream, species_info, species_transport, ntfit=50)
    light_specs(fstream, idx_light_specs)
    thermaldiffratios(
        fstream,
        species_info,
        species_transport,
        idx_light_specs,
        ntfit=50,
    )


def analyze_transport(mechanism, species_info):
    """Extract transport model coefficients."""
    transdata = OrderedDict()

    for spec in species_info.nonqssa_species:
        species = mechanism.species(spec.name)

        m1 = species.transport
        if m1.geometry == "atom":
            lin = 0
        elif m1.geometry == "linear":
            lin = 1
        elif m1.geometry == "nonlinear":
            lin = 2
        else:
            raise ValueError("Unrecognized species geometry in transport")
        eps = (m1.well_depth * cc.ureg.joule).to(cc.ureg.erg).m / cc.kb
        sig = (m1.diameter * cc.ureg.meter).to(cc.ureg.angstrom).m
        dip = (m1.dipole * cc.ureg.coulomb * cc.ureg.m).to(cc.ureg.debye).m
        pol = (m1.polarizability * cc.ureg.meter**3).to(cc.ureg.angstrom**3).m
        zrot = m1.rotational_relaxation

        transdata[spec] = [lin, eps, sig, dip, pol, zrot]

    return transdata


def misc_trans_info(fstream, kk, n_lite, no=4):
    """Write transport information."""
    cw.writer(fstream)
    lenimc = 4 * kk + n_lite
    generate_trans_routine_integer(
        fstream,
        [
            "egtransetLENIMC",
            "EGTRANSETLENIMC",
            "egtransetlenimc",
            "egtransetlenimc_",
            "LENIMC",
        ],
        lenimc,
    )

    cw.writer(fstream)
    cw.writer(fstream)
    lenrmc = (19 + 2 * no + no * n_lite) * kk + (15 + no) * kk**2
    generate_trans_routine_integer(
        fstream,
        [
            "egtransetLENRMC",
            "EGTRANSETLENRMC",
            "egtransetlenrmc",
            "egtransetlenrmc_",
            "LENRMC",
        ],
        lenrmc,
    )

    cw.writer(fstream)
    cw.writer(fstream)
    generate_trans_routine_integer(
        fstream,
        [
            "egtransetNO",
            "EGTRANSETNO",
            "egtransetno",
            "egtransetno_",
            "NO",
        ],
        no,
    )

    cw.writer(fstream)
    cw.writer(fstream)
    generate_trans_routine_integer(
        fstream,
        [
            "egtransetKK",
            "EGTRANSETKK",
            "egtransetkk",
            "egtransetkk_",
            "KK",
        ],
        kk,
    )

    cw.writer(fstream)
    cw.writer(fstream)
    generate_trans_routine_integer(
        fstream,
        [
            "egtransetNLITE",
            "EGTRANSETNLITE",
            "egtransetnlite",
            "egtransetnlite_",
            "NLITE",
        ],
        n_lite,
    )

    cw.writer(fstream)
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Patm in ergs/cm3"))

    cw.writer(fstream, "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE")
    cw.writer(fstream, "void egtransetPATM(amrex::Real* PATM) {")
    cw.writer(fstream, "*PATM =   0.1013250000000000E+07;}")


def generate_trans_routine_integer(fstream, nametab, expression):
    """Write generic integer transport routine."""
    cw.writer(fstream, "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE")
    cw.writer(fstream, f"void {nametab[0]:s}(int* {nametab[4]:s} ) {{")
    cw.writer(fstream, f"*{nametab[4]:s} = {expression:d};}}")


def generate_trans_routine_simple(
    fstream, mechanism, species_info, nametab, idx, species_transport
):
    """Write generic transport routine."""
    cw.writer(fstream, "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE")
    cw.writer(fstream, f"void {nametab[0]:s}(amrex::Real* {nametab[4]:s} ) {{")

    for spec in species_info.nonqssa_species:
        cw.writer(
            fstream,
            f"{nametab[4]}[{spec.idx}] = {float(species_transport[spec][idx]):.8E};",
        )
    cw.writer(fstream, "}")


def wt(fstream, species_info):
    """Write molecular weights function."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("the molecular weights in g/mol"))

    cw.writer(fstream, "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE")
    cw.writer(fstream, "void egtransetWT(amrex::Real* WT ) {")

    n_species = species_info.n_species
    for sp in range(n_species):
        species = species_info.nonqssa_species[sp]
        cw.writer(
            fstream,
            f"{'WT'}[{species.idx}] = {float(species.weight):.8E};",
        )

    cw.writer(fstream, "}")


def eps(fstream, mechanism, species_info, species_transport):
    """Write the lennard-jones potential well depth function."""
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("the lennard-jones potential well depth eps/kb in K"),
    )
    generate_trans_routine_simple(
        fstream,
        mechanism,
        species_info,
        [
            "egtransetEPS",
            "EGTRANSETEPS",
            "egtranseteps",
            "egtranseteps_",
            "EPS",
        ],
        1,
        species_transport,
    )


def sig(fstream, mechanism, species_info, species_transport):
    """Write the the lennard-jones collision diameter function."""
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("the lennard-jones collision diameter in Angstroms"),
    )
    generate_trans_routine_simple(
        fstream,
        mechanism,
        species_info,
        [
            "egtransetSIG",
            "EGTRANSETSIG",
            "egtransetsig",
            "egtransetsig_",
            "SIG",
        ],
        2,
        species_transport,
    )


def dip(fstream, mechanism, species_info, species_transport):
    """Write the dipole moment function."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("the dipole moment in Debye"))
    generate_trans_routine_simple(
        fstream,
        mechanism,
        species_info,
        [
            "egtransetDIP",
            "EGTRANSETDIP",
            "egtransetdip",
            "egtransetdip_",
            "DIP",
        ],
        3,
        species_transport,
    )


def pol(fstream, mechanism, species_info, species_transport):
    """Write the polarizability function."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("the polarizability in cubic Angstroms"))
    generate_trans_routine_simple(
        fstream,
        mechanism,
        species_info,
        [
            "egtransetPOL",
            "EGTRANSETPOL",
            "egtransetpol",
            "egtransetpol_",
            "POL",
        ],
        4,
        species_transport,
    )


def zrot(fstream, mechanism, species_info, species_transport):
    """Write the rotational relaxation collision number."""
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("the rotational relaxation collision number at 298 K"),
    )
    generate_trans_routine_simple(
        fstream,
        mechanism,
        species_info,
        [
            "egtransetZROT",
            "EGTRANSETZROT",
            "egtransetzrot",
            "egtransetzrot_",
            "ZROT",
        ],
        5,
        species_transport,
    )


def nlin(fstream, mechanism, species_info, species_transport):
    """Write the (monoatomic, linear, nonlinear) information."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("0: monoatomic, 1: linear, 2: nonlinear"))

    cw.writer(fstream, "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE")
    cw.writer(fstream, "void egtransetNLIN(int* NLIN) {")

    for species in species_info.nonqssa_species:
        cw.writer(
            fstream,
            f"{'NLIN'}[{species.idx}] = {int(species_transport[species][0])};",
        )

    cw.writer(fstream, "}")


def viscosity(fstream, mechanism, species_info, species_transport, ntfit):
    """Write the viscosity function."""
    n_species = species_info.n_species
    # compute single constants in g/cm/s
    na = 6.02214199e23
    ru = 8.31447e7
    # conversion coefs
    a2cm = 1.0e-8
    debye2cgs = 1.0e-18
    # temperature increment
    dt = (species_info.high_temp - species_info.low_temp) / (ntfit - 1)
    # factor dependent upon the molecule
    m_crot = np.zeros(n_species)
    m_cvib = np.zeros(n_species)
    isatm = np.zeros(n_species)
    for spec in species_transport:
        if int(species_transport[spec][0]) == 0:
            m_crot[spec.idx] = 0.0
            m_cvib[spec.idx] = 0.0
            isatm[spec.idx] = 0.0
        elif int(species_transport[spec][0]) == 1:
            m_crot[spec.idx] = 1.0
            m_cvib[spec.idx] = 5.0 / 2.0
            isatm[spec.idx] = 1.0
        else:
            m_crot[spec.idx] = 1.5
            m_cvib[spec.idx] = 3.0
            isatm[spec.idx] = 1.0
    # viscosities coefs (4 per spec)
    cofeta = OrderedDict()
    # conductivities coefs (4 per spec)
    coflam = OrderedDict()
    for spec in species_transport:
        spvisc = []
        spcond = []
        tlog = []
        for n in range(ntfit):
            t = species_info.low_temp + dt * n
            # variables
            # eq. (2)
            tr = t / float(species_transport[spec][1])
            conversion = debye2cgs * debye2cgs / a2cm / a2cm / a2cm / cc.kb
            dst = (
                0.5
                * conversion
                * float(species_transport[spec][3]) ** 2
                / (
                    float(species_transport[spec][1])
                    * float(species_transport[spec][2]) ** 3
                )
            )
            # viscosity of spec at t
            # eq. (1)
            conversion = a2cm * a2cm
            visc = (
                (5.0 / 16.0)
                * np.sqrt(np.pi * spec.weight * cc.kb * t / na)
                / (
                    om22_chemkin(tr, dst)
                    * np.pi
                    * float(species_transport[spec][2])
                    * float(species_transport[spec][2])
                    * conversion
                )
            )
            # conductivity of spec at t
            # eq. (30)
            conversion = a2cm * a2cm
            m_red = spec.weight / (2.0 * na)
            diffcoef = (
                (3.0 / 16.0)
                * np.sqrt(2.0 * np.pi * cc.kb**3 * t**3 / m_red)
                / (
                    10.0
                    * np.pi
                    * om11_chemkin(tr, dst)
                    * float(species_transport[spec][2])
                    * float(species_transport[spec][2])
                    * conversion
                )
            )
            # eq. (19)
            cv_vib_r = (
                cth.eval_cv_species(mechanism, spec, t) - m_cvib[spec.idx]
            ) * isatm[spec.idx]
            rho_atm = 10.0 * spec.weight / (ru * t)
            f_vib = rho_atm * diffcoef / visc
            # eq. (20)
            a = 2.5 - f_vib
            # eqs. (21) + (32-33)
            cv_rot_r = m_crot[spec.idx]
            # note: the T corr is not applied in CANTERA
            b = float(species_transport[spec][5]) * f_corr(
                298.0, float(species_transport[spec][1])
            ) / f_corr(t, float(species_transport[spec][1])) + (2.0 / np.pi) * (
                (5.0 / 3.0) * cv_rot_r + f_vib
            )
            # eq. (18)
            f_rot = f_vib * (1.0 + 2.0 / np.pi * a / b)
            # eq. (17)
            cv_trans_r = 3.0 / 2.0
            f_trans = 5.0 / 2.0 * (1.0 - 2.0 / np.pi * a / b * cv_rot_r / cv_trans_r)
            if int(species_transport[spec][0]) == 0:
                cond = ((visc * ru / spec.weight)) * (5.0 / 2.0) * cv_trans_r
            else:
                cond = ((visc * ru / spec.weight)) * (
                    f_trans * cv_trans_r + f_rot * cv_rot_r + f_vib * cv_vib_r
                )

            # log transformation for polyfit
            tlog.append(np.log(t))
            spvisc.append(np.log(visc))
            spcond.append(np.log(cond))

        cofeta[spec.idx] = np.polyfit(tlog, spvisc, 3)
        coflam[spec.idx] = np.polyfit(tlog, spcond, 3)

    # header for visco
    cw.writer(fstream)
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Poly fits for the viscosities, dim NO*KK"))

    # visco coefs
    cw.writer(fstream, "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE")
    cw.writer(fstream, "void egtransetCOFETA(amrex::Real* COFETA) {")

    for spec in species_info.nonqssa_species:
        for i in range(4):
            cw.writer(
                fstream,
                f"{'COFETA'}[{spec.idx * 4 + i}] = {cofeta[spec.idx][3 - i]:.8E};",
            )

    cw.writer(fstream, "}")

    # header for cond
    cw.writer(fstream)
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Poly fits for the conductivities, dim NO*KK"))

    # visco coefs
    cw.writer(fstream, "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE")
    cw.writer(fstream, "void egtransetCOFLAM(amrex::Real* COFLAM) {")

    for spec in species_info.nonqssa_species:
        for i in range(4):
            cw.writer(
                fstream,
                f"{'COFLAM'}[{spec.idx * 4 + i}] = {coflam[spec.idx][3 - i]:.8E};",
            )

    cw.writer(fstream, "}")


def diffcoefs(fstream, species_info, species_transport, ntfit):
    """Write the diffusion coefficients."""
    # REORDERING OF SPECS
    spec_ordered = []
    n_species = species_info.n_species
    for i in range(n_species):
        for spec in species_transport:
            if spec.idx == i:
                spec_ordered.append(spec)
                break

    # Ern & Giovangigli, Multicomponent Transport Algorithms, (1994),
    # eqs. (6.1.3) - (6.1.8)
    # compute single constants in g/cm/s
    na = 6.02214199e23
    # conversion coefs
    a2cm = 1.0e-8
    debye2cgs = 1.0e-18
    patm = 0.1013250000000000e07
    # temperature increment
    dt = (species_info.high_temp - species_info.low_temp) / (ntfit - 1)
    # diff coefs (4 per spec pair)
    cofd = []
    for i, spec1 in enumerate(spec_ordered):
        cofd.append([])
        assert i == spec1.idx
        for j, spec2 in enumerate(spec_ordered[0 : i + 1]):
            assert j == spec2.idx
            # eq. (9)
            sigm = (
                0.5
                * (
                    float(species_transport[spec1][2])
                    + float(species_transport[spec2][2])
                )
                * a2cm
            ) * xi(spec1, spec2, species_transport) ** (-1.0 / 6.0)
            # eq. (4)
            m_red = spec1.weight * spec2.weight / (spec1.weight + spec2.weight) / na
            # eq. (8) & (14)
            epsm_k = (
                np.sqrt(
                    float(species_transport[spec1][1])
                    * float(species_transport[spec2][1])
                )
                * xi(spec1, spec2, species_transport) ** 2.0
            )

            # eq. (15)
            conversion = debye2cgs * debye2cgs / cc.kb
            dst = (
                0.5
                * conversion
                * float(species_transport[spec1][3])
                * float(species_transport[spec2][3])
                / (epsm_k * sigm**3)
            )
            if not xi_bool(spec1, spec2, species_transport):
                dst = 0.0
            # enter the loop on temperature
            spdiffcoef = []
            tlog = []
            for n in range(ntfit):
                t = species_info.low_temp + dt * n
                tr = t / epsm_k
                # eq. (3)
                # note: these are "corrected" in chemkin not in CANTERA... we chose not to
                difcoeff = (
                    3.0
                    / 16.0
                    * 1
                    / patm
                    * (
                        np.sqrt(2.0 * np.pi * t**3 * cc.kb**3 / m_red)
                        / (np.pi * sigm * sigm * om11_chemkin(tr, dst))
                    )
                )
                # log transformation for polyfit
                tlog.append(np.log(t))
                spdiffcoef.append(np.log(difcoeff))

            cofd[i].append(np.polyfit(tlog, spdiffcoef, 3))

    # use the symmetry for upper triangular terms
    # note: starting with this would be preferable (only one bigger loop)
    # note2: or write stuff differently !
    # for i,spec1 in enumerate(spec_ordered):
    #    for j,spec2 in enumerate(spec_ordered[i+1:]):
    #        cofd[i].append(cofd[spec2.id][spec1.id])

    # header for diffusion coefs
    cw.writer(fstream)
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("Poly fits for the diffusion coefficients, dim NO*KK*KK"),
    )

    # coefs
    cw.writer(fstream, "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE")
    cw.writer(fstream, "void egtransetCOFD(amrex::Real* COFD) {")

    # map the upper matrix to the one dimensional array
    for i, _ in enumerate(spec_ordered):
        for j, _ in enumerate(spec_ordered[0 : i + 1]):
            for k in range(4):
                cw.writer(
                    fstream,
                    f"{'COFD'}[{int(4 * (i*(i + 1)/2 + j) + k)}]"
                    f" = {cofd[i][j][3 - k]:.8E};",
                )

    cw.writer(fstream, "}")


def light_specs(fstream, speclist):
    """Write list of specs with small weight, dim n_lite."""
    # header
    cw.writer(fstream)
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("List of specs with small weight, dim NLITE"))

    # coefs
    cw.writer(fstream, "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE")
    if len(speclist) > 0:
        cw.writer(fstream, "void egtransetKTDIF(int* KTDIF) {")
    else:
        cw.writer(fstream, "void egtransetKTDIF(int* /*KTDIF*/) {")

    for i in range(len(speclist)):
        cw.writer(fstream, f"{'KTDIF'}[{i}] = {speclist[i]};")

    cw.writer(fstream, "}")


def thermaldiffratios(
    fstream,
    species_info,
    species_transport,
    light_spec_list,
    ntfit,
):
    """Write thermal diffusion ratios."""
    n_species = species_info.n_species
    # This is an overhaul of chemkin version III
    # REORDERING OF SPECS
    spec_ordered = []
    for i in range(n_species):
        for spec in species_transport:
            if spec.idx == i:
                spec_ordered.append(spec)
                break

    # compute single constants in g/cm/s
    # conversion coefs
    debye2cgs = 1.0e-18
    a2cm = 1.0e-8
    # temperature increment
    dt = (species_info.high_temp - species_info.low_temp) / (ntfit - 1)
    # diff ratios (4 per spec pair involving light species)
    coftd = []
    k = -1
    for i, spec1 in enumerate(spec_ordered):
        assert i == spec1.idx
        if spec1.idx in light_spec_list:
            k = k + 1
            assert light_spec_list[k] == spec1.idx
            coftd.append([])
            epsi = float(species_transport[spec1][1]) * cc.kb
            sigi = float(species_transport[spec1][2]) * a2cm
            poli = float(species_transport[spec1][4]) * a2cm * a2cm * a2cm
            # eq. (12)
            poli_red = poli / sigi**3
            for j, spec2 in enumerate(spec_ordered):
                assert j == spec2.idx
                # eq. (53)
                wji = (spec2.weight - spec1.weight) / (spec1.weight + spec2.weight)
                epsj = float(species_transport[spec2][1]) * cc.kb
                sigj = float(species_transport[spec2][2]) * a2cm
                dipj = float(species_transport[spec2][3]) * debye2cgs
                # eq. (13)
                dipj_red = dipj / np.sqrt(epsj * sigj**3)
                eps_ratio = epsj / epsi
                tse = 1.0 + 0.25 * poli_red * dipj_red**2 * np.sqrt(eps_ratio)
                eok = tse**2 * np.sqrt(
                    float(species_transport[spec1][1])
                    * float(species_transport[spec2][1])
                )
                # enter the loop on temperature
                spthdiffcoef = []
                t_tab = []
                for n in range(ntfit):
                    t = species_info.low_temp + dt * n
                    tslog = np.log(t) - np.log(eok)
                    # eq. (53)
                    thdifcoeff = (
                        15.0
                        / 2.0
                        * wji
                        * (2.0 * astar(tslog) + 5.0)
                        * (6.0 * cstar(tslog) - 5.0)
                        / (
                            astar(tslog)
                            * (16.0 * astar(tslog) - 12.0 * bstar(tslog) + 55.0)
                        )
                    )

                    # log transformation for polyfit
                    t_tab.append(t)
                    spthdiffcoef.append(thdifcoeff)

                coftd[k].append(np.polyfit(t_tab, spthdiffcoef, 3))

    # header for thermal diff ratios
    cw.writer(fstream)
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("Poly fits for thermal diff ratios, dim NO*NLITE*KK"),
    )

    # visco coefs
    cw.writer(fstream, "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE")
    if len(coftd) > 0:
        cw.writer(fstream, "void egtransetCOFTD(amrex::Real* COFTD) {")
    else:
        cw.writer(fstream, "void egtransetCOFTD(amrex::Real* /*COFTD*/) {")

    for i in range(len(coftd)):
        for j in range(n_species):
            for k in range(4):
                cw.writer(
                    fstream,
                    f"{'COFTD'}[{i * 4 * n_species + j * 4 + k}] ="
                    f" {coftd[i][j][3 - k]:.8E};",
                )

    cw.writer(fstream, "}")


def astar(tslog):
    """Compute astar."""
    a_tab = [
        0.1106910525e01,
        -0.7065517161e-02,
        -0.1671975393e-01,
        0.1188708609e-01,
        0.7569367323e-03,
        -0.1313998345e-02,
        0.1720853282e-03,
    ]

    a = a_tab[6]
    for i in range(6):
        a = a_tab[5 - i] + a * tslog

    return a


def bstar(tslog):
    """Compute bstar."""
    b_tab = [
        0.1199673577e01,
        -0.1140928763e00,
        -0.2147636665e-02,
        0.2512965407e-01,
        -0.3030372973e-02,
        -0.1445009039e-02,
        0.2492954809e-03,
    ]

    b = b_tab[6]
    for i in range(6):
        b = b_tab[5 - i] + b * tslog

    return b


def cstar(tslog):
    """Compute cstar."""
    c_tab = [
        0.8386993788e00,
        0.4748325276e-01,
        0.3250097527e-01,
        -0.1625859588e-01,
        -0.2260153363e-02,
        0.1844922811e-02,
        -0.2115417788e-03,
    ]

    c = c_tab[6]
    for i in range(6):
        c = c_tab[5 - i] + c * tslog

    return c


def om22_chemkin(tr, dst):
    """Compute OM22."""
    # This is an overhaul of CANTERA version 2.3
    # range of dst
    dst_tab = [0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5]

    # range of tr
    tr_tab = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.2,
        1.4,
        1.6,
        1.8,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
        50.0,
        75.0,
        100.0,
    ]

    # tab of omega22 corresp. to (tr, dst)
    # CANTERA
    omega_tab = [
        4.1005,
        4.266,
        4.833,
        5.742,
        6.729,
        8.624,
        10.34,
        11.89,
        3.2626,
        3.305,
        3.516,
        3.914,
        4.433,
        5.57,
        6.637,
        7.618,
        2.8399,
        2.836,
        2.936,
        3.168,
        3.511,
        4.329,
        5.126,
        5.874,
        2.531,
        2.522,
        2.586,
        2.749,
        3.004,
        3.64,
        4.282,
        4.895,
        2.2837,
        2.277,
        2.329,
        2.46,
        2.665,
        3.187,
        3.727,
        4.249,
        2.0838,
        2.081,
        2.13,
        2.243,
        2.417,
        2.862,
        3.329,
        3.786,
        1.922,
        1.924,
        1.97,
        2.072,
        2.225,
        2.614,
        3.028,
        3.435,
        1.7902,
        1.795,
        1.84,
        1.934,
        2.07,
        2.417,
        2.788,
        3.156,
        1.6823,
        1.689,
        1.733,
        1.82,
        1.944,
        2.258,
        2.596,
        2.933,
        1.5929,
        1.601,
        1.644,
        1.725,
        1.838,
        2.124,
        2.435,
        2.746,
        1.4551,
        1.465,
        1.504,
        1.574,
        1.67,
        1.913,
        2.181,
        2.451,
        1.3551,
        1.365,
        1.4,
        1.461,
        1.544,
        1.754,
        1.989,
        2.228,
        1.28,
        1.289,
        1.321,
        1.374,
        1.447,
        1.63,
        1.838,
        2.053,
        1.2219,
        1.231,
        1.259,
        1.306,
        1.37,
        1.532,
        1.718,
        1.912,
        1.1757,
        1.184,
        1.209,
        1.251,
        1.307,
        1.451,
        1.618,
        1.795,
        1.0933,
        1.1,
        1.119,
        1.15,
        1.193,
        1.304,
        1.435,
        1.578,
        1.0388,
        1.044,
        1.059,
        1.083,
        1.117,
        1.204,
        1.31,
        1.428,
        0.99963,
        1.004,
        1.016,
        1.035,
        1.062,
        1.133,
        1.22,
        1.319,
        0.96988,
        0.9732,
        0.983,
        0.9991,
        1.021,
        1.079,
        1.153,
        1.236,
        0.92676,
        0.9291,
        0.936,
        0.9473,
        0.9628,
        1.005,
        1.058,
        1.121,
        0.89616,
        0.8979,
        0.903,
        0.9114,
        0.923,
        0.9545,
        0.9955,
        1.044,
        0.87272,
        0.8741,
        0.878,
        0.8845,
        0.8935,
        0.9181,
        0.9505,
        0.9893,
        0.85379,
        0.8549,
        0.858,
        0.8632,
        0.8703,
        0.8901,
        0.9164,
        0.9482,
        0.83795,
        0.8388,
        0.8414,
        0.8456,
        0.8515,
        0.8678,
        0.8895,
        0.916,
        0.82435,
        0.8251,
        0.8273,
        0.8308,
        0.8356,
        0.8493,
        0.8676,
        0.8901,
        0.80184,
        0.8024,
        0.8039,
        0.8065,
        0.8101,
        0.8201,
        0.8337,
        0.8504,
        0.78363,
        0.784,
        0.7852,
        0.7872,
        0.7899,
        0.7976,
        0.8081,
        0.8212,
        0.76834,
        0.7687,
        0.7696,
        0.7712,
        0.7733,
        0.7794,
        0.7878,
        0.7983,
        0.75518,
        0.7554,
        0.7562,
        0.7575,
        0.7592,
        0.7642,
        0.7711,
        0.7797,
        0.74364,
        0.7438,
        0.7445,
        0.7455,
        0.747,
        0.7512,
        0.7569,
        0.7642,
        0.71982,
        0.72,
        0.7204,
        0.7211,
        0.7221,
        0.725,
        0.7289,
        0.7339,
        0.70097,
        0.7011,
        0.7014,
        0.7019,
        0.7026,
        0.7047,
        0.7076,
        0.7112,
        0.68545,
        0.6855,
        0.6858,
        0.6861,
        0.6867,
        0.6883,
        0.6905,
        0.6932,
        0.67232,
        0.6724,
        0.6726,
        0.6728,
        0.6733,
        0.6743,
        0.6762,
        0.6784,
        0.65099,
        0.651,
        0.6512,
        0.6513,
        0.6516,
        0.6524,
        0.6534,
        0.6546,
        0.61397,
        0.6141,
        0.6143,
        0.6145,
        0.6147,
        0.6148,
        0.6148,
        0.6147,
        0.5887,
        0.5889,
        0.5894,
        0.59,
        0.5903,
        0.5901,
        0.5895,
        0.5885,
    ]

    # First test on tr
    if tr > 75.0:
        omeg12 = 0.703 - 0.146e-2 * tr + 0.357e-5 * tr * tr - 0.343e-8 * tr * tr * tr
    else:
        # Find tr idx in tr_tab
        if tr <= 0.2:
            ii = 1
        else:
            ii = 36
        for i in range(1, 37):
            if (tr > tr_tab[i - 1]) and (tr <= tr_tab[i]):
                ii = i
                break
        # Find dst idx in dst_tab
        if abs(dst) >= 1.0e-5:
            if dst <= 0.25:
                kk = 1
            else:
                kk = 6
            for i in range(1, 7):
                if (dst_tab[i - 1] < dst) and (dst_tab[i] >= dst):
                    kk = i
                    break
            # Find surrounding values and interpolate
            # First on dst
            vert = np.zeros(3)
            for i in range(3):
                arg = np.zeros(3)
                val = np.zeros(3)
                for k in range(3):
                    arg[k] = dst_tab[kk - 1 + k]
                    val[k] = omega_tab[8 * (ii - 1 + i) + (kk - 1 + k)]
                vert[i] = qinterp(dst, arg, val)
            # Second on tr
            arg = np.zeros(3)
            for i in range(3):
                arg[i] = tr_tab[ii - 1 + i]
            omeg12 = qinterp(tr, arg, vert)
        else:
            arg = np.zeros(3)
            val = np.zeros(3)
            for i in range(3):
                arg[i] = tr_tab[ii - 1 + i]
                val[i] = omega_tab[8 * (ii - 1 + i)]
            omeg12 = qinterp(tr, arg, val)
    return omeg12


def om11_chemkin(tr, dst):
    """Compute OM11."""
    # This is an overhaul of CANTERA version 2.3
    # range of dst
    dst_tab = [0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5]

    # range of tr
    tr_tab = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.2,
        1.4,
        1.6,
        1.8,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
        50.0,
        75.0,
        100.0,
    ]

    # tab of omega11 corresp. to (tr, dst)
    # CANTERA
    omega_tab = [
        4.008,
        4.002,
        4.655,
        5.52,
        6.454,
        8.214,
        9.824,
        11.31,
        3.130,
        3.164,
        3.355,
        3.721,
        4.198,
        5.23,
        6.225,
        7.160,
        2.649,
        2.657,
        2.77,
        3.002,
        3.319,
        4.054,
        4.785,
        5.483,
        2.314,
        2.32,
        2.402,
        2.572,
        2.812,
        3.386,
        3.972,
        4.539,
        2.066,
        2.073,
        2.14,
        2.278,
        2.472,
        2.946,
        3.437,
        3.918,
        1.877,
        1.885,
        1.944,
        2.06,
        2.225,
        2.628,
        3.054,
        3.747,
        1.729,
        1.738,
        1.79,
        1.893,
        2.036,
        2.388,
        2.763,
        3.137,
        1.6122,
        1.622,
        1.67,
        1.76,
        1.886,
        2.198,
        2.535,
        2.872,
        1.517,
        1.527,
        1.572,
        1.653,
        1.765,
        2.044,
        2.35,
        2.657,
        1.44,
        1.45,
        1.49,
        1.564,
        1.665,
        1.917,
        2.196,
        2.4780,
        1.3204,
        1.33,
        1.364,
        1.425,
        1.51,
        1.72,
        1.956,
        2.199,
        1.234,
        1.24,
        1.272,
        1.324,
        1.394,
        1.573,
        1.777,
        1.99,
        1.168,
        1.176,
        1.202,
        1.246,
        1.306,
        1.46,
        1.64,
        1.827,
        1.1166,
        1.124,
        1.146,
        1.185,
        1.237,
        1.372,
        1.53,
        1.7,
        1.075,
        1.082,
        1.102,
        1.135,
        1.181,
        1.3,
        1.441,
        1.592,
        1.0006,
        1.005,
        1.02,
        1.046,
        1.08,
        1.17,
        1.278,
        1.397,
        0.95,
        0.9538,
        0.9656,
        0.9852,
        1.012,
        1.082,
        1.168,
        1.265,
        0.9131,
        0.9162,
        0.9256,
        0.9413,
        0.9626,
        1.019,
        1.09,
        1.17,
        0.8845,
        0.8871,
        0.8948,
        0.9076,
        0.9252,
        0.972,
        1.03,
        1.098,
        0.8428,
        0.8446,
        0.850,
        0.859,
        0.8716,
        0.9053,
        0.9483,
        0.9984,
        0.813,
        0.8142,
        0.8183,
        0.825,
        0.8344,
        0.8598,
        0.8927,
        0.9316,
        0.7898,
        0.791,
        0.794,
        0.7993,
        0.8066,
        0.8265,
        0.8526,
        0.8836,
        0.7711,
        0.772,
        0.7745,
        0.7788,
        0.7846,
        0.8007,
        0.822,
        0.8474,
        0.7555,
        0.7562,
        0.7584,
        0.7619,
        0.7667,
        0.78,
        0.7976,
        0.8189,
        0.7422,
        0.743,
        0.7446,
        0.7475,
        0.7515,
        0.7627,
        0.7776,
        0.796,
        0.72022,
        0.7206,
        0.722,
        0.7241,
        0.7271,
        0.7354,
        0.7464,
        0.76,
        0.7025,
        0.703,
        0.704,
        0.7055,
        0.7078,
        0.7142,
        0.7228,
        0.7334,
        0.68776,
        0.688,
        0.6888,
        0.6901,
        0.6919,
        0.697,
        0.704,
        0.7125,
        0.6751,
        0.6753,
        0.676,
        0.677,
        0.6785,
        0.6827,
        0.6884,
        0.6955,
        0.664,
        0.6642,
        0.6648,
        0.6657,
        0.6669,
        0.6704,
        0.6752,
        0.681,
        0.6414,
        0.6415,
        0.6418,
        0.6425,
        0.6433,
        0.6457,
        0.649,
        0.653,
        0.6235,
        0.6236,
        0.6239,
        0.6243,
        0.6249,
        0.6267,
        0.629,
        0.632,
        0.60882,
        0.6089,
        0.6091,
        0.6094,
        0.61,
        0.6112,
        0.613,
        0.6154,
        0.5964,
        0.5964,
        0.5966,
        0.597,
        0.5972,
        0.5983,
        0.600,
        0.6017,
        0.5763,
        0.5763,
        0.5764,
        0.5766,
        0.5768,
        0.5775,
        0.5785,
        0.58,
        0.5415,
        0.5415,
        0.5416,
        0.5416,
        0.5418,
        0.542,
        0.5424,
        0.543,
        0.518,
        0.518,
        0.5182,
        0.5184,
        0.5184,
        0.5185,
        0.5186,
        0.5187,
    ]

    # First test on tr
    if tr > 75.0:
        omeg12 = 0.623 - 0.136e-2 * tr + 0.346e-5 * tr * tr - 0.343e-8 * tr * tr * tr
    else:
        # Find tr idx in tr_tab
        if tr <= 0.2:
            ii = 1
        else:
            ii = 36
        for i in range(1, 37):
            if (tr > tr_tab[i - 1]) and (tr <= tr_tab[i]):
                ii = i
                break
        # Find dst idx in dst_tab
        if abs(dst) >= 1.0e-5:
            if dst <= 0.25:
                kk = 1
            else:
                kk = 6
            for i in range(1, 7):
                if (dst_tab[i - 1] < dst) and (dst_tab[i] >= dst):
                    kk = i
                    break
            # Find surrounding values and interpolate
            # First on dst
            vert = np.zeros(3)
            for i in range(3):
                arg = np.zeros(3)
                val = np.zeros(3)
                for k in range(3):
                    arg[k] = dst_tab[kk - 1 + k]
                    val[k] = omega_tab[8 * (ii - 1 + i) + (kk - 1 + k)]
                vert[i] = qinterp(dst, arg, val)
            # Second on tr
            arg = np.zeros(3)
            for i in range(3):
                arg[i] = tr_tab[ii - 1 + i]
            omeg12 = qinterp(tr, arg, vert)
        else:
            arg = np.zeros(3)
            val = np.zeros(3)
            for i in range(3):
                arg[i] = tr_tab[ii - 1 + i]
                val[i] = omega_tab[8 * (ii - 1 + i)]
            omeg12 = qinterp(tr, arg, val)
    return omeg12


def qinterp(x0, x, y):
    """Compute interpolated value."""
    val1 = y[0] + (x0 - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    val2 = y[1] + (x0 - x[1]) * (y[2] - y[1]) / (x[2] - x[1])
    fac1 = (x0 - x[0]) / (x[1] - x[0]) / 2.0
    fac2 = (x[2] - x0) / (x[2] - x[1]) / 2.0
    if x0 >= x[1]:
        val = (val1 * fac2 + val2) / (1.0 + fac2)
    else:
        val = (val1 + val2 * fac1) / (1.0 + fac1)
    return val


def f_corr(t, eps_k):
    """Compute f_corr value."""
    thtwo = 3.0 / 2.0
    return (
        1
        + np.pi ** (thtwo) / 2.0 * np.sqrt((eps_k / t))
        + (np.pi**2 / 4.0 + 2.0) * ((eps_k / t))
        + ((np.pi * eps_k / t)) ** (thtwo)
    )


def xi(spec1, spec2, species_transport):
    """Compute xi."""
    dipmin = 1e-20
    # Ern & Giovangigli, Multicomponent Transport Algorithms, (1994),
    # eq. (6.1.5)
    # 1 is polar, 2 is nonpolar
    if (float(species_transport[spec2][3]) < dipmin) and (
        float(species_transport[spec1][3]) > dipmin
    ):
        xi = 1.0 + 1.0 / 4.0 * red_pol(spec2, species_transport) * red_dip(
            spec1, species_transport
        ) * red_dip(spec1, species_transport) * np.sqrt(
            float(species_transport[spec1][1]) / float(species_transport[spec2][1])
        )
    # 1 is nonpolar, 2 is polar
    elif (float(species_transport[spec2][3]) > dipmin) and (
        float(species_transport[spec1][3]) < dipmin
    ):
        xi = 1.0 + 1.0 / 4.0 * red_pol(spec1, species_transport) * red_dip(
            spec2, species_transport
        ) * red_dip(spec2, species_transport) * np.sqrt(
            float(species_transport[spec2][1]) / float(species_transport[spec1][1])
        )
    # normal case, either both polar or both nonpolar
    else:
        xi = 1.0

    return xi


def xi_bool(spec1, spec2, species_transport):
    """Compute the boolean of xi."""
    dipmin = 1e-20
    # 1 is polar, 2 is nonpolar
    if (float(species_transport[spec2][3]) < dipmin) and (
        float(species_transport[spec1][3]) > dipmin
    ):
        xi_b = False
    # 1 is nonpolar, 2 is polar
    elif (float(species_transport[spec2][3]) > dipmin) and (
        float(species_transport[spec1][3]) < dipmin
    ):
        xi_b = False
    # normal case, either both polar or both nonpolar
    else:
        xi_b = True

    return xi_b


def red_pol(spec, species_transport):
    """Compute polarization value."""
    return float(species_transport[spec][4]) / float(species_transport[spec][2]) ** 3.0


def red_dip(spec, species_transport):
    """Compute dipole value."""
    # compute single constants in g/cm/s
    # conversion coefs
    a2cm = 1.0e-8
    debye2cgs = 1.0e-18
    convert = debye2cgs / np.sqrt(cc.kb * a2cm**3.0)
    return (
        convert
        * float(species_transport[spec][3])
        / np.sqrt(
            float(species_transport[spec][1]) * float(species_transport[spec][2]) ** 3.0
        )
    )


def critical_parameters(fstream, mechanism, species_info):
    """Write the critical parameters."""
    tabulated_critical_params = {
        "H2": {
            "Tci": 33.145,
            "Pci": 12.964,
            "wt": 2.01588,
            "acentric_factor": -0.219,
        },
        "O2": {
            "Tci": 154.581,
            "Pci": 50.4304658,
            "wt": 31.9988,
            "acentric_factor": 0.0222,
        },
        "H2O": {
            "Tci": 647.096,
            "Pci": 220.640,
            "wt": 18.015340,
            "acentric_factor": 0.3443,
        },
        "N2": {
            "Tci": 126.192,
            "Pci": 33.958,
            "wt": 28.013400,
            "acentric_factor": 0.0372,
        },
        "CH4": {
            "Tci": 190.56,
            "Pci": 45.99,
            "wt": 16.043030,
            "acentric_factor": 0.011,
        },
        "C2H6": {
            "Tci": 305.32,
            "Pci": 48.72,
            "wt": 30.070120,
            "acentric_factor": 0.099,
        },
        "C3H8": {
            "Tci": 369.83,
            "Pci": 42.48,
            "wt": 44.097210,
            "acentric_factor": 0.152,
        },
        "CO2": {
            "Tci": 304.12,
            "Pci": 73.74,
            "wt": 44.009950,
            "acentric_factor": 0.225,
        },
        "He": {
            "Tci": 5.1953,
            "Pci": 2.2746,
            "wt": 4.002602,
            "acentric_factor": -0.382,
        },
        "CO": {
            "Tci": 132.85,
            "Pci": 34.94,
            "wt": 28.010,
            "acentric_factor": 0.045,
        },
        "AR": {
            "Tci": 150.86,
            "Pci": 48.98,
            "wt": 39.948,
            "acentric_factor": -0.002,
        },
        "NO": {
            "Tci": 180.0,
            "Pci": 64.80,
            "wt": 30.006,
            "acentric_factor": 0.582,
        },
        "CH3OH": {
            "Tci": 512.64,
            "Pci": 80.97,
            "wt": 32.042,
            "acentric_factor": 0.565,
        },
        "C2H2": {
            "Tci": 308.30,
            "Pci": 61.14,
            "wt": 26.038,
            "acentric_factor": 0.189,
        },
        "C2H4": {
            "Tci": 282.34,
            "Pci": 50.41,
            "wt": 28.054,
            "acentric_factor": 0.087,
        },
        "N2O": {
            "Tci": 309.60,
            "Pci": 72.55,
            "wt": 44.013,
            "acentric_factor": 0.162,
        },
    }

    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("compute the critical parameters for each species"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void"
        " GET_CRITPARAMS(amrex::Real *  Tci, amrex::Real *  ai,"
        " amrex::Real *  bi, amrex::Real *  acentric_i)",
    )
    cw.writer(fstream, "{")
    cw.writer(fstream)

    cw.writer(fstream, f"amrex::Real   EPS[{n_species}];")
    cw.writer(fstream, f"amrex::Real   SIG[{n_species}];")
    cw.writer(fstream, f"amrex::Real    wt[{n_species}];")
    cw.writer(fstream, "amrex::Real Rcst = 83.144598;" + cw.comment("in bar [CGS] !"))
    if not all(
        (species.name in tabulated_critical_params)
        for species in species_info.nonqssa_species
    ):
        cw.writer(fstream, "amrex::Real avogadro = 6.02214199e23;")
        cw.writer(
            fstream,
            "amrex::Real boltzmann = 1.3806503e-16;" + cw.comment("we work in CGS"),
        )

    cw.writer(fstream)

    cw.writer(fstream, "egtransetEPS(EPS);")
    cw.writer(fstream, "egtransetSIG(SIG);")
    cw.writer(fstream, "get_mw(wt);")

    for species in species_info.nonqssa_species:
        if species.name in tabulated_critical_params:
            cw.writer(fstream)
            cw.writer(
                fstream,
                cw.comment(f"species {species.idx}: {species.name}"),
            )
            cw.writer(fstream, cw.comment("Imported from NIST"))
            cw.writer(
                fstream,
                f"Tci[{species.idx}] ="
                f" {tabulated_critical_params[species.name]['Tci']:f} ; ",
            )
            cw.writer(
                fstream,
                f"ai[{species.idx}] = 1e6 * 0.42748 * Rcst * Rcst *"
                f" Tci[{species.idx}] * Tci[{species.idx}] /"
                f" ({tabulated_critical_params[species.name]['wt']:f} *"
                f" {tabulated_critical_params[species.name]['wt']:f} *"
                f" {tabulated_critical_params[species.name]['Pci']:f}); ",
            )
            cw.writer(
                fstream,
                f"bi[{species.idx}] = 0.08664 * Rcst * Tci[{species.idx}]"
                f" / ({tabulated_critical_params[species.name]['wt']:f} *"
                f" {tabulated_critical_params[species.name]['Pci']:f}); ",
            )
            cw.writer(
                fstream,
                f"acentric_i[{species.idx}] ="
                f" {tabulated_critical_params[species.name]['acentric_factor']:f} ;",
            )
        else:
            cw.writer(fstream)
            cw.writer(
                fstream,
                cw.comment(f"species {species.idx}: {species.name}"),
            )
            cw.writer(
                fstream,
                f"Tci[{species.idx}] = 1.316 * EPS[{species.idx}] ; ",
            )
            cw.writer(
                fstream,
                f"ai[{species.idx}] = (5.55 * avogadro * avogadro *"
                f" EPS[{species.idx}]*boltzmann * 1e-24 *"
                f" SIG[{species.idx}] * SIG[{species.idx}] *"
                f" SIG[{species.idx}] ) / (wt[{species.idx}] *"
                f" wt[{species.idx}]); ",
            )
            cw.writer(
                fstream,
                f"bi[{species.idx}] = 0.855 * avogadro * 1e-24 *"
                f" SIG[{species.idx}] * SIG[{species.idx}] *"
                f" SIG[{species.idx}] / (wt[{species.idx}]); ",
            )
            cw.writer(fstream, f"acentric_i[{species.idx}] = 0.0 ;")

    cw.writer(fstream)
    cw.writer(fstream, "}")

    # Critical parameters pre-evaluations necessary for SRK
    # SRK parameters - CGS for constants
    f0 = 0.48508e0
    f1 = 1.5517e0
    f2 = -0.151613e0
    rcst = 83.144598
    avogadro = 6.02214199e23
    boltzmann = 1.3806503e-16
    species_transport = analyze_transport(mechanism, species_info)

    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment(
            "compute the critical parameter quantities for each species for SRK"
        ),
    )
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void"
        " GET_CRITPARAMS_SRK(amrex::Real *  sqrtOneOverTc, amrex::Real *  sqrtAsti,"
        " amrex::Real *  Bi, amrex::Real *  Fomega)",
    )

    cw.writer(fstream, "{")
    # Loop over species to compute quantities
    for species in species_info.nonqssa_species:
        cw.writer(fstream)
        cw.writer(
            fstream,
            cw.comment(f"species {species.idx}: {species.name}"),
        )
        if species.name in tabulated_critical_params:
            cw.writer(fstream, cw.comment("Imported from NIST"))
            tci = tabulated_critical_params[species.name]["Tci"]
            ai = (
                1e6
                * 0.42748
                * rcst**2
                * tci**2
                / (
                    tabulated_critical_params[species.name]["wt"] ** 2
                    * tabulated_critical_params[species.name]["Pci"]
                )
            )
            bi = (
                0.08664
                * rcst
                * tci
                / (
                    tabulated_critical_params[species.name]["wt"]
                    * tabulated_critical_params[species.name]["Pci"]
                )
            )
            omega = tabulated_critical_params[species.name]["acentric_factor"]
        else:
            cw.writer(fstream, cw.comment("Computed from Lennard-Jones"))
            eps = float(species_transport[species][1])
            sig = float(species_transport[species][2])
            wt = species.weight
            tci = 1.316 * eps
            ai = 5.55 * avogadro**2 * eps * boltzmann * 1e-24 * sig**3 / wt**2
            bi = 0.855 * avogadro * 1e-24 * sig**3 / wt
            omega = 0.0
        sqrt_oneovertc = np.sqrt(1.0 / tci)
        sqrt_asti = np.sqrt(ai)
        fomega = f0 + omega * (f1 + f2 * omega)

        cw.writer(fstream, f"sqrtOneOverTc[{species.idx}] = {sqrt_oneovertc:.13e};")
        cw.writer(fstream, f"sqrtAsti[{species.idx}] = {sqrt_asti:.13e};")
        cw.writer(fstream, f"Bi[{species.idx}] = {bi:.13e};")
        cw.writer(fstream, f"Fomega[{species.idx}] = {fomega:.13e};")

    cw.writer(fstream)
    cw.writer(fstream, "}")
