"""CK routines."""

import ceptr.constants as cc
import ceptr.thermo as cth
import ceptr.writer as cw


def ckawt(fstream, mechanism):
    """Write ckawt."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("get atomic weight for all elements"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKAWT"
        + cc.sym
        + "( amrex::Real *  awt)",
    )
    cw.writer(fstream, "{")
    cw.writer(fstream, "atomicWeight(awt);")
    cw.writer(fstream, "}")


def ckncf(fstream, mechanism, species_info):
    """Write ckncf."""
    n_elements = mechanism.n_elements
    n_species = species_info.n_species

    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns the elemental composition "))
    cw.writer(fstream, cw.comment("of the speciesi (mdim is num of elements)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKNCF" + cc.sym + "(int * ncf)",
    )
    cw.writer(fstream, "{")
    cw.writer(fstream, f"int kd = {n_elements}; ")
    cw.writer(fstream, cw.comment("Zero ncf"))
    cw.writer(fstream, f"for (int id = 0; id < kd * {n_species}; ++ id) {{")
    cw.writer(fstream, " ncf[id] = 0; ")
    cw.writer(fstream, "}")

    cw.writer(fstream)
    for sp in species_info.nonqssa_species_list:
        spec_idx = species_info.ordered_idx_map[sp]
        species = species_info.nonqssa_species[spec_idx]
        cw.writer(fstream, cw.comment(f"{species.name}"))
        for elem, coef in mechanism.species(sp).composition.items():
            cw.writer(
                fstream,
                f"ncf[ {species_info.ordered_idx_map[sp]} * kd +"
                f" {mechanism.element_index(elem)} ] = {int(coef)}; "
                + cw.comment(f"{elem}"),
            )
        cw.writer(fstream)
    cw.writer(fstream, "}")


def cksyme_str(fstream, mechanism, species_info):
    """Write cksyme."""
    n_elements = mechanism.n_elements
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns the vector of strings of element names"))
    cw.writer(
        fstream,
        "void CKSYME_STR" + cc.sym + "(amrex::Vector<std::string>& ename)",
    )
    cw.writer(fstream, "{")
    cw.writer(fstream, f"ename.resize({n_elements});")
    for elem in mechanism.element_names:
        cw.writer(
            fstream,
            f'ename[{mechanism.element_index(elem)}] = "{elem}";',
        )
    cw.writer(fstream, "}")


def cksyms_str(fstream, mechanism, species_info):
    """Write cksyms."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns the vector of strings of species names"))
    cw.writer(
        fstream,
        "void CKSYMS_STR" + cc.sym + "(amrex::Vector<std::string>& kname)",
    )
    cw.writer(fstream, "{")
    cw.writer(fstream, f"kname.resize({n_species});")
    for species in species_info.nonqssa_species_list:
        cw.writer(
            fstream,
            f'kname[{species_info.ordered_idx_map[species]}] = "{species}";',
        )
    cw.writer(fstream, "}")


def ckindx(fstream, mechanism, species_info):
    """Write ckindx."""
    cw.writer(fstream)
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("A few mechanism parameters"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKINDX"
        + cc.sym
        + "(int& mm, int& kk, int& ii, int& nfit)",
    )
    cw.writer(fstream, "{")
    cw.writer(fstream, f"mm = {mechanism.n_elements};")
    cw.writer(fstream, f"kk = {species_info.n_species};")
    cw.writer(fstream, f"ii = {mechanism.n_reactions};")
    cw.writer(fstream, "nfit = -1; " + cw.comment("Why do you need this anyway ? "))
    cw.writer(fstream, "}")


def ckrp(fstream, mechanism, species_info):
    """Write ckrp."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment(" Returns R, Rc, Patm"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKRP"
        + cc.sym
        + "(amrex::Real& ru, amrex::Real& ruc, amrex::Real& pa)",
    )
    cw.writer(fstream, "{")
    cw.writer(
        fstream,
        f" ru  = {(cc.R * cc.ureg.mole * cc.ureg.kelvin / cc.ureg.erg).m:1.14e}; ",
    )
    cw.writer(
        fstream,
        f" ruc = {(cc.Rc * (cc.ureg.mole * cc.ureg.kelvin / cc.ureg.cal)).m:.20f}; ",
    )
    cw.writer(fstream, f" pa  = {cc.Patm:g}; ")
    cw.writer(fstream, "}")


def ckcpbl(fstream, mechanism, species_info):
    """Write ckpbl."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns the mean specific heat at CP (Eq. 33)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCPBL"
        + cc.sym
        + "(const amrex::Real T, const amrex::Real x[], amrex::Real& cpbl)",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real result = 0; ")

    cw.writer(
        fstream,
        f"amrex::Real cpor[{n_species}]; " + cw.comment(" temporary storage"),
    )

    # call routine
    cw.writer(fstream, "cp_R(cpor, T);")

    # dot product
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("perform dot product"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")

    cw.writer(fstream, "result += x[id]*cpor[id];")

    cw.writer(fstream, "}")

    cw.writer(fstream)
    cw.writer(
        fstream,
        "cpbl = result *"
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e};",
    )
    cw.writer(fstream, "}")


def ckcpbs(fstream, mechanism, species_info):
    """Write ckpbs."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns the mean specific heat at CP (Eq. 34)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCPBS"
        + cc.sym
        + "(const amrex::Real T, const amrex::Real y[], amrex::Real& cpbs)",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real result = 0.0; ")

    cw.writer(fstream)

    models = cth.analyze_thermodynamics(mechanism, species_info.nonqssa_species_list)
    cw.writer(fstream, cw.comment("compute Cp/R at the given temperature"))
    cth.generate_thermo_routine(
        fstream,
        species_info,
        "cp_R",
        models,
        0,
        None,
        True,
    )
    cw.writer(fstream)
    cw.writer(
        fstream,
        "cpbs = result *"
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e};",
    )
    cw.writer(fstream, "}")


def ckcvbl(fstream, mechanism, species_info):
    """Write ckcvbl."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns the mean specific heat at CV (Eq. 35)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCVBL"
        + cc.sym
        + "(const amrex::Real T, const amrex::Real x[], amrex::Real& cvbl)",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real result = 0; ")

    cw.writer(
        fstream,
        f"amrex::Real cvor[{n_species}]; " + cw.comment(" temporary storage"),
    )

    # call routine
    cw.writer(fstream, "cv_R(cvor, T);")

    # dot product
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("perform dot product"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")

    cw.writer(fstream, "result += x[id]*cvor[id];")

    cw.writer(fstream, "}")

    cw.writer(fstream)
    cw.writer(
        fstream,
        "cvbl = result *"
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e};",
    )
    cw.writer(fstream, "}")


def ckcvbs(fstream, mechanism, species_info):
    """Write ckcvbs."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns the mean specific heat at CV (Eq. 36)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCVBS"
        + cc.sym
        + "(const amrex::Real T, const amrex::Real y[],  amrex::Real& cvbs)",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real result = 0.0; ")

    models = cth.analyze_thermodynamics(mechanism, species_info.nonqssa_species_list)
    cw.writer(fstream, cw.comment("compute Cv/R at the given temperature"))
    cth.generate_thermo_routine(
        fstream,
        species_info,
        "cv_R",
        models,
        0,
        None,
        True,
    )

    cw.writer(fstream)
    cw.writer(
        fstream,
        "cvbs = result *"
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e};",
    )
    cw.writer(fstream, "}")


def ckhbml(fstream, mechanism, species_info):
    """Write ckhbml."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("Returns the mean enthalpy of the mixture in molar units"),
    )
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKHBML"
        + cc.sym
        + "(const amrex::Real T, const amrex::Real x[], amrex::Real& hbml)",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real result = 0; ")

    cw.writer(
        fstream,
        f"amrex::Real hml[{n_species}]; " + cw.comment(" temporary storage"),
    )
    cw.writer(
        fstream,
        "amrex::Real RT ="
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e}*T; "
        + cw.comment("R*T"),
    )

    # call routine
    cw.writer(fstream, "speciesEnthalpy(hml, T);")

    # dot product
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("perform dot product"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")

    cw.writer(fstream, "result += x[id]*hml[id];")

    cw.writer(fstream, "}")

    cw.writer(fstream)
    cw.writer(fstream, "hbml = result * RT;")
    cw.writer(fstream, "}")


def ckhbms(fstream, mechanism, species_info):
    """Write ckhbms."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns mean enthalpy of mixture in mass units"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKHBMS"
        + cc.sym
        + "(const amrex::Real T, const amrex::Real y[],  amrex::Real& hbms)",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real result = 0.0;")

    models = cth.analyze_thermodynamics(mechanism, species_info.nonqssa_species_list)
    cth.generate_thermo_routine(
        fstream,
        species_info,
        "speciesEnthalpy",
        models,
        0,
        None,
        True,
    )
    cw.writer(fstream)

    cw.writer(
        fstream,
        "const amrex::Real RT ="
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e}*T; "
        + cw.comment("R*T"),
    )

    cw.writer(fstream)
    cw.writer(fstream, "hbms = result * RT;")
    cw.writer(fstream, "}")


def ckubml(fstream, mechanism, species_info):
    """Write ckubml."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("get mean internal energy in molar units"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKUBML"
        + cc.sym
        + "(const amrex::Real T, const amrex::Real x[], amrex::Real& ubml)",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real result = 0; ")

    cw.writer(
        fstream,
        f"amrex::Real uml[{n_species}]; " + cw.comment(" temporary energy array"),
    )
    cw.writer(
        fstream,
        "amrex::Real RT ="
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e}*T; "
        + cw.comment("R*T"),
    )

    # call routine
    cw.writer(fstream, "speciesInternalEnergy(uml, T);")

    # dot product
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("perform dot product"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "result += x[id]*uml[id];")
    cw.writer(fstream, "}")
    cw.writer(fstream)
    cw.writer(fstream, "ubml = result * RT;")
    cw.writer(fstream, "}")


def ckubms(fstream, mechanism, species_info):
    """Write ckubms."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("get mean internal energy in mass units"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKUBMS"
        + cc.sym
        + "(const amrex::Real T, const amrex::Real y[], amrex::Real& ubms)",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real result = 0.0;")

    models = cth.analyze_thermodynamics(mechanism, species_info.nonqssa_species_list)
    cth.generate_thermo_routine(
        fstream,
        species_info,
        "speciesInternalEnergy",
        models,
        0,
        None,
        True,
    )
    cw.writer(fstream)

    cw.writer(
        fstream,
        "const amrex::Real RT ="
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e}*T; "
        + cw.comment("R*T"),
    )

    cw.writer(fstream)
    cw.writer(fstream, "ubms = result * RT;")
    cw.writer(fstream, "}")


def cksbml(fstream, mechanism, species_info):
    """Write cksbml."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("get mixture entropy in molar units"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKSBML"
        + cc.sym
        + "(const amrex::Real P, const amrex::Real T,"
        + "const amrex::Real x[], amrex::Real& sbml)",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real result = 0; ")

    cw.writer(
        fstream,
        cw.comment("Log of normalized pressure in cgs units dynes/cm^2 by Patm"),
    )
    cw.writer(fstream, "amrex::Real logPratio = log ( P / 1013250.0 ); ")
    cw.writer(
        fstream,
        f"amrex::Real sor[{n_species}]; " + cw.comment(" temporary storage"),
    )

    # call routine
    cw.writer(fstream, "speciesEntropy(sor, T);")

    # Equation 42
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Compute Eq 42"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")

    cw.writer(
        fstream,
        f"result += x[id]*(sor[id]-log((x[id]+{cc.smallnum:g}))-logPratio);",
    )

    cw.writer(fstream, "}")

    cw.writer(fstream)

    cw.writer(
        fstream,
        "sbml = result *"
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e};",
    )
    cw.writer(fstream, "}")


def cksbms(fstream, mechanism, species_info):
    """Write cksbms."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("get mixture entropy in mass units"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKSBMS"
        + cc.sym
        + "(const amrex::Real P, const amrex::Real T,"
        + "const amrex::Real y[], amrex::Real& sbms)",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real result = 0; ")

    cw.writer(
        fstream,
        cw.comment("Log of normalized pressure in cgs units dynes/cm^2 by Patm"),
    )
    cw.writer(fstream, "amrex::Real logPratio = log ( P / 1013250.0 ); ")
    cw.writer(
        fstream,
        f"amrex::Real sor[{n_species}]; " + cw.comment(" temporary storage"),
    )
    cw.writer(
        fstream,
        f"amrex::Real x[{n_species}]; " + cw.comment(" need a ytx conversion"),
    )

    cw.writer(
        fstream,
        "amrex::Real YOW = 0; " + cw.comment("See Eq 4, 6 in CK Manual"),
    )

    # compute inverse of mean molecular weight first (eq 3)
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Compute inverse of mean molecular wt first"))
    cw.writer(
        fstream,
        f"for (int i = 0; i < {len(species_info.nonqssa_species_list)}; i++)",
    )
    cw.writer(fstream, "{")
    cw.writer(fstream, "YOW += y[i]* imw(i);")
    cw.writer(fstream, "}")

    # now to ytx
    cw.writer(fstream, cw.comment("Now compute y to x conversion"))
    for sp in species_info.nonqssa_species_list:
        spec_idx = species_info.ordered_idx_map[sp]
        species = species_info.nonqssa_species[spec_idx]
        cw.writer(
            fstream,
            f"x[{spec_idx}] = y[{spec_idx}]/({species.weight:.16e}*YOW); ",
        )

    # call routine
    cw.writer(fstream, "speciesEntropy(sor, T);")

    # Equation 42 and 43
    cw.writer(fstream, cw.comment("Perform computation in Eq 42 and 43"))
    cw.writer(
        fstream,
        f"for (int i = 0; i < {len(species_info.nonqssa_species_list)}; i++)",
    )
    cw.writer(fstream, "{")
    cw.writer(
        fstream,
        f"result += x[i]*(sor[i]-log((x[i]+{cc.smallnum:g}))-logPratio);",
    )
    cw.writer(fstream, "}")

    cw.writer(fstream, cw.comment("Scale by R/W"))
    cw.writer(
        fstream,
        "sbms = result *"
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e} *"
        " YOW;",
    )
    cw.writer(fstream, "}")


def ckgbml(fstream, mechanism, species_info):
    """Write ckgbml."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns mean gibbs free energy in molar units"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKGBML"
        + cc.sym
        + "(const amrex::Real P, const amrex::Real T,"
        + "const amrex::Real x[], amrex::Real& gbml)",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real result = 0; ")

    cw.writer(
        fstream,
        cw.comment("Log of normalized pressure in cgs units dynes/cm^2 by Patm"),
    )
    cw.writer(fstream, "amrex::Real logPratio = log ( P / 1013250.0 ); ")
    cw.writer(
        fstream,
        "amrex::Real RT ="
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e}*T; "
        + cw.comment("R*T"),
    )
    cw.writer(
        fstream,
        f"amrex::Real gort[{n_species}]; " + cw.comment(" temporary storage"),
    )

    # call routine
    cw.writer(fstream, cw.comment("Compute g/RT"))
    cw.writer(fstream, "gibbs(gort, T);")

    # Equation 44
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Compute Eq 44"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(
        fstream,
        f"result += x[id]*(gort[id]+log((x[id]+{cc.smallnum:g}))+logPratio);",
    )
    cw.writer(fstream, "}")
    cw.writer(fstream)
    cw.writer(fstream, "gbml = result * RT;")
    cw.writer(fstream, "}")


def ckgbms(fstream, mechanism, species_info):
    """Write ckgbms."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns mixture gibbs free energy in mass units"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKGBMS"
        + cc.sym
        + "(const amrex::Real P, const amrex::Real T,"
        + "const amrex::Real y[], amrex::Real& gbms)",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real result = 0; ")

    cw.writer(
        fstream,
        cw.comment("Log of normalized pressure in cgs units dynes/cm^2 by Patm"),
    )
    cw.writer(fstream, "amrex::Real logPratio = log ( P / 1013250.0 ); ")
    cw.writer(
        fstream,
        "amrex::Real RT ="
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e}*T; "
        + cw.comment("R*T"),
    )
    cw.writer(
        fstream,
        f"amrex::Real gort[{n_species}]; " + cw.comment(" temporary storage"),
    )
    cw.writer(
        fstream,
        f"amrex::Real x[{n_species}]; " + cw.comment(" need a ytx conversion"),
    )

    cw.writer(
        fstream,
        "amrex::Real YOW = 0; " + cw.comment("To hold 1/molecularweight"),
    )

    # compute inverse of mean molecular weight first (eq 3)
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Compute inverse of mean molecular wt first"))
    for sp in species_info.nonqssa_species_list:
        spec_idx = species_info.ordered_idx_map[sp]
        species = species_info.nonqssa_species[spec_idx]
        cw.writer(
            fstream,
            f"YOW += y[{spec_idx}]*imw({spec_idx}); " + cw.comment(f"{species.name}"),
        )

    # now to ytx
    cw.writer(fstream, cw.comment("Now compute y to x conversion"))
    for sp in species_info.nonqssa_species_list:
        spec_idx = species_info.ordered_idx_map[sp]
        species = species_info.nonqssa_species[spec_idx]
        cw.writer(
            fstream,
            f"x[{spec_idx}] = y[{spec_idx}]/({species.weight:.16e}*YOW); ",
        )

    # call routine
    cw.writer(fstream, "gibbs(gort, T);")

    # Equation 42 and 43
    cw.writer(fstream, cw.comment("Perform computation in Eq 44"))
    for sp in species_info.nonqssa_species_list:
        idx = species_info.ordered_idx_map[sp]
        cw.writer(
            fstream,
            "result +="
            f" x[{idx}]*(gort[{idx}]+log((x[{idx}]+{cc.smallnum:g}))+logPratio);",
        )

    cw.writer(fstream, cw.comment("Scale by RT/W"))
    cw.writer(fstream, "gbms = result * RT * YOW;")
    cw.writer(fstream, "}")


def ckabml(fstream, mechanism, species_info):
    """Write ckabml."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("Returns mean helmholtz free energy in molar units"),
    )
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKABML"
        + cc.sym
        + "(const amrex::Real P, const amrex::Real T,"
        + "const amrex::Real x[], amrex::Real& abml)",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real result = 0; ")

    cw.writer(
        fstream,
        cw.comment("Log of normalized pressure in cgs units dynes/cm^2 by Patm"),
    )
    cw.writer(fstream, "amrex::Real logPratio = log ( P / 1013250.0 ); ")
    cw.writer(
        fstream,
        "amrex::Real RT ="
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e}*T; "
        + cw.comment("R*T"),
    )
    cw.writer(
        fstream,
        f"amrex::Real aort[{n_species}]; " + cw.comment(" temporary storage"),
    )

    # call routine
    cw.writer(fstream, cw.comment("Compute g/RT"))
    cw.writer(fstream, "helmholtz(aort, T);")

    # Equation 44
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Compute Eq 44"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")

    cw.writer(
        fstream,
        f"result += x[id]*(aort[id]+log((x[id]+{cc.smallnum:g}))+logPratio);",
    )

    cw.writer(fstream, "}")
    cw.writer(fstream)
    cw.writer(fstream, "abml = result * RT;")
    cw.writer(fstream, "}")


def ckabms(fstream, mechanism, species_info):
    """Write ckabms."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("Returns mixture helmholtz free energy in mass units"),
    )
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKABMS"
        + cc.sym
        + "(const amrex::Real P, const amrex::Real T,"
        + "const amrex::Real y[], amrex::Real& abms)",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real result = 0; ")

    cw.writer(
        fstream,
        cw.comment("Log of normalized pressure in cgs units dynes/cm^2 by Patm"),
    )
    cw.writer(fstream, "amrex::Real logPratio = log ( P / 1013250.0 ); ")
    cw.writer(
        fstream,
        "amrex::Real RT ="
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e}*T; "
        + cw.comment("R*T"),
    )
    cw.writer(
        fstream,
        f"amrex::Real aort[{n_species}]; " + cw.comment(" temporary storage"),
    )
    cw.writer(
        fstream,
        f"amrex::Real x[{n_species}]; " + cw.comment(" need a ytx conversion"),
    )

    cw.writer(
        fstream,
        "amrex::Real YOW = 0; " + cw.comment("To hold 1/molecularweight"),
    )

    # compute inverse of mean molecular weight first (eq 3)
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Compute inverse of mean molecular wt first"))
    for sp in species_info.nonqssa_species_list:
        spec_idx = species_info.ordered_idx_map[sp]
        species = species_info.nonqssa_species[spec_idx]
        cw.writer(
            fstream,
            f"YOW += y[{spec_idx}]*imw({spec_idx}); " + cw.comment(f"{species.name}"),
        )

    # now to ytx
    cw.writer(fstream, cw.comment("Now compute y to x conversion"))
    for sp in species_info.nonqssa_species_list:
        spec_idx = species_info.ordered_idx_map[sp]
        species = species_info.nonqssa_species[spec_idx]
        cw.writer(
            fstream,
            f"x[{spec_idx}] = y[{spec_idx}]/({species.weight:.16e}*YOW); ",
        )

    # call routine
    cw.writer(fstream, "helmholtz(aort, T);")

    # Equation 42 and 43
    cw.writer(fstream, cw.comment("Perform computation in Eq 44"))
    for sp in species_info.nonqssa_species_list:
        idx = species_info.ordered_idx_map[sp]
        cw.writer(
            fstream,
            "result +="
            f" x[{idx}]*(aort[{idx}]+log((x[{idx}]+{cc.smallnum:g}))+logPratio);",
        )

    cw.writer(fstream, cw.comment("Scale by RT/W"))
    cw.writer(fstream, "abms = result * RT * YOW;")

    cw.writer(fstream, "}")


def ckpx(fstream, mechanism, species_info):
    """Write ckpx."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Compute P = rhoRT/W(x)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKPX"
        + cc.sym
        + "(const amrex::Real rho, const amrex::Real T, const amrex::Real x[],"
        " amrex::Real& P)",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        "amrex::Real XW = 0;" + cw.comment(" To hold mean molecular wt"),
    )

    # molecular weights of all species
    for sp in species_info.nonqssa_species_list:
        spec_idx = species_info.ordered_idx_map[sp]
        species = species_info.nonqssa_species[spec_idx]
        cw.writer(
            fstream,
            f"XW += x[{spec_idx}]*{species.weight:.16e}; "
            + cw.comment(f"{species.name}"),
        )

    cw.writer(
        fstream,
        "P = rho *"
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e} * T"
        " / XW; " + cw.comment("P = rho*R*T/W"),
    )

    cw.writer(fstream)

    cw.writer(fstream, "}")


def ckpy(fstream, mechanism, species_info):
    """Write ckpy."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Compute P = rhoRT/W(y)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKPY"
        + cc.sym
        + "(const amrex::Real rho, const amrex::Real T, const amrex::Real"
        " y[], "
        " amrex::Real& P)",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real YOW = 0;" + cw.comment(" for computing mean MW"))

    # molecular weights of all species
    cw.writer(fstream)
    cw.writer(
        fstream,
        f"for (int i = 0; i < {len(species_info.nonqssa_species_list)}; i++)",
    )
    cw.writer(fstream, "{")
    cw.writer(fstream, "YOW += y[i]*imw(i);")
    cw.writer(fstream, "}")

    cw.comment("YOW holds the reciprocal of the mean molecular wt")
    cw.writer(
        fstream,
        "P = rho *"
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e} * T"
        " * YOW; " + cw.comment("P = rho*R*T/W"),
    )

    cw.writer(fstream)

    cw.writer(fstream, "}")


def ckpc(fstream, mechanism, species_info):
    """Write ckpc."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Compute P = rhoRT/W(c)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKPC"
        + cc.sym
        + "(const amrex::Real rho, const amrex::Real T, const amrex::Real"
        " c[], "
        " amrex::Real& P)",
    )

    cw.writer(fstream, "{")

    cw.writer(fstream, cw.comment("See Eq 5 in CK Manual"))
    cw.writer(fstream, "amrex::Real W = 0;")
    cw.writer(fstream, "amrex::Real sumC = 0;")

    # molecular weights of all species
    for sp in species_info.nonqssa_species_list:
        spec_idx = species_info.ordered_idx_map[sp]
        species = species_info.nonqssa_species[spec_idx]
        cw.writer(
            fstream,
            f"W += c[{spec_idx}]*{species.weight:.16e}; "
            + cw.comment(f"{species.name}"),
        )

    cw.writer(fstream)
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "sumC += c[id];")
    cw.writer(fstream, "}")

    cw.comment("W/sumC holds the mean molecular wt")
    cw.writer(
        fstream,
        "P = rho *"
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e} * T"
        " * sumC / W; " + cw.comment("P = rho*R*T/W"),
    )

    cw.writer(fstream)

    cw.writer(fstream, "}")


def ckrhox(fstream, mechanism, species_info):
    """Write ckrhox."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Compute rho = PW(x)/RT"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKRHOX"
        + cc.sym
        + "(const amrex::Real P, const amrex::Real T,"
        + "const amrex::Real x[], amrex::Real& rho)",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        "amrex::Real XW = 0;" + cw.comment(" To hold mean molecular wt"),
    )

    # molecular weights of all species
    for sp in species_info.nonqssa_species_list:
        spec_idx = species_info.ordered_idx_map[sp]
        species = species_info.nonqssa_species[spec_idx]
        cw.writer(
            fstream,
            f"XW += x[{spec_idx}]*{species.weight:.16e}; "
            + cw.comment(f"{species.name}"),
        )

    cw.writer(
        fstream,
        "rho = P * XW /"
        f" ({(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e} *"
        " T); " + cw.comment("rho = P*W/(R*T)"),
    )

    cw.writer(fstream)

    cw.writer(fstream, "}")


def ckrhoy(fstream, mechanism, species_info):
    """Write ckrhoy."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Compute rho = P*W(y)/RT"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKRHOY"
        + cc.sym
        + "(const amrex::Real P, const amrex::Real T,"
        + "const amrex::Real y[], amrex::Real& rho)",
    )
    cw.writer(fstream, "{")
    cw.writer(fstream, "amrex::Real YOW = 0;")
    cw.writer(fstream)
    cw.writer(fstream, f"for (int i = 0; i < {n_species}; i++)")
    cw.writer(fstream, "{")
    cw.writer(fstream, "YOW += y[i]*imw(i);")
    cw.writer(fstream, "}")
    cw.writer(fstream, "")
    cw.writer(
        fstream,
        "rho = P /"
        f" ({(cc.R * cc.ureg.mole * cc.ureg.kelvin / cc.ureg.erg).m:1.14e} * T"
        " * YOW);" + cw.comment("rho = P*W/(R*T)"),
    )

    cw.writer(fstream, "}")


def ckrhoc(fstream, mechanism, species_info):
    """Write ckrhoc."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Compute rho = P*W(c)/(R*T)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKRHOC"
        + cc.sym
        + "(const amrex::Real P, const amrex::Real T,"
        + "const amrex::Real c[], amrex::Real& rho)",
    )

    cw.writer(fstream, "{")

    cw.writer(fstream, cw.comment("See Eq 5 in CK Manual"))
    cw.writer(fstream, "amrex::Real W = 0;")
    cw.writer(fstream, "amrex::Real sumC = 0;")

    # molecular weights of all species
    for sp in species_info.nonqssa_species_list:
        spec_idx = species_info.ordered_idx_map[sp]
        species = species_info.nonqssa_species[spec_idx]
        cw.writer(
            fstream,
            f"W += c[{spec_idx}]*{species.weight:.16e}; "
            + cw.comment(f"{species.name}"),
        )

    cw.writer(fstream)
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "sumC += c[id];")
    cw.writer(fstream, "}")

    cw.comment("W/sumC holds the mean molecular wt")
    cw.writer(
        fstream,
        "rho = P * W / (sumC * T *"
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e}); "
        + cw.comment("rho = PW/(R*T)"),
    )

    cw.writer(fstream)

    cw.writer(fstream, "}")


def ckwt(fstream, mechanism, species_info):
    """Write ckwt."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("get molecular weight for all species"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKWT"
        + cc.sym
        + "( amrex::Real wt[])",
    )
    cw.writer(fstream, "{")
    # call molecularWeight
    cw.writer(fstream, "get_mw(wt);")
    cw.writer(fstream, "}")


def ckmmwy(fstream, mechanism, species_info):
    """Write ckmmwy."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("given y[species]: mass fractions"))
    cw.writer(fstream, cw.comment("s mean molecular weight (gm/mole)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKMMWY"
        + cc.sym
        + "(const amrex::Real y[], amrex::Real& wtm)",
    )
    cw.writer(fstream, "{")
    cw.writer(fstream, "amrex::Real YOW = 0;")
    cw.writer(fstream)
    cw.writer(fstream, f"for (int i = 0; i < {n_species}; i++)")
    cw.writer(fstream, "{")
    cw.writer(fstream, "YOW += y[i]*imw(i);")
    cw.writer(fstream, "}")
    cw.writer(fstream, "")
    cw.writer(fstream, "wtm = 1.0 / YOW;")

    cw.writer(fstream, "}")


def ckmmwx(fstream, mechanism, species_info):
    """Write ckmmwx."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("given x[species]: mole fractions"))
    cw.writer(fstream, cw.comment("returns mean molecular weight (gm/mole)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKMMWX"
        + cc.sym
        + "(const amrex::Real x[],  amrex::Real& wtm)",
    )
    cw.writer(fstream, "{")
    cw.writer(fstream, "amrex::Real XW = 0;" + cw.comment(" see Eq 4 in CK Manual"))
    # molecular weights of all species
    for sp in species_info.nonqssa_species_list:
        spec_idx = species_info.ordered_idx_map[sp]
        species = species_info.nonqssa_species[spec_idx]
        cw.writer(
            fstream,
            f"XW += x[{spec_idx}]*{species.weight:.16e}; "
            + cw.comment(f"{species.name}"),
        )
    cw.writer(fstream, "wtm = XW;")
    cw.writer(fstream)

    cw.writer(fstream, "}")


def ckmmwc(fstream, mechanism, species_info):
    """Write ckmmwc."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("given c[species]: molar concentration"))
    cw.writer(fstream, cw.comment("returns mean molecular weight (gm/mole)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKMMWC"
        + cc.sym
        + "(const amrex::Real c[],  amrex::Real& wtm)",
    )
    cw.writer(fstream, "{")
    cw.writer(fstream, cw.comment("See Eq 5 in CK Manual"))
    cw.writer(fstream, "amrex::Real W = 0;")
    cw.writer(fstream, "amrex::Real sumC = 0;")
    # molecular weights of all species
    for sp in species_info.nonqssa_species_list:
        spec_idx = species_info.ordered_idx_map[sp]
        species = species_info.nonqssa_species[spec_idx]
        cw.writer(
            fstream,
            f"W += c[{spec_idx}]*{species.weight:.16e}; "
            + cw.comment(f"{species.name}"),
        )
    cw.writer(fstream)
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "sumC += c[id];")
    cw.writer(fstream, "}")
    cw.writer(fstream, cw.comment(" CK provides no guard against division by zero"))
    cw.writer(fstream, "wtm = W/sumC;")
    cw.writer(fstream)

    cw.writer(fstream, "}")


def ckcpor(fstream, mechanism, species_info):
    """Write ckcpor."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("get Cp/R as a function of T "))
    cw.writer(fstream, cw.comment("for all species (Eq 19)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCPOR"
        + cc.sym
        + "(const amrex::Real T, amrex::Real cpor[])",
    )
    cw.writer(fstream, "{")

    # call routine
    cw.writer(fstream, "cp_R(cpor, T);")
    cw.writer(fstream, "}")


def ckhort(fstream, mechanism, species_info):
    """Write ckhort."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("get H/RT as a function of T "))
    cw.writer(fstream, cw.comment("for all species (Eq 20)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKHORT"
        + cc.sym
        + "(const amrex::Real T, amrex::Real hort[])",
    )
    cw.writer(fstream, "{")

    # call routine
    cw.writer(fstream, "speciesEnthalpy(hort, T);")
    cw.writer(fstream, "}")


def cksor(fstream, mechanism, species_info):
    """Write cksor."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("get S/R as a function of T "))
    cw.writer(fstream, cw.comment("for all species (Eq 21)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKSOR"
        + cc.sym
        + "(const amrex::Real T, amrex::Real sor[])",
    )
    cw.writer(fstream, "{")

    # call routine
    cw.writer(fstream, "speciesEntropy(sor, T);")
    cw.writer(fstream, "}")


def ckytx(fstream, mechanism, species_info):
    """Write ckytx."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("convert y[species] (mass fracs) to x[species] (mole fracs)"),
    )
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKYTX"
        + cc.sym
        + "(const amrex::Real y[], amrex::Real x[])",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real YOW = 0;")
    cw.writer(fstream)
    cw.writer(fstream, f"for (int i = 0; i < {n_species}; i++)")
    cw.writer(fstream, "{")
    cw.writer(fstream, "YOW += y[i]*imw(i);")
    cw.writer(fstream, "}")
    cw.writer(fstream, "")
    cw.writer(fstream, "amrex::Real YOWINV = 1.0/YOW;")
    cw.writer(fstream, "")
    cw.writer(fstream, f"for (int i = 0; i < {n_species}; i++)")
    cw.writer(fstream, "{")
    cw.writer(fstream, "x[i] = y[i]*imw(i)*YOWINV;")
    cw.writer(fstream, "}")

    cw.writer(fstream, "}")


def ckytcp(fstream, mechanism, species_info):
    """Write ckytcp."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("convert y[species] (mass fracs) to c[species] (molar conc)"),
    )
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKYTCP"
        + cc.sym
        + "(const amrex::Real P, const amrex::Real T,"
        + "const amrex::Real y[], amrex::Real c[])",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real YOW = 0;")
    cw.writer(fstream, "amrex::Real PWORT;")
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Compute inverse of mean molecular wt first"))
    cw.writer(fstream, f"for (int i = 0; i < {n_species}; i++)")
    cw.writer(fstream, "{")
    cw.writer(fstream, "c[i] = y[i]*imw(i);")
    cw.writer(fstream, "}")
    cw.writer(fstream, f"for (int i = 0; i < {n_species}; i++)")
    cw.writer(fstream, "{")
    cw.writer(fstream, "YOW += c[i];")
    cw.writer(fstream, "}")
    cw.writer(fstream, "")
    cw.writer(fstream, cw.comment("PW/RT (see Eq. 7)"))
    cw.writer(
        fstream,
        "PWORT = P/(YOW *"
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e} *"
        " T); ",
    )

    # now compute conversion
    cw.writer(fstream, cw.comment("Now compute conversion"))
    cw.writer(fstream, "")
    cw.writer(fstream, f"for (int i = 0; i < {n_species}; i++)")
    cw.writer(fstream, "{")
    cw.writer(fstream, "c[i] = PWORT * y[i] * imw(i);")
    cw.writer(fstream, "}")

    cw.writer(fstream, "}")


def ckytcr(fstream, mechanism, species_info):
    """Write ckytcr."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("convert y[species] (mass fracs) to c[species] (molar conc)"),
    )
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKYTCR"
        + cc.sym
        + "(const amrex::Real rho, amrex::Real /*T*/, const amrex::Real y[], "
        " amrex::Real c[])",
    )
    cw.writer(fstream, "{")
    cw.writer(fstream)
    cw.writer(fstream, f"for (int i = 0; i < {n_species}; i++)")
    cw.writer(fstream, "{")
    cw.writer(fstream, "c[i] = rho  * y[i] * imw(i);")
    cw.writer(fstream, "}")
    cw.writer(fstream, "}")


def ckxty(fstream, mechanism, species_info):
    """Write ckxty."""
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("convert x[species] (mole fracs) to y[species] (mass fracs)"),
    )
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKXTY"
        + cc.sym
        + "(const amrex::Real x[], amrex::Real y[])",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        "amrex::Real XW = 0; " + cw.comment("See Eq 4, 9 in CK Manual"),
    )

    # compute mean molecular weight first (eq 3)
    cw.writer(fstream, cw.comment("Compute mean molecular wt first"))
    for sp in species_info.nonqssa_species_list:
        spec_idx = species_info.ordered_idx_map[sp]
        species = species_info.nonqssa_species[spec_idx]
        cw.writer(
            fstream,
            f"XW += x[{spec_idx}]*{species.weight:.16e}; "
            + cw.comment(f"{species.name}"),
        )

    # now compute conversion
    cw.writer(fstream, cw.comment("Now compute conversion"))
    cw.writer(fstream, "amrex::Real XWinv = 1.0/XW;")
    for sp in species_info.nonqssa_species_list:
        spec_idx = species_info.ordered_idx_map[sp]
        species = species_info.nonqssa_species[spec_idx]
        cw.writer(
            fstream,
            f"y[{spec_idx}] = x[{spec_idx}]*{species.weight:.16e}*XWinv; ",
        )

    cw.writer(fstream)

    cw.writer(fstream, "}")


def ckxtcp(fstream, mechanism, species_info):
    """Write ckxtcp."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("convert x[species] (mole fracs) to c[species] (molar conc)"),
    )
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKXTCP"
        + cc.sym
        + "(const amrex::Real P, const amrex::Real T,"
        + "const amrex::Real x[], amrex::Real c[])",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        "amrex::Real PORT ="
        f" P/({(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e} *"
        " T); " + cw.comment("P/RT"),
    )
    # now compute conversion
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Compute conversion, see Eq 10"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "c[id] = x[id]*PORT;")
    cw.writer(fstream, "}")

    cw.writer(fstream)

    cw.writer(fstream, "}")


def ckxtcr(fstream, mechanism, species_info):
    """Write ckxtcr."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("convert x[species] (mole fracs) to c[species] (molar conc)"),
    )
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKXTCR"
        + cc.sym
        + "(const amrex::Real rho, const amrex::Real /*T*/, const amrex::Real"
        " x[], amrex::Real c[])",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        "amrex::Real XW = 0; " + cw.comment("See Eq 4, 11 in CK Manual"),
    )
    cw.writer(fstream, "amrex::Real ROW; ")

    # compute mean molecular weight first (eq 3)
    cw.writer(fstream, cw.comment("Compute mean molecular wt first"))
    for sp in range(n_species):
        species = species_info.nonqssa_species[sp]
        cw.writer(
            fstream,
            f"XW += x[{species.idx}]*{species.weight:.16e}; "
            + cw.comment(f"{species.name}"),
        )

    # now compute conversion
    cw.writer(fstream, "ROW = rho / XW;")
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Compute conversion, see Eq 11"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "c[id] = x[id]*ROW;")
    cw.writer(fstream, "}")

    cw.writer(fstream)

    cw.writer(fstream, "}")


def ckctx(fstream, mechanism, species_info):
    """Write ckctx."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("convert c[species] (molar conc) to x[species] (mole fracs)"),
    )
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCTX"
        + cc.sym
        + "(const amrex::Real c[], amrex::Real x[])",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real sumC = 0; ")

    cw.writer(fstream)
    cw.writer(fstream, cw.comment("compute sum of c "))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "sumC += c[id];")
    cw.writer(fstream, "}")

    # now compute conversion
    cw.writer(fstream)
    cw.writer(fstream, cw.comment(" See Eq 13 "))
    cw.writer(fstream, "amrex::Real sumCinv = 1.0/sumC;")
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "x[id] = c[id]*sumCinv;")
    cw.writer(fstream, "}")

    cw.writer(fstream)

    cw.writer(fstream, "}")


def ckcty(fstream, mechanism, species_info):
    """Write ckcty."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("convert c[species] (molar conc) to y[species] (mass fracs)"),
    )
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCTY"
        + cc.sym
        + "(const amrex::Real c[], amrex::Real y[])",
    )
    cw.writer(fstream, "{")

    cw.writer(fstream, "amrex::Real CW = 0; " + cw.comment("See Eq 12 in CK Manual"))

    # compute denominator in eq 12
    cw.writer(fstream, cw.comment("compute denominator in eq 12 first"))
    for sp in range(n_species):
        species = species_info.nonqssa_species[sp]
        cw.writer(
            fstream,
            f"CW += c[{species.idx}]*{species.weight:.16e}; "
            + cw.comment(f"{species.name}"),
        )

    # now compute conversion
    cw.writer(fstream, cw.comment("Now compute conversion"))
    cw.writer(fstream, "amrex::Real CWinv = 1.0/CW;")
    for sp in range(n_species):
        species = species_info.nonqssa_species[sp]
        cw.writer(
            fstream,
            f"y[{species.idx}] = c[{species.idx}]*{species.weight:.16e}*CWinv; ",
        )

    cw.writer(fstream)

    cw.writer(fstream, "}")


def ckcvml(fstream, mechanism, species_info):
    """Write ckcvml."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("get specific heat at constant volume as a function "),
    )
    cw.writer(fstream, cw.comment("of T for all species (molar units)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCVML"
        + cc.sym
        + "(const amrex::Real T, amrex::Real cvml[])",
    )
    cw.writer(fstream, "{")

    # call routine
    cw.writer(fstream, "cv_R(cvml, T);")

    # convert cv/R to cv
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("convert to chemkin units"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(
        fstream,
        f"cvml[id] *= {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e};",
    )
    cw.writer(fstream, "}")

    cw.writer(fstream, "}")


def ckcpml(fstream, mechanism, species_info):
    """Write ckcpml."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("get specific heat at constant pressure as a "))
    cw.writer(fstream, cw.comment("function of T for all species (molar units)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCPML"
        + cc.sym
        + "(const amrex::Real T, amrex::Real cpml[])",
    )
    cw.writer(fstream, "{")

    # call routine
    cw.writer(fstream, "cp_R(cpml, T);")

    # convert cp/R to cp
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("convert to chemkin units"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(
        fstream,
        f"cpml[id] *= {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e};",
    )
    cw.writer(fstream, "}")

    cw.writer(fstream, "}")


def ckuml(fstream, mechanism, species_info):
    """Write ckuml."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("get internal energy as a function "))
    cw.writer(fstream, cw.comment("of T for all species (molar units)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKUML"
        + cc.sym
        + "(const amrex::Real T, amrex::Real uml[])",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        "amrex::Real RT ="
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e}*T; "
        + cw.comment("R*T"),
    )

    # call routine
    cw.writer(fstream, "speciesInternalEnergy(uml, T);")

    # convert e/RT to e with molar units
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("convert to chemkin units"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "uml[id] *= RT;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "}")


def ckhml(fstream, mechanism, species_info):
    """Write ckhml."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("get enthalpy as a function "))
    cw.writer(fstream, cw.comment("of T for all species (molar units)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKHML"
        + cc.sym
        + "(const amrex::Real T, amrex::Real hml[])",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        "amrex::Real RT ="
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e}*T; "
        + cw.comment("R*T"),
    )

    # call routine
    cw.writer(fstream, "speciesEnthalpy(hml, T);")

    # convert h/RT to h with molar units
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("convert to chemkin units"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "hml[id] *= RT;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "}")


def ckgml(fstream, mechanism, species_info):
    """Write ckgml."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("get standard-state Gibbs energy as a function "))
    cw.writer(fstream, cw.comment("of T for all species (molar units)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKGML"
        + cc.sym
        + "(const amrex::Real T, amrex::Real gml[])",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        "amrex::Real RT ="
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e}*T; "
        + cw.comment("R*T"),
    )

    # call routine
    cw.writer(fstream, "gibbs(gml, T);")

    # convert g/RT to g with molar units
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("convert to chemkin units"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "gml[id] *= RT;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "}")


def ckaml(fstream, mechanism, species_info):
    """Write ckaml."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("get standard-state Helmholtz free energy as a "))
    cw.writer(fstream, cw.comment("function of T for all species (molar units)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKAML"
        + cc.sym
        + "(const amrex::Real T, amrex::Real aml[])",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        "amrex::Real RT ="
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e}*T; "
        + cw.comment("R*T"),
    )

    # call routine
    cw.writer(fstream, "helmholtz(aml, T);")

    # convert A/RT to A with molar units
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("convert to chemkin units"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "aml[id] *= RT;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "}")


def cksml(fstream, mechanism, species_info):
    """Write cksml."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("Returns the standard-state entropies in molar units"),
    )
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKSML"
        + cc.sym
        + "(const amrex::Real T, amrex::Real sml[])",
    )
    cw.writer(fstream, "{")

    # call routine
    cw.writer(fstream, "speciesEntropy(sml, T);")

    # convert s/R to s
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("convert to chemkin units"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(
        fstream,
        f"sml[id] *= {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e};",
    )
    cw.writer(fstream, "}")
    cw.writer(fstream, "}")


def ckcvms(fstream, mechanism, species_info):
    """Write ckcvms."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns the specific heats at constant volume"))
    cw.writer(fstream, cw.comment("in mass units (Eq. 29)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCVMS"
        + cc.sym
        + "(const amrex::Real T, amrex::Real cvms[])",
    )
    cw.writer(fstream, "{")

    # call routine
    cw.writer(fstream, "cv_R(cvms, T);")

    # convert cv/R to cv with mass units
    cw.writer(fstream, cw.comment("multiply by R/molecularweight"))
    for sp in species_info.nonqssa_species_list:
        spec_idx = species_info.ordered_idx_map[sp]
        species = species_info.nonqssa_species[spec_idx]
        row = (cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m / species.weight
        cw.writer(
            fstream,
            f"cvms[{spec_idx}] *= {row:20.15e}; " + cw.comment(f"{species.name}"),
        )

    cw.writer(fstream, "}")


def ckcpms(fstream, mechanism, species_info):
    """Write ckcpms."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns the specific heats at constant pressure"))
    cw.writer(fstream, cw.comment("in mass units (Eq. 26)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKCPMS"
        + cc.sym
        + "(const amrex::Real T, amrex::Real cpms[])",
    )
    cw.writer(fstream, "{")

    # call routine
    cw.writer(fstream, "cp_R(cpms, T);")

    # convert cp/R to cp with mass units
    cw.writer(fstream, cw.comment("multiply by R/molecularweight"))
    for sp in species_info.nonqssa_species_list:
        spec_idx = species_info.ordered_idx_map[sp]
        species = species_info.nonqssa_species[spec_idx]
        row = (cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m / species.weight
        cw.writer(
            fstream,
            f"cpms[{spec_idx}] *= {row:20.15e}; " + cw.comment(f"{species.name}"),
        )

    cw.writer(fstream, "}")


def ckums(fstream, mechanism, species_info):
    """Write ckums."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns internal energy in mass units (Eq 30.)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKUMS"
        + cc.sym
        + "(const amrex::Real T, amrex::Real ums[])",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        "amrex::Real RT ="
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e}*T; "
        + cw.comment("R*T"),
    )

    # call routine
    cw.writer(fstream)
    cw.writer(fstream, "speciesInternalEnergy(ums, T);")
    cw.writer(fstream)

    cw.writer(fstream, f"for (int i = 0; i < {n_species}; i++)")
    cw.writer(fstream, "{")
    cw.writer(fstream, "ums[i] *= RT*imw(i);")
    cw.writer(fstream, "}")
    cw.writer(fstream, "}")


def ckhms(fstream, mechanism, species_info):
    """Write ckhms."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns enthalpy in mass units (Eq 27.)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKHMS"
        + cc.sym
        + "(const amrex::Real T, amrex::Real hms[])",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        "amrex::Real RT ="
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e}*T; "
        + cw.comment("R*T"),
    )

    # call routine
    cw.writer(fstream)
    cw.writer(fstream, "speciesEnthalpy(hms, T);")
    cw.writer(fstream)

    cw.writer(fstream, f"for (int i = 0; i < {n_species}; i++)")
    cw.writer(fstream, "{")
    cw.writer(fstream, "hms[i] *= RT*imw(i);")
    cw.writer(fstream, "}")
    cw.writer(fstream, "}")


def ckgms(fstream, mechanism, species_info):
    """Write ckgms."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns gibbs in mass units (Eq 31.)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKGMS"
        + cc.sym
        + "(const amrex::Real T, amrex::Real gms[])",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        "amrex::Real RT ="
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e}*T; "
        + cw.comment("R*T"),
    )

    # call routine
    cw.writer(fstream)
    cw.writer(fstream, "gibbs(gms, T);")
    cw.writer(fstream)

    cw.writer(fstream, f"for (int i = 0; i < {n_species}; i++)")
    cw.writer(fstream, "{")
    cw.writer(fstream, "gms[i] *= RT*imw(i);")
    cw.writer(fstream, "}")
    cw.writer(fstream, "}")


def ckams(fstream, mechanism, species_info):
    """Write ckams."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns helmholtz in mass units (Eq 32.)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKAMS"
        + cc.sym
        + "(const amrex::Real T, amrex::Real ams[])",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        "amrex::Real RT ="
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e}*T; "
        + cw.comment("R*T"),
    )

    # call routine
    cw.writer(fstream)
    cw.writer(fstream, "helmholtz(ams, T);")

    cw.writer(fstream, f"for (int i = 0; i < {n_species}; i++)")
    cw.writer(fstream, "{")
    cw.writer(fstream, "ams[i] *= RT*imw(i);")
    cw.writer(fstream, "}")
    cw.writer(fstream, "}")


def cksms(fstream, mechanism, species_info):
    """Write cksms."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns the entropies in mass units (Eq 28.)"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKSMS"
        + cc.sym
        + "(const amrex::Real T, amrex::Real sms[])",
    )
    cw.writer(fstream, "{")

    # call routine
    cw.writer(fstream, "speciesEntropy(sms, T);")

    # convert s/R to s with mass units
    cw.writer(fstream, cw.comment("multiply by R/molecularweight"))
    for sp in species_info.nonqssa_species_list:
        spec_idx = species_info.ordered_idx_map[sp]
        species = species_info.nonqssa_species[spec_idx]
        row = (cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m / species.weight
        cw.writer(
            fstream,
            f"sms[{spec_idx}] *= {row:20.15e}; " + cw.comment(f"{species.name}"),
        )

    cw.writer(fstream, "}")


def ckwc(fstream, mechanism, species_info):
    """Write ckwc."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("compute the production rate for each species"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKWC"
        + cc.sym
        + "(const amrex::Real T, amrex::Real C[], amrex::Real wdot[])",
    )
    cw.writer(fstream, "{")

    # convert C to SI units
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("convert to SI"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "C[id] *= 1.0e6;")
    cw.writer(fstream, "}")

    # call productionRate
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("convert to chemkin units"))
    cw.writer(fstream, "productionRate(wdot, C, T);")

    # convert C and wdot to chemkin units
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("convert to chemkin units"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "C[id] *= 1.0e-6;")
    cw.writer(fstream, "wdot[id] *= 1.0e-6;")
    cw.writer(fstream, "}")

    cw.writer(fstream, "}")


def ckwyp(fstream, mechanism, species_info):
    """Write ckwyp."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns the molar production rate of species"))
    cw.writer(fstream, cw.comment("Given P, T, and mass fractions"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKWYP"
        + cc.sym
        + "(const amrex::Real P, const amrex::Real T,"
        + "const amrex::Real y[], amrex::Real wdot[])",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        f"amrex::Real c[{n_species}]; " + cw.comment("temporary storage"),
    )
    cw.writer(fstream, "amrex::Real YOW = 0; ")
    cw.writer(fstream, "amrex::Real PWORT; ")
    cw.writer(fstream)

    # compute inverse of mean molecular weight first (eq 3)
    cw.writer(fstream, cw.comment("Compute inverse of mean molecular wt first"))
    cw.writer(
        fstream,
        f"for (int i = 0; i < {len(species_info.nonqssa_species_list)}; i++)",
    )
    cw.writer(fstream, "{")
    cw.writer(fstream, "YOW += y[i]*imw(i);")
    cw.writer(fstream, "}")

    cw.writer(fstream, cw.comment("PW/RT (see Eq. 7)"))
    cw.writer(
        fstream,
        "PWORT = P/(YOW *"
        f" {(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e} *"
        " T); ",
    )

    cw.writer(fstream, cw.comment("multiply by 1e6 so c goes to SI"))
    cw.writer(fstream, "PWORT *= 1e6; ")

    # now compute conversion
    cw.writer(fstream, cw.comment("Now compute conversion (and go to SI)"))
    cw.writer(
        fstream,
        f"for (int i = 0; i < {len(species_info.nonqssa_species_list)}; i++)",
    )
    cw.writer(fstream, "{")
    cw.writer(fstream, "c[i] = PWORT * y[i]*imw(i);")
    cw.writer(fstream, "}")

    # call productionRate
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("convert to chemkin units"))
    cw.writer(fstream, "productionRate(wdot, c, T);")

    # convert wdot to chemkin units
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("convert to chemkin units"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "wdot[id] *= 1.0e-6;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "}")


def ckwxp(fstream, mechanism, species_info):
    """Write ckwxp."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns the molar production rate of species"))
    cw.writer(fstream, cw.comment("Given P, T, and mole fractions"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKWXP"
        + cc.sym
        + "(const amrex::Real P, const amrex::Real T,"
        + "const amrex::Real x[], amrex::Real wdot[])",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        f"amrex::Real c[{n_species}]; " + cw.comment("temporary storage"),
    )

    cw.writer(
        fstream,
        "amrex::Real PORT = 1e6 *"
        f" P/({(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e} *"
        " T); " + cw.comment("1e6 * P/RT so c goes to SI units"),
    )

    # now compute conversion
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Compute conversion, see Eq 10"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "c[id] = x[id]*PORT;")
    cw.writer(fstream, "}")

    # call productionRate
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("convert to chemkin units"))
    cw.writer(fstream, "productionRate(wdot, c, T);")

    # convert wdot to chemkin units
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("convert to chemkin units"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "wdot[id] *= 1.0e-6;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "}")


def ckwyr(fstream, mechanism, species_info):
    """Write ckwyr."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns the molar production rate of species"))
    cw.writer(fstream, cw.comment("Given rho, T, and mass fractions"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKWYR"
        + cc.sym
        + "(const amrex::Real rho, const amrex::Real T, const amrex::Real"
        " y[], "
        " amrex::Real wdot[])",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        f"amrex::Real c[{n_species}]; " + cw.comment("temporary storage"),
    )
    cw.writer(fstream)

    # now compute conversion
    cw.writer(fstream, cw.comment("See Eq 8 with an extra 1e6 so c goes to SI"))
    cw.writer(
        fstream,
        f"for (int i = 0; i < {len(species_info.nonqssa_species_list)}; i++)",
    )
    cw.writer(fstream, "{")
    cw.writer(fstream, "c[i] = 1e6 * rho * y[i]*imw(i);")
    cw.writer(fstream, "}")

    # call productionRate
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("call productionRate"))
    cw.writer(fstream, "productionRate(wdot, c, T);")

    # convert wdot to chemkin units
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("convert to chemkin units"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "wdot[id] *= 1.0e-6;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "}")


def ckwxr(fstream, mechanism, species_info):
    """Write ckwxr."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns the molar production rate of species"))
    cw.writer(fstream, cw.comment("Given rho, T, and mole fractions"))
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void CKWXR"
        + cc.sym
        + "(const amrex::Real rho, const amrex::Real T, const amrex::Real"
        " x[], "
        " amrex::Real wdot[])",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        f"amrex::Real c[{n_species}]; " + cw.comment("temporary storage"),
    )

    cw.writer(
        fstream,
        "amrex::Real XW = 0; " + cw.comment("See Eq 4, 11 in CK Manual"),
    )
    cw.writer(fstream, "amrex::Real ROW; ")

    # compute mean molecular weight first (eq 3)
    cw.writer(fstream, cw.comment("Compute mean molecular wt first"))
    for sp in range(n_species):
        species = species_info.nonqssa_species[sp]
        cw.writer(
            fstream,
            f"XW += x[{species.idx}]*{species.weight:.16e}; "
            + cw.comment(f"{species.name}"),
        )

    # now compute conversion
    cw.writer(fstream, cw.comment("Extra 1e6 factor to take c to SI"))
    cw.writer(fstream, "ROW = 1e6*rho / XW;")
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Compute conversion, see Eq 11"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "c[id] = x[id]*ROW;")
    cw.writer(fstream, "}")

    # call productionRate
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("convert to chemkin units"))
    cw.writer(fstream, "productionRate(wdot, c, T);")

    # convert wdot to chemkin units
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("convert to chemkin units"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "wdot[id] *= 1.0e-6;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "}")


def ckchrg(fstream, self):
    """Write the species unit charge number."""
    cw.writer(fstream)
    cw.writer(fstream, cw.comment(" species unit charge number "))
    cw.writer(fstream, "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void")
    cw.writer(fstream, "CKCHRG(int kcharge[])")
    cw.writer(fstream, "{")
    for i in range(0, self.species_info.n_species):
        species = self.species_info.nonqssa_species[i]
        text = f"kcharge[{i}] = {(int(species.charge)):d};"
        cw.writer(fstream, text + cw.comment(f"{species.name}"))
    cw.writer(fstream, "}")


def ckchrgmass(fstream, species_info):
    """Write the species charge per unit mass."""
    n_species = species_info.n_species
    cw.writer(fstream)
    cw.writer(fstream, cw.comment(" species charge per unit mass "))
    cw.writer(fstream, "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void")
    cw.writer(fstream, "CKCHRGMASS(amrex::Real zk[])")
    cw.writer(fstream, "{")
    cw.writer(fstream)
    cw.writer(fstream, f"int kchrg[{n_species}];")
    cw.writer(fstream, "CKCHRG(kchrg);")
    cw.writer(fstream)
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(
        fstream,
        f"zk[id] = {cc.Na:.8e} * {cc.qc:.8e} * kchrg[id] * imw(id);",
    )
    cw.writer(fstream, "}")
    cw.writer(fstream, "}")


def temp_given_ey(fstream, mechanism, species_list):
    """Write temperature given internal energy."""
    tmin_ar, tmax_ar = zip(
        *[
            (
                mechanism.species(symbol).thermo.min_temp,
                mechanism.species(symbol).thermo.max_temp,
            )
            for symbol in species_list
        ],
        strict=False,
    )
    tmax = min(tmax_ar)
    tmin = max(tmin_ar)

    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment(
            " get temperature given internal energy in mass units and mass fracs"
        ),
    )
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void"
        " GET_T_GIVEN_EY(const amrex::Real e, const amrex::Real y[],"
        " amrex::Real& t, int& ierr)",
    )
    cw.writer(fstream, "{")
    cw.writer(fstream, "#ifdef CONVERGENCE")
    cw.writer(fstream, "const int maxiter = 5000;")
    cw.writer(fstream, "const amrex::Real tol = 1.e-12;")
    cw.writer(fstream, "#else")
    cw.writer(fstream, "const int maxiter = 200;")
    cw.writer(fstream, "const amrex::Real tol = 1.e-6;")
    cw.writer(fstream, "#endif")
    cw.writer(
        fstream,
        f"amrex::Real tmin = {tmin};" + cw.comment("max lower bound for thermo def"),
    )

    cw.writer(
        fstream,
        f"amrex::Real tmax = {tmax};" + cw.comment("min upper bound for thermo def"),
    )
    cw.writer(fstream, "amrex::Real e1,emin,emax,cv,t1,dt;")
    cw.writer(fstream, "CKUBMS(tmin, y, emin);")
    cw.writer(fstream, "CKUBMS(tmax, y, emax);")
    cw.writer(fstream, "if (e < emin) {")
    cw.writer(fstream, cw.comment("Linear Extrapolation below tmin"))
    cw.writer(fstream, "CKCVBS(tmin, y, cv);")
    cw.writer(fstream, "t = tmin - (emin-e)/cv;")
    cw.writer(fstream, "ierr = 1;")
    cw.writer(fstream, "return;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "if (e > emax) {")
    cw.writer(fstream, cw.comment("Linear Extrapolation above tmax"))
    cw.writer(fstream, "CKCVBS(tmax, y, cv);")
    cw.writer(fstream, "t = tmax - (emax-e)/cv;")
    cw.writer(fstream, "ierr = 1;")
    cw.writer(fstream, "return;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "t1 = t;")
    cw.writer(fstream, "if (t1 < tmin || t1 > tmax) {")
    cw.writer(fstream, "t1 = tmin + (tmax-tmin)/(emax-emin)*(e-emin);")
    cw.writer(fstream, "}")
    cw.writer(fstream, "for (int i = 0; i < maxiter; ++i) {")
    cw.writer(fstream, "CKUBMS(t1,y,e1);")
    cw.writer(fstream, "CKCVBS(t1,y,cv);")
    cw.writer(fstream, "dt = (e - e1) / cv;")
    cw.writer(fstream, "if (dt > 100.) { dt = 100.; }")
    cw.writer(fstream, "else if (dt < -100.) { dt = -100.; }")
    cw.writer(fstream, "else if (fabs(dt) < tol) {break;}")
    cw.writer(fstream, "t1 += dt;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "t = t1;")
    cw.writer(fstream, "ierr = 0;")
    cw.writer(fstream, "}")
    cw.writer(fstream)


def temp_given_hy(fstream, mechanism, species_list):
    """Write temperature given enthalpy."""
    tmin_ar, tmax_ar = zip(
        *[
            (
                mechanism.species(symbol).thermo.min_temp,
                mechanism.species(symbol).thermo.max_temp,
            )
            for symbol in species_list
        ],
        strict=False,
    )
    tmax = min(tmax_ar)
    tmin = max(tmin_ar)

    cw.writer(
        fstream,
        cw.comment(" get temperature given enthalpy in mass units and mass fracs"),
    )
    cw.writer(
        fstream,
        "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void"
        " GET_T_GIVEN_HY(const amrex::Real h, const amrex::Real y[],"
        " amrex::Real& t, int& ierr)",
    )
    cw.writer(fstream, "{")
    cw.writer(fstream, "#ifdef CONVERGENCE")
    cw.writer(fstream, "const int maxiter = 5000;")
    cw.writer(fstream, "const amrex::Real tol = 1.e-12;")
    cw.writer(fstream, "#else")
    cw.writer(fstream, "const int maxiter = 200;")
    cw.writer(fstream, "const amrex::Real tol = 1.e-6;")
    cw.writer(fstream, "#endif")
    cw.writer(
        fstream,
        f"amrex::Real tmin = {tmin};" + cw.comment("max lower bound for thermo def"),
    )
    cw.writer(
        fstream,
        f"amrex::Real tmax = {tmax};" + cw.comment("min upper bound for thermo def"),
    )
    cw.writer(fstream, "amrex::Real h1,hmin,hmax,cp,t1,dt;")
    cw.writer(fstream, "CKHBMS(tmin, y, hmin);")
    cw.writer(fstream, "CKHBMS(tmax, y, hmax);")
    cw.writer(fstream, "if (h < hmin) {")
    cw.writer(fstream, cw.comment("Linear Extrapolation below tmin"))
    cw.writer(fstream, "CKCPBS(tmin, y, cp);")
    cw.writer(fstream, "t = tmin - (hmin-h)/cp;")
    cw.writer(fstream, "ierr = 1;")
    cw.writer(fstream, "return;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "if (h > hmax) {")
    cw.writer(fstream, cw.comment("Linear Extrapolation above tmax"))
    cw.writer(fstream, "CKCPBS(tmax, y, cp);")
    cw.writer(fstream, "t = tmax - (hmax-h)/cp;")
    cw.writer(fstream, "ierr = 1;")
    cw.writer(fstream, "return;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "t1 = t;")
    cw.writer(fstream, "if (t1 < tmin || t1 > tmax) {")
    cw.writer(fstream, "t1 = tmin + (tmax-tmin)/(hmax-hmin)*(h-hmin);")
    cw.writer(fstream, "}")
    cw.writer(fstream, "for (int i = 0; i < maxiter; ++i) {")
    cw.writer(fstream, "CKHBMS(t1,y,h1);")
    cw.writer(fstream, "CKCPBS(t1,y,cp);")
    cw.writer(fstream, "dt = (h - h1) / cp;")
    cw.writer(fstream, "if (dt > 100.) { dt = 100.; }")
    cw.writer(fstream, "else if (dt < -100.) { dt = -100.; }")
    cw.writer(fstream, "else if (fabs(dt) < tol) {break;}")
    cw.writer(fstream, "t1 += dt;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "t = t1;")
    cw.writer(fstream, "ierr = 0;")
    cw.writer(fstream, "}")


# NEED TO DEAL WITH THIS WHEN QSS
def ckinu(fstream, mechanism, species_info, reaction_info, write_sk=False):
    """Write ckinu/skinu."""
    n_reactions = mechanism.n_reactions
    n_gas_reactions = reaction_info.n_reactions
    phase = "surface" if write_sk else "gas"
    function_prefix = "S" if write_sk else "C"
    function_args = (
        "int* /*ki*/, int* /*nu*/" if n_reactions == 0 else "int ki[], int nu[]"
    )

    maxsp = 0

    ns = [0 for _ in range(n_reactions)]
    ki = [[] for _ in range(n_reactions)]
    nu = [[] for _ in range(n_reactions)]

    for orig_idx, _ in reaction_info.idxmap.items():
        # ignore heterogeneous reactions for CKINU and homogeneous reactions for SKINU
        if (phase == "gas" and orig_idx >= n_gas_reactions) or (
            phase == "surface" and orig_idx < n_gas_reactions
        ):
            continue
        # ensure orig_idx is in the range 0, NUM_SURFACE_REACTIONS for SKINU
        if phase == "surface":
            orig_idx -= n_gas_reactions

        reaction = mechanism.reaction(orig_idx)

        for symbol, coefficient in reaction.reactants.items():
            ki[orig_idx].append(species_info.ordered_idx_map[symbol])
            nu[orig_idx].append(-int(coefficient))
        for symbol, coefficient in reaction.products.items():
            ki[orig_idx].append(species_info.ordered_idx_map[symbol])
            nu[orig_idx].append(int(coefficient))

        maxsp = max(maxsp, len(ki[orig_idx]))

    for orig_idx, _ in reaction_info.idxmap.items():
        # ignore heterogeneous reactions for CKINU and homogeneous reactions for SKINU
        if (phase == "gas" and orig_idx >= n_gas_reactions) or (
            phase == "surface" and orig_idx < n_gas_reactions
        ):
            continue
        # ensure orig_idx is in the range 0, NUM_SURFACE_REACTIONS for SKINU
        if phase == "surface":
            orig_idx -= n_gas_reactions

        reaction = mechanism.reaction(orig_idx)

        ns[orig_idx] = len(ki[orig_idx])
        for _ in range(ns[orig_idx], maxsp):
            ki[orig_idx].append(0)
            nu[orig_idx].append(0)

    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment(
            f"Returns a count of {phase} species in a {phase} "
            "reaction, and their indices"
        ),
    )
    cw.writer(fstream, cw.comment("and stoichiometric coefficients. (Eq 50)"))
    cw.writer(
        fstream,
        f"void {function_prefix}KINU"
        + cc.sym
        + f"(const int i, int& nspec, {function_args})",
    )
    cw.writer(fstream, "{")

    if n_reactions > 0:
        str_ns = ",".join(str(x) for x in ns)
        cw.writer(
            fstream,
            f"const int ns[NUM_{phase.upper()}_REACTIONS] =\n     {{{str_ns:s}}};",
        )

        str_ki = ",".join(",".join(str(x) for x in ki[j]) for j in range(n_reactions))
        cw.writer(
            fstream,
            f"const int kiv[NUM_{phase.upper()}_REACTIONS*{maxsp}] =\n    "
            f" {{{str_ki:s}}};",
        )

        str_nu = ",".join(",".join(str(x) for x in nu[j]) for j in range(n_reactions))
        cw.writer(
            fstream,
            f"const int nuv[NUM_{phase.upper()}_REACTIONS*{maxsp}] =\n    "
            f" {{{str_nu:s}}};",
        )

    cw.writer(fstream, "if (i < 1) {")

    cw.writer(fstream, cw.comment("Return max num species per reaction"))
    cw.writer(fstream, f"nspec = {maxsp};")
    cw.writer(fstream, "} else {")

    if n_reactions == 0:
        cw.writer(fstream, "nspec = -1;")
    else:
        cw.writer(fstream, f"if (i > NUM_{phase.upper()}_REACTIONS) {{")
        cw.writer(fstream, "nspec = -1;")
        cw.writer(fstream, "} else {")

        cw.writer(fstream, "nspec = ns[i-1];")
        cw.writer(fstream, "for (int j=0; j<nspec; ++j) {")
        cw.writer(fstream, f"ki[j] = kiv[(i-1)*{maxsp} + j] + 1;")
        cw.writer(fstream, f"nu[j] = nuv[(i-1)*{maxsp} + j];")
        cw.writer(fstream, "}")
        cw.writer(fstream, "}")
    cw.writer(fstream, "}")
    cw.writer(fstream, "}")


def ckkfkr(fstream, mechanism, species_info):
    """Write ckkfkr."""
    n_species = species_info.n_species
    n_reactions = mechanism.n_reactions

    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Returns the progress rates of each reactions"))
    cw.writer(fstream, cw.comment("Given P, T, and mole fractions"))
    cw.writer(
        fstream,
        "void CKKFKR"
        + cc.sym
        + "(const amrex::Real P, const amrex::Real T, const amrex::Real x[]"
        + ", amrex::Real q_f[], amrex::Real q_r[])",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        f"amrex::Real c[{n_species}]; " + cw.comment("temporary storage"),
    )
    cw.writer(
        fstream,
        "amrex::Real PORT = 1e6 *"
        f" P/({(cc.R * cc.ureg.kelvin * cc.ureg.mole / cc.ureg.erg).m:1.14e} *"
        " T); " + cw.comment("1e6 * P/RT so c goes to SI units"),
    )

    # now compute conversion
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("Compute conversion, see Eq 10"))
    cw.writer(fstream, f"for (int id = 0; id < {n_species}; ++id) {{")
    cw.writer(fstream, "c[id] = x[id]*PORT;")
    cw.writer(fstream, "}")

    # call progressRateFR
    cw.writer(fstream)
    cw.writer(fstream, cw.comment("convert to chemkin units"))
    cw.writer(fstream, "progressRateFR(q_f, q_r, c, T);")

    # convert qdot to chemkin units
    cw.writer(fstream)
    if n_reactions > 0:
        cw.writer(fstream, cw.comment("convert to chemkin units"))
        cw.writer(fstream, f"for (int id = 0; id < {n_reactions}; ++id) {{")
        cw.writer(fstream, "q_f[id] *= 1.0e-6;")
        cw.writer(fstream, "q_r[id] *= 1.0e-6;")
        cw.writer(fstream, "}")

    cw.writer(fstream, "}")
