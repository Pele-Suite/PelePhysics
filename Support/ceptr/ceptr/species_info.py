"""Species information."""

from collections import OrderedDict

import pandas as pd

import ceptr.qssa_info as cqi


class SpeciesDb:
    """Species database."""

    def __init__(self, idx, ordered_id, name, mwt, chrg):
        self.mech_idx = idx
        self.idx = ordered_id
        self.name = name
        self.weight = mwt
        self.charge = chrg

    def __str__(self):
        """Print members of SpeciesDb."""
        return (
            f"""name = {self.name}, """
            + f"""mech_idx = {self.mech_idx}, """
            + f"""idx = {self.idx}, """
            + f"""weight = {self.weight}"""
            + f"""charge = {self.charge}"""
        )

    def __repr__(self):
        """Representation of SpeciesDb."""
        return f"""SpeciesDb({self.mech_idx}, {self.idx}, {self.name},
                             {self.weight}, {self.charge})"""


class SpeciesInfo:
    """Information on species."""

    def __init__(self):
        # Species
        # non QSSA
        # list of speciesDb for each non QSSA spec
        self.nonqssa_species = []
        # list of non QSSA species names
        self.nonqssa_species_list = []
        self.nonqssa_species_formatted_list = []
        # number of non QSSA species
        self.n_species = 0

        # list of surface species
        self.surface_species_list = list()

        # all Species
        self.all_species = []
        self.all_species_list = []
        self.all_species_formatted_list = []
        self.n_all_species = 0
        # Ordered dict for matching species to indices
        self.ordered_idx_map = OrderedDict()
        self.mech_idx_map = OrderedDict()

        self.low_temp = 0.0
        self.high_temp = 1000000.0

        # list of speciesDb for each QSSA spec
        self.qssa_species = []
        # list of QSSA species names
        self.qssa_species_list = []
        self.qssa_species_formatted_list = []
        # number of QSSA species
        self.n_qssa_species = 0
        self.qssa_info = cqi.QSSAInfo()

    def create_dicts(self):
        """Create species dicts for ease of use."""
        self.dict_species = {v: i for i, v in enumerate(self.all_species_list)}
        self.dict_qss_species = {v: i for i, v in enumerate(self.qssa_species_list)}
        self.dict_nonqss_species = {
            v: i for i, v in enumerate(self.nonqssa_species_list)
        }

    def identify_qss_dependencies(self, syms):
        """Identify QSS species dependencies from syms object."""
        self.dict_qssdepend_scqss = {}
        self.dict_qssdepend_sc = {}
        self.dict_qssdepend_gRTqss = {}
        self.dict_qssdepend_gRT = {}
        self.dict_qssdepend_kf = {}
        self.dict_qssdepend_kr = {}
        self.sc_qss_chain_stop = []
        for symbol in self.dict_qss_species:
            free_symb = sorted(
                syms.sc_qss_smp[self.dict_qss_species[symbol]].free_symbols, key=str
            )
            qss_symb = []
            sc_symb = []
            g_rt_qss_symb = []
            g_rt_symb = []
            kf_symb = []
            kr_symb = []
            for ss in free_symb:
                if "sc_qss" in str(ss):
                    qss_symb.append(ss)
                elif "sc" in str(ss):
                    sc_symb.append(ss)
                elif "g_RT_qss" in str(ss):
                    g_rt_qss_symb.append(ss)
                elif "g_RT" in str(ss):
                    g_rt_symb.append(ss)
                elif "kf" in str(ss):
                    kf_symb.append(ss)
                elif "kr" in str(ss):
                    kr_symb.append(ss)

            self.dict_qssdepend_scqss[symbol] = qss_symb
            self.dict_qssdepend_sc[symbol] = sc_symb
            self.dict_qssdepend_gRTqss[symbol] = g_rt_qss_symb
            self.dict_qssdepend_gRT[symbol] = g_rt_symb
            self.dict_qssdepend_kf[symbol] = kf_symb
            self.dict_qssdepend_kr[symbol] = kr_symb

            if not self.dict_qssdepend_scqss[symbol]:
                self.sc_qss_chain_stop.append(symbol)

    def identify_nonqss_dependencies(self, syms):
        """Identify nonQSS species dependencies from syms object."""
        self.dict_nonqssdepend_scqss = {}
        self.dict_nonqssdepend_sc = {}
        self.dict_nonqssdepend_gRTqss = {}
        self.dict_nonqssdepend_gRT = {}
        self.dict_nonqssdepend_kf = {}
        self.dict_nonqssdepend_kr = {}
        for symbol in self.dict_nonqss_species:
            free_symb = sorted(
                syms.sc_smp[self.dict_nonqss_species[symbol]].free_symbols, key=str
            )
            qss_symb = []
            sc_symb = []
            g_rt_qss_symb = []
            g_rt_symb = []
            kf_symb = []
            kr_symb = []
            for ss in free_symb:
                if "sc_qss" in str(ss):
                    qss_symb.append(ss)
                elif "sc" in str(ss):
                    sc_symb.append(ss)
                elif "g_RT_qss" in str(ss):
                    g_rt_qss_symb.append(ss)
                elif "g_RT" in str(ss):
                    g_rt_symb.append(ss)
                elif "kf" in str(ss):
                    kf_symb.append(ss)
                elif "kr" in str(ss):
                    kr_symb.append(ss)

            self.dict_nonqssdepend_scqss[symbol] = qss_symb
            self.dict_nonqssdepend_sc[symbol] = sc_symb
            self.dict_nonqssdepend_gRTqss[symbol] = g_rt_qss_symb
            self.dict_nonqssdepend_gRT[symbol] = g_rt_symb
            self.dict_nonqssdepend_kf[symbol] = kf_symb
            self.dict_nonqssdepend_kr[symbol] = kr_symb

    def identify_wdot_dependencies(self, syms):
        """Identify wdot dependencies from syms object."""
        self.dict_wdot_scqss = {}
        self.dict_wdot_sc = {}

        for symbol in self.dict_nonqss_species:
            symbolic_wdot = syms.wdot_smp[self.dict_nonqss_species[symbol]]
            if isinstance(symbolic_wdot, float):
                free_symb = []
            else:
                free_symb = sorted(symbolic_wdot.free_symbols, key=str)

            qss_symb = []
            sc_symb = []

            for ss in free_symb:
                if "sc_qss" in str(ss):
                    qss_symb.append(ss)
                elif "sc" in str(ss):
                    sc_symb.append(ss)

            self.dict_wdot_scqss[symbol] = qss_symb
            self.dict_wdot_sc[symbol] = sc_symb

    def make_scqss_dataframe(self):
        """Create lists for names, levels, and plus, mult."""
        name_list = []
        numb_list = []
        symb_list = []
        lev_list = []

        # Loop over stopping terms first
        for stp in self.sc_qss_chain_stop:
            scqssnum = self.dict_qss_species[stp]
            name_list.append(f"sc_qss[{scqssnum}]")
            numb_list.append(scqssnum)
            symb_list.append(stp)
            lev_list.append(0)

        # Initialize the dataframe
        scqss_df = pd.DataFrame(
            {
                "name": name_list,
                "number": numb_list,
                "symbol": symb_list,
                "level": lev_list,
            }
        )

        # Find the list of qss terms that have been filled
        qss_filled = scqss_df["name"].unique()

        # Loop until all qss terms have been filled
        level = 1
        while len(qss_filled) < len(self.qssa_species_list):
            # Find the sc_qss that only depend upon filled terms
            for symb in self.qssa_species_list:
                scqssnum = self.dict_qss_species[symb]
                if f"sc_qss[{scqssnum}]" in qss_filled:
                    continue
                else:
                    qss_depend = self.dict_qssdepend_scqss[symb]

                    # Determine if the qss species have been previously filled
                    f_vec = []
                    for qss_depend in self.dict_qssdepend_scqss[symb]:
                        f_vec.append(str(qss_depend) in qss_filled)

                    # Only depends upon previously filled entries
                    if all(f_vec):
                        name_list.append(f"sc_qss[{scqssnum}]")
                        numb_list.append(scqssnum)
                        symb_list.append(symb)
                        lev_list.append(level)

            scqss_df = pd.DataFrame(
                {
                    "name": name_list,
                    "number": numb_list,
                    "symbol": symb_list,
                    "level": lev_list,
                }
            )
            qss_filled = scqss_df["name"].unique()

            level += 1

        scqss_df["sc_dep"] = ""
        scqss_df["scqss_dep"] = ""
        # Add in a few more attributes for easy of use
        for idx, item in scqss_df.iterrows():
            scqss_df.at[idx, "sc_dep"] = self.dict_qssdepend_sc[item["symbol"]]
            scqss_df.at[idx, "scqss_dep"] = self.dict_qssdepend_scqss[item["symbol"]]

        # Return a deepcopy to self
        self.scqss_df = scqss_df.copy(deep=True)

    def make_sc_dataframe(self):
        """Make dataframe for sc species."""
        name_list = []
        numb_list = []
        symb_list = []

        for idx in range(self.n_species):
            name_list.append(f"sc[{idx}]")
            numb_list.append(idx)
            symb_list.append(self.nonqssa_species_list[idx])

        # Initialize the dataframe
        sc_df = pd.DataFrame(
            {
                "name": name_list,
                "number": numb_list,
                "symbol": symb_list,
            }
        )

        sc_df["scqss_rely"] = ""
        # Loop over the scqss_df and add in scqss dependence upon sc terms
        for sc_idx, sc in sc_df.iterrows():
            scqss_list = []
            for _, scqss in self.scqss_df.iterrows():
                if sc["name"] in str(scqss["sc_dep"]):
                    scqss_list.append(scqss["name"])

            sc_df.at[sc_idx, "scqss_rely"] = scqss_list

        # Return a deepcopy to self
        self.sc_df = sc_df.copy(deep=True)

    def make_wdot_dataframe(self):
        """Create lists for names, levels, and plus, mult."""
        name_list = []
        numb_list = []
        symb_list = []

        for idx in range(self.n_species):
            name_list.append(f"wdot[{idx}]")
            numb_list.append(idx)
            symb_list.append(self.nonqssa_species_list[idx])

        # Initialize the dataframe
        wdot_df = pd.DataFrame(
            {
                "name": name_list,
                "number": numb_list,
                "symbol": symb_list,
            }
        )

        wdot_df["sc_dep"] = ""
        wdot_df["scqss_dep"] = ""
        # Add in a few more attributes for easy of use
        for idx, item in wdot_df.iterrows():
            wdot_df.at[idx, "sc_dep"] = self.dict_wdot_sc[item["symbol"]]
            wdot_df.at[idx, "scqss_dep"] = self.dict_wdot_scqss[item["symbol"]]

        # Return a deepcopy to self
        self.wdot_df = wdot_df.copy(deep=True)

    def set_low_high_temperatures(self, mechanism):
        """Set low and high temperatures."""
        self.low_temp = max(
            mechanism.species(s).thermo.min_temp for s in self.nonqssa_species_list
        )
        self.high_temp = min(
            mechanism.species(s).thermo.max_temp for s in self.nonqssa_species_list
        )
