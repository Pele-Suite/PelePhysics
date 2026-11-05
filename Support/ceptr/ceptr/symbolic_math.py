"""Symbolic math for symbolic differentiation."""

import re
from collections import OrderedDict

import pandas as pd
import symengine as sme
import sympy as smp

import ceptr.inputs as ci
import ceptr.thermo as cth


class SymbolicMath:
    """Symbols to carry throughout operations."""

    def __init__(
        self,
        species_info,
        reaction_info,
        mechanism,
        format_input,
    ):
        # Formatting options
        params = ci.Input()
        if format_input is not None:
            params.from_toml(format_input)

        self.hformat = params.inputs["Readability"]["hformat"].value

        self.remove_1 = params.inputs["Arithmetic"]["remove_1"].value
        self.remove_pow = params.inputs["Arithmetic"]["remove_pow"].value
        self.remove_pow10 = params.inputs["Arithmetic"]["remove_pow10"].value

        self.min_op_count = params.inputs["Replacement"]["min_op_count"].value
        self.min_op_count_all = params.inputs["Replacement"]["min_op_count_all"].value
        self.gradual_op_count = params.inputs["Replacement"]["gradual_op_count"].value
        self.remove_single_symbols_cse = params.inputs["Replacement"][
            "remove_single_symbols_cse"
        ].value

        self.recycle_cse = params.inputs["Recycle"]["recycle_cse"].value
        self.store_in_jacobian = params.inputs["Recycle"]["store_in_jacobian"].value
        if 2 * reaction_info.n_qssa_reactions > (species_info.n_species + 1) ** 2:
            self.store_in_jacobian = False

        self.round_decimals = params.inputs["Characters"]["round_decimals"].value

        # Set to False to use bottom up approach
        self.top_bottom = True

        n_species = species_info.n_species
        n_qssa_species = species_info.n_qssa_species

        self.T_smp = sme.symbols("T")
        self.invT_smp = sme.symbols("invT")
        self.invT2_smp = self.invT_smp * self.invT_smp
        self.logT_smp = sme.symbols("logT")
        self.T2_smp = sme.symbols("T2")
        self.T3_smp = sme.symbols("T3")
        self.T4_smp = sme.symbols("T4")

        # Keep refC and refCinv as symbols to avoid doing divisions all the time
        # coeff1 = cc.Patm_pa
        # coeff2 = cc.R.to(cc.ureg.joule / (cc.ureg.mole / cc.ureg.kelvin)).m
        # if self.remove_1:
        #    self.refC_smp = float(coeff1 / coeff2) * self.invT_smp
        # else:
        #    self.refC_smp = coeff1 / coeff2 * self.invT_smp
        # if self.remove_1:
        #    self.refCinv_smp = float(1.0) / self.refC_smp
        # else:
        #    self.refCinv_smp = 1.0 / self.refC_smp
        self.refCinv_smp = sme.symbols("refCinv")
        self.refC_smp = sme.symbols("refC")

        self.sc_smp = [sme.symbols("sc[" + str(i) + "]") for i in range(n_species)]
        self.g_RT_smp = [sme.symbols("g_RT[" + str(i) + "]") for i in range(n_species)]
        self.h_RT_smp = [sme.symbols("h_RT[" + str(i) + "]") for i in range(n_species)]
        self.wdot_smp = [sme.symbols("wdot[" + str(i) + "]") for i in range(n_species)]
        self.jac_smp = [
            sme.symbols(f"J[{str(i)}][{str(j)}]")
            for i in range(n_species)
            for j in range(n_species + 1)
        ]

        models = cth.analyze_thermodynamics(
            mechanism, species_info.nonqssa_species_list
        )
        self.models_smp_tmp = [
            {
                "species": x["species"],
                "interval": x["interval"],
                "gibbs": [None] * len(x["coefficients"]),
                "gibbs_qss": [None] * len(x["coefficients"]),
                "speciesEnthalpy": [None] * len(x["coefficients"]),
                "speciesEnthalpy_qss": [None] * len(x["coefficients"]),
            }
            for x in models
        ]

        # mixture (useful for third body reactions)
        self.mixture_smp = 0.0
        for i in range(n_species):
            self.mixture_smp += self.sc_smp[i]

        # for storing intermediate terms used in definition of sc_qss
        self.intermediate_helpers_smp = OrderedDict()
        self.intermediate_terms_smp = {}

        if n_qssa_species > 0:
            n_qssa_reactions = reaction_info.n_qssa_reactions

            self.sc_qss_smp = [
                sme.symbols("sc_qss[" + str(i) + "]") for i in range(n_qssa_species)
            ]
            self.kf_qss_smp = [
                sme.symbols("kf_qss[" + str(i) + "]") for i in range(n_qssa_reactions)
            ]
            self.qf_qss_smp = [
                sme.symbols("qf_qss[" + str(i) + "]") for i in range(n_qssa_reactions)
            ]
            self.qr_qss_smp = [
                sme.symbols("qr_qss[" + str(i) + "]") for i in range(n_qssa_reactions)
            ]
            self.g_RT_qss_smp = [
                sme.symbols("g_RT_qss[" + str(i) + "]") for i in range(n_qssa_species)
            ]
            self.h_RT_qss_smp = [
                sme.symbols("h_RT_qss[" + str(i) + "]") for i in range(n_qssa_species)
            ]

            models = cth.analyze_thermodynamics(
                mechanism, species_info.qssa_species_list
            )
            self.models_qss_smp_tmp = [
                {
                    "species": x["species"],
                    "interval": x["interval"],
                    "gibbs": [None] * len(x["coefficients"]),
                    "gibbs_qss": [None] * len(x["coefficients"]),
                    "speciesEnthalpy": [None] * len(x["coefficients"]),
                    "speciesEnthalpy_qss": [None] * len(x["coefficients"]),
                }
                for x in models
            ]

            self.kf_qss_smp_tmp = [
                sme.symbols("kf_qss[" + str(i) + "]") for i in range(n_qssa_reactions)
            ]

            # Create dict to hold end of chain rule dscqssdsc terms
            self.dscqssdsc_stop = {"info": ""}
            # Create dict to hold intermediate chain rule dscqssdsc terms
            self.dscqssdsc_interm = {}
            # Create dict to hold dscqss_dsc terms
            self.dscqssdsc = {}
            # Create dict to hold dscqss_dscqss terms
            self.dscqssdscqss = {}
            # Create dict to hold dwdot_dsc terms
            self.dwdotdsc = {}
            # Create dict to hold dwdot_dscqss terms
            self.dwdotdscqss = {}
            # Create dict to hold jacobian terms
            self.jacobian = {}

            self.dscqssdscqss_slow = {}
            self.dscqssdsc_slow = {}

    def round_in_string(self, string, maxdec=6):
        """Round decimal numbers if possible."""
        list_decimal = re.findall(r"\d+\.\d+", string)
        list_num = [float(s) for s in list_decimal]
        idec = 0
        for inum, num in enumerate(list_num):
            while True:
                rounded = round(float(num), idec)
                if abs(rounded - float(num)) < 1e-12:
                    string = string.replace(list_decimal[inum], str(rounded))
                    break
                else:
                    idec += 1
                if idec > maxdec:
                    break
        return string

    def convert_to_cpp(self, sym_smp):
        """Convert sympy object to C code compatible string.

        Also apply some formatting.
        """
        user_functions = {}

        if self.remove_pow or self.remove_pow10:
            user_functions["Pow"] = []

        if self.remove_pow:
            # Positive exponents
            user_functions["Pow"].append(
                (
                    lambda b, e: (e.is_Integer or e.is_Float)
                    and (
                        abs(e - 1) < 1e-16 or abs(e - 2) < 1e-16 or abs(e - 3) < 1e-16
                    ),
                    lambda b, e: "(" + "*".join(["(" + b + ")"] * int(float(e))) + ")",
                )
            )
            # Negative exponents
            user_functions["Pow"].append(
                (
                    lambda b, e: (e.is_Integer or e.is_Float)
                    and (
                        abs(e + 1) < 1e-16 or abs(e + 2) < 1e-16 or abs(e + 3) < 1e-16
                    ),
                    lambda b, e: "("
                    + "1.0/"
                    + "("
                    + "*".join(["(" + b + ")"] * int(-float(e)))
                    + ")"
                    + ")",
                )
            )

        if self.remove_pow10:
            user_functions["Pow"].append(
                # (
                #    lambda b, e: (b.is_Integer or b.is_Float) and (abs(b-10)<1e-16),
                #    lambda b, e: ""
                #    + "exp10("+e+")"
                #    + "",
                # ),
                (
                    lambda b, e: (b.is_Integer or b.is_Float) and (abs(b - 10) < 1e-16),
                    lambda b, e: "" + "exp(M_LN10 * (" + e + "))" + "",
                )
            )

        if self.remove_pow or self.remove_pow10:
            user_functions["Pow"].append((lambda b, e: "pow"))
            cppcode = smp.ccode(sym_smp, user_functions=user_functions)

        else:
            cppcode = sme.ccode(sym_smp)

        cpp_str = str(cppcode)

        if self.remove_1:
            cpp_str = cpp_str.replace("(1.0*", "(")
            cpp_str = cpp_str.replace(" 1.0*", " ")
            cpp_str = cpp_str.replace("+1.0*", "+")
            cpp_str = cpp_str.replace("-1.0*", "-")
            cpp_str = cpp_str.replace("/1.0+", "+")
            cpp_str = cpp_str.replace("/1.0*", "*")
            cpp_str = cpp_str.replace("/1.0-", "-")
            cpp_str = cpp_str.replace("*1.0+", "+")
            cpp_str = cpp_str.replace("*1.0*", "*")
            cpp_str = cpp_str.replace("*1.0-", "-")
            if cpp_str.startswith("1.0*"):
                cpp_str = cpp_str[4:]

        if self.round_decimals:
            cpp_str = self.round_in_string(cpp_str)

        return cpp_str

    def syms_to_specnum(self, sym_smp):
        """Extract number from syms string."""
        num = re.findall(r"\[(.*?)\]", str(sym_smp))
        return int(num[0])

    def convert_number_to_int(self, number):
        """Convert number to int if possible."""
        factor = float(number)
        if self.remove_1 and abs(factor) < 1.1 and abs(factor - int(factor)) < 1e-16:
            factor = int(factor)
        return factor

    def convert_symb_to_int(self, symb):
        """Convert symbol to int if possible."""
        try:
            number = float(symb)
            number = self.convert_number_to_int(number)
            return number
        except RuntimeError:
            return symb

    def reduce_expr(self, orig):
        """Reduce expression interface."""
        n_cse = len(orig[0])
        n_exp = len(orig[1])
        common_expr_lhs = [orig[0][i][0] for i in range(n_cse)]
        common_expr_rhs = [orig[0][i][1] for i in range(n_cse)]
        final_expr = [orig[1][i] for i in range(n_exp)]
        to_replace = []
        replace_with = []
        print(
            f"Starting expression reduction from {n_cse} expressions",
            end="...\n",
        )
        if self.min_op_count_all > 0:
            if self.gradual_op_count:
                for count_lim in range(1, self.min_op_count_all + 1):
                    print(
                        f"\tStarting min op count ALL, count_lim={count_lim}",
                        end="...",
                        flush=True,
                    )
                    (
                        common_expr_lhs,
                        common_expr_rhs,
                        final_expr,
                    ) = self.reduce_expr_top_bottom_rec_count(
                        orig,
                        count_lim,
                        common_expr_lhs,
                        common_expr_rhs,
                        final_expr,
                    )
                    print("Done!", flush=True)
            else:
                count_lim = self.min_op_count_all
                print(
                    f"\tStarting min op count ALL, count_lim={count_lim}",
                    end="...",
                    flush=True,
                )
                (
                    common_expr_lhs,
                    common_expr_rhs,
                    final_expr,
                ) = self.reduce_expr_top_bottom_rec_count(
                    orig,
                    count_lim,
                    common_expr_lhs,
                    common_expr_rhs,
                    final_expr,
                )
                print("Done!", flush=True)

        if self.min_op_count > 0:
            if self.gradual_op_count:
                for count_lim in range(1, self.min_op_count + 1):
                    print(
                        f"\tStarting min op count, count_lim={count_lim}",
                        end="...",
                        flush=True,
                    )
                    if self.top_bottom:
                        (
                            common_expr_lhs,
                            common_expr_rhs,
                            final_expr,
                        ) = self.reduce_expr_top_bottom(
                            orig,
                            count_lim,
                            common_expr_lhs,
                            common_expr_rhs,
                            final_expr,
                        )
                    else:
                        (
                            common_expr_lhs,
                            common_expr_rhs,
                            final_expr,
                        ) = self.reduce_expr_bottom_up(
                            orig,
                            count_lim,
                            common_expr_lhs,
                            common_expr_rhs,
                            final_expr,
                        )
                    print("Done!", flush=True)
            else:
                count_lim = self.min_op_count
                print(
                    f"\tStarting min op count, count_lim={count_lim}",
                    end="...",
                    flush=True,
                )
                if self.top_bottom:
                    (
                        common_expr_lhs,
                        common_expr_rhs,
                        final_expr,
                    ) = self.reduce_expr_top_bottom(
                        orig,
                        count_lim,
                        common_expr_lhs,
                        common_expr_rhs,
                        final_expr,
                    )
                else:
                    (
                        common_expr_lhs,
                        common_expr_rhs,
                        final_expr,
                    ) = self.reduce_expr_bottom_up(
                        orig,
                        count_lim,
                        common_expr_lhs,
                        common_expr_rhs,
                        final_expr,
                    )
                print("Done!", flush=True)

        if self.remove_single_symbols_cse:
            print("\tStarting single symbol removal", end="...", flush=True)
            (
                common_expr_lhs,
                common_expr_rhs,
                final_expr,
            ) = self.remove_single_symbol(
                orig,
                common_expr_lhs,
                common_expr_rhs,
                final_expr,
            )
            print("Done!", flush=True)

        if self.recycle_cse:
            print("\tStarting cse recycling", end="...", flush=True)
            (
                common_expr_lhs,
                common_expr_rhs,
                final_expr,
                to_replace,
                replace_with,
            ) = self.recycle_cse_post(
                orig, common_expr_lhs, common_expr_rhs, final_expr
            )
            print("Done!", flush=True)
        print("Done!", flush=True)

        return (
            common_expr_lhs,
            common_expr_rhs,
            final_expr,
            to_replace,
            replace_with,
        )

    def remove_single_symbol(
        self,
        orig,
        common_expr_lhs,
        common_expr_rhs,
        final_expr,
    ):
        """
        Remove cses made of single symbols.

        Those are typically of the for "-xi" where the operation may disappear
        after substitution.
        """
        replacements = []
        common_expr_symbols = [rhs.free_symbols for rhs in common_expr_rhs]
        final_expr_symbols = [expr.free_symbols for expr in final_expr]

        # Replacement loop
        for i, (lhs, rhs) in enumerate(zip(common_expr_lhs, common_expr_rhs)):
            op_count = sme.count_ops(rhs)
            is_float = True
            is_single_symbol = True
            try:
                number = float(rhs)
                rhs = number
            except RuntimeError:
                is_float = False
            if not is_float:
                if op_count > 1 or len(rhs.free_symbols) > 1:
                    is_single_symbol = False
            if is_float or is_single_symbol:
                replacements.append(i)
                ind = [j + i for j, s in enumerate(common_expr_symbols[i:]) if lhs in s]
                for j in ind:
                    common_expr_rhs[j] = common_expr_rhs[j].subs(lhs, rhs)
                    # common_expr_symbols[j].remove(lhs)
                ind = [j for j, s in enumerate(final_expr_symbols) if lhs in s]
                for j in ind:
                    final_expr[j] = final_expr[j].subs(lhs, rhs)
                    # final_expr_symbols[j].remove(lhs)

        replacements.reverse()
        for rep in replacements:
            del common_expr_lhs[rep]
            del common_expr_rhs[rep]

        print(f"Remaining expressions = {len(common_expr_lhs)}", end="...")

        return common_expr_lhs, common_expr_rhs, final_expr

    def reduce_expr_top_bottom_rec_count(
        self,
        orig,
        count_lim,
        common_expr_lhs,
        common_expr_rhs,
        final_expr,
    ):
        """
        Reduce number of expressions in cse list.

        Top bottom loop over common and final expressions and remove the ones that have
        a number of operation < count_lim including operations of variables that use it
        """
        replacements = []
        common_expr_symbols = [rhs.free_symbols for rhs in common_expr_rhs]
        final_expr_symbols = [expr.free_symbols for expr in final_expr]

        # Replacement loop
        for i, (lhs, rhs) in enumerate(zip(common_expr_lhs, common_expr_rhs)):
            op_count = sme.count_ops(rhs)
            # count how many times the expression is used later
            ind_rhs = [j + i for j, s in enumerate(common_expr_symbols[i:]) if lhs in s]
            ind_final_expr = [j for j, s in enumerate(final_expr_symbols) if lhs in s]
            rec_count = 0
            for ind in ind_rhs:
                rec_count += smp.sympify(common_expr_rhs[ind]).count(lhs)
            # rec_count_cse = rec_count
            for ind in ind_final_expr:
                rec_count += smp.sympify(final_expr[ind]).count(lhs)

            total_op = (rec_count - 1) * op_count
            is_float = True
            try:
                number = float(rhs)
                rhs = number
            except RuntimeError:
                is_float = False
            if total_op < count_lim or is_float:
                replacements.append(i)
                ind = [j + i for j, s in enumerate(common_expr_symbols[i:]) if lhs in s]
                for j in ind:
                    common_expr_rhs[j] = common_expr_rhs[j].subs(lhs, rhs)
                    common_expr_symbols[j].remove(lhs)
                ind = [j for j, s in enumerate(final_expr_symbols) if lhs in s]
                for j in ind:
                    final_expr[j] = final_expr[j].subs(lhs, rhs)
                    final_expr_symbols[j].remove(lhs)

        replacements.reverse()
        for rep in replacements:
            del common_expr_lhs[rep]
            del common_expr_rhs[rep]

        print(f"Remaining expressions = {len(common_expr_lhs)}", end="...")

        return common_expr_lhs, common_expr_rhs, final_expr

    def reduce_expr_top_bottom(
        self,
        orig,
        count_lim,
        common_expr_lhs,
        common_expr_rhs,
        final_expr,
    ):
        """
        Reduce number of expressions in cse list.

        Top bottom loop over common and final expressions and remove the ones that have
        a number of operation < count_lim
        """
        replacements = []
        common_expr_symbols = [rhs.free_symbols for rhs in common_expr_rhs]
        final_expr_symbols = [expr.free_symbols for expr in final_expr]

        # Replacement loop
        for i, (lhs, rhs) in enumerate(zip(common_expr_lhs, common_expr_rhs)):
            op_count = sme.count_ops(rhs)
            is_float = True
            try:
                number = float(rhs)
                rhs = number
            except RuntimeError:
                is_float = False
            if op_count < count_lim or is_float:
                replacements.append(i)
                ind = [j + i for j, s in enumerate(common_expr_symbols[i:]) if lhs in s]
                for j in ind:
                    common_expr_rhs[j] = common_expr_rhs[j].subs(lhs, rhs)
                    common_expr_symbols[j].remove(lhs)
                ind = [j for j, s in enumerate(final_expr_symbols) if lhs in s]
                for j in ind:
                    final_expr[j] = final_expr[j].subs(lhs, rhs)
                    final_expr_symbols[j].remove(lhs)

        replacements.reverse()
        for rep in replacements:
            del common_expr_lhs[rep]
            del common_expr_rhs[rep]

        print(f"Remaining expressions = {len(common_expr_lhs)}", end="...")

        return common_expr_lhs, common_expr_rhs, final_expr

    def reduce_expr_bottom_up(
        self,
        orig,
        count_lim,
        common_expr_lhs,
        common_expr_rhs,
        final_expr,
    ):
        """
        Reduce number of expressions in cse list.

        Bottom up loop over common and final expressions and remove the ones that have
        a number of operation < count_lim
        """
        replacements = []
        common_expr_symbols = [rhs.free_symbols for rhs in common_expr_rhs]
        final_expr_symbols = [expr.free_symbols for expr in final_expr]

        # Replacement loop
        for i, (lhs, rhs) in reversed(
            list(enumerate(zip(common_expr_lhs, common_expr_rhs)))
        ):
            op_count = sme.count_ops(rhs)
            is_float = True
            try:
                number = float(rhs)
                rhs = number
            except RuntimeError:
                is_float = False
            if op_count < count_lim or is_float:
                replacements.append(i)
                ind = [j + i for j, s in enumerate(common_expr_symbols[i:]) if lhs in s]
                for j in ind:
                    common_expr_rhs[j] = common_expr_rhs[j].subs(lhs, rhs)
                    common_expr_symbols[j].remove(lhs)
                    try:
                        common_expr_symbols[j].update(rhs.free_symbols)
                    except AttributeError:
                        pass
                ind = [j for j, s in enumerate(final_expr_symbols) if lhs in s]
                for j in ind:
                    final_expr[j] = final_expr[j].subs(lhs, rhs)
                    final_expr_symbols[j].remove(lhs)
                    try:
                        final_expr_symbols[j].update(rhs.free_symbols)
                    except AttributeError:
                        pass

        for rep in replacements:
            del common_expr_lhs[rep]
            del common_expr_rhs[rep]

        print(f"Remaining expressions = {len(common_expr_lhs)}", end="...")

        return common_expr_lhs, common_expr_rhs, final_expr

    def recycle_cse_post(
        self,
        orig,
        common_expr_lhs,
        common_expr_rhs,
        final_expr,
    ):
        """Recycle cse that are not used later."""
        to_replace = []
        replace_with = []

        n_cse = len(common_expr_lhs)
        common_expr_symbols = [rhs.free_symbols for rhs in common_expr_rhs]
        final_expr_symbols = [expr.free_symbols for expr in final_expr]

        # Figure out which symbols may be recycled
        for isymb, symb in enumerate(common_expr_lhs):
            ind_final = [j for j, s in enumerate(final_expr_symbols) if symb in s]
            if not ind_final:
                ind_cse = [
                    j + isymb
                    for j, s in enumerate(common_expr_symbols[isymb:])
                    if symb in s
                ]
                if ind_cse:
                    # This is the symbol we would like to replace
                    target_replace = ind_cse[-1]
                    # Make sure that we haven't replaced it already
                    while (
                        target_replace < n_cse
                        and common_expr_lhs[target_replace] in to_replace
                    ):
                        target_replace += 1
                    if target_replace < n_cse:
                        to_replace.append(common_expr_lhs[target_replace])
                        # If the symbol we want to replace with is already replaced,
                        # Make sure we are consistent
                        if symb in to_replace:
                            ind = to_replace.index(symb)
                            replace_with.append(replace_with[ind])
                        else:
                            replace_with.append(symb)
        # Use the recycling list to actually recycle
        for isr, symb_replace in enumerate(to_replace):
            ind_rhs = [
                j for j, s in enumerate(common_expr_symbols) if symb_replace in s
            ]
            for ind in ind_rhs:
                common_expr_rhs[ind] = common_expr_rhs[ind].subs(
                    symb_replace, sme.symbols(replace_with[isr].name)
                )
            ind_exp = [j for j, s in enumerate(final_expr_symbols) if symb_replace in s]
            for ind in ind_exp:
                final_expr[ind] = final_expr[ind].subs(
                    symb_replace, sme.symbols(replace_with[isr].name)
                )

        print(f"Remaining expressions = {n_cse - len(to_replace)}", end="...")

        return (
            common_expr_lhs,
            common_expr_rhs,
            final_expr,
            to_replace,
            replace_with,
        )

    def write_array_to_cpp(self, list_smp, array_str, cw, fstream, index_list=None):
        """Convert sympy array to C code compatible string."""
        n = len(list_smp)

        # Write common expressions
        print("Start making common subexpressions", end="...", flush=True)
        array_cse = sme.cse(list_smp)
        for cse_idx in range(len(array_cse[0])):
            left_cse = self.convert_to_cpp(array_cse[0][cse_idx][0])
            right_cse = self.convert_to_cpp(array_cse[0][cse_idx][1])
            cw.writer(
                fstream,
                f"const amrex::Real {left_cse} = {right_cse};",
            )
            # Debugging prints in mechanism.H...
            # cw.writer(
            #     fstream,
            #     f"""std::cout << "{left_cse} = " << {left_cse} << std::endl;""",
            # )

        print("Done!", flush=True)

        # Write all the entries
        for i in range(n):
            # The full expression is stored in array_cse index 1
            cpp_str = self.convert_to_cpp(array_cse[1][i])
            if index_list is None:
                cw.writer(
                    fstream,
                    f"{array_str}[{str(i)}] = {cpp_str};",
                )
            else:
                cw.writer(
                    fstream,
                    f"{array_str}[{str(index_list[i])}] = {cpp_str};",
                )

    def write_dscqss_to_cpp(self, species_info, cw, fstream):
        """Write dscqss terms as functions of common subexpressions."""
        n_dscqssdscqss = len(self.dscqssdscqss)

        list_smp = list(self.dscqssdscqss.values()) + list(self.dscqssdsc.values())
        n_total = len(list_smp)

        dscqssdscqss_tuples = list(self.dscqssdscqss.keys())
        dscqssdsc_tuples = list(self.dscqssdsc.keys())
        tuple_list = dscqssdscqss_tuples + dscqssdsc_tuples

        # Write common expressions
        print(
            "Start making common subexpressions for dscqss",
            end="...",
            flush=True,
        )
        array_cse = sme.cse(list_smp)
        for cse_idx in range(len(array_cse[0])):
            left_cse = self.convert_to_cpp(array_cse[0][cse_idx][0])
            right_cse = self.convert_to_cpp(array_cse[0][cse_idx][1])
            cw.writer(
                fstream,
                f"const amrex::Real {left_cse} = {right_cse};",
            )
        print("Done!", flush=True)

        cw.writer(fstream, cw.comment("Write dscqss_dsc terms..."))

        # Write all the entries
        for i in range(n_total):
            # The full expression is stored in array_cse index 1
            cpp_str = self.convert_to_cpp(array_cse[1][i])

            num_idx = tuple_list[i][0]
            den_idx = tuple_list[i][1]

            if i < n_dscqssdscqss:
                cw.writer(
                    fstream,
                    f"const amrex::Real dscqss{num_idx}dscqss{den_idx} = {cpp_str};",
                )
            else:
                cw.writer(
                    fstream,
                    f"const amrex::Real dscqss{num_idx}dsc{den_idx} = {cpp_str};",
                )

        cw.writer(fstream, cw.comment("Write chain rule terms..."))

        # Now write the chain rule terms
        for _, item in species_info.scqss_df.iterrows():
            for scnum in range(species_info.n_species):
                start_string = f"""dscqss{item["number"]}dsc{scnum}"""
                chain_string = []
                for scqss_dep in item["scqss_dep"]:
                    scqssdepnum = self.syms_to_specnum(scqss_dep)
                    chain_string.append(
                        f"""dscqss{item["number"]}dscqss{scqssdepnum} * """
                        f"""dscqss_dsc[{species_info.n_species*scqssdepnum + scnum}]"""
                    )

                if chain_string:
                    final_string = f"{start_string} + {chain_string[0]}"
                    for ics in range(len(chain_string) - 1):
                        final_string += f" + {chain_string[ics+1]}"
                else:
                    final_string = start_string

                cw.writer(
                    fstream,
                    f"dscqss_dsc[{species_info.n_species * item['number'] + scnum}]"
                    f" = {final_string};",
                )

    def write_symjac_to_cpp_cpu(self, species_info, cw, fstream):
        """Write species jacobian terms as functions of common subexpressions.

        Many variables are created to ensure readability.
        The memory constraint makes the format useful for CPU.
        """
        n_dscqssdscqss = len(self.dscqssdscqss)
        n_dscqssdsc = len(self.dscqssdsc)
        n_dwdotdscqss = len(self.dwdotdscqss)

        list_smp = (
            list(self.dscqssdscqss.values())
            + list(self.dscqssdsc.values())
            + list(self.dwdotdscqss.values())
            + list(self.dwdotdsc.values())
        )
        n_total = len(list_smp)

        dscqssdscqss_tuples = list(self.dscqssdscqss.keys())
        dscqssdsc_tuples = list(self.dscqssdsc.keys())
        dwdotdscqss_tuples = list(self.dwdotdscqss.keys())
        dwdotdsc_tuples = list(self.dwdotdsc.keys())
        tuple_list = (
            dscqssdscqss_tuples
            + dscqssdsc_tuples
            + dwdotdscqss_tuples
            + dwdotdsc_tuples
        )

        # Write common expressions
        print("Starting jacobian common subexpressions", end="...", flush=True)
        array_cse = sme.cse(list_smp)
        print("Done!", flush=True)

        (
            common_expr_lhs,
            common_expr_rhs,
            final_expr,
            to_replace,
            replace_with,
        ) = self.reduce_expr(array_cse)
        for cse_idx in range(len(common_expr_lhs)):
            if common_expr_lhs[cse_idx] in to_replace:
                ind = to_replace.index(common_expr_lhs[cse_idx])
                left_cse = self.convert_to_cpp(replace_with[ind])
                right_cse = self.convert_to_cpp(common_expr_rhs[cse_idx])
                if not left_cse == right_cse:
                    cw.writer(
                        fstream,
                        f"{left_cse} = {right_cse};",
                    )
            elif common_expr_lhs[cse_idx] in replace_with:
                left_cse = self.convert_to_cpp(common_expr_lhs[cse_idx])
                right_cse = self.convert_to_cpp(common_expr_rhs[cse_idx])
                cw.writer(
                    fstream,
                    f"amrex::Real {left_cse} = {right_cse};",
                )
            else:
                left_cse = self.convert_to_cpp(common_expr_lhs[cse_idx])
                right_cse = self.convert_to_cpp(common_expr_rhs[cse_idx])
                cw.writer(
                    fstream,
                    f"const amrex::Real {left_cse} = {right_cse};",
                )

        cw.writer(
            fstream,
            cw.comment(
                "Write base terms for dscqssdscqss, dscqssdsc, dwdotdscqss,"
                " and dwdotdsc..."
            ),
        )

        # Write all the entries in human readable format
        for i in range(n_total):
            # The full expression is stored in array_cse index 1
            cpp_str = self.convert_to_cpp(final_expr[i])

            num_idx = tuple_list[i][0]
            den_idx = tuple_list[i][1]

            if i < n_dscqssdscqss:
                cw.writer(
                    fstream,
                    f"const amrex::Real dscqss{num_idx}dscqss{den_idx} = {cpp_str};",
                )
            elif i < n_dscqssdscqss + n_dscqssdsc:
                cw.writer(
                    fstream,
                    f"const amrex::Real dscqss{num_idx}dsc{den_idx} = {cpp_str};",
                )
            elif i < n_dscqssdscqss + n_dscqssdsc + n_dwdotdscqss:
                cw.writer(
                    fstream,
                    f"const amrex::Real dwdot{num_idx}dscqss{den_idx} = {cpp_str};",
                )
            else:
                cw.writer(
                    fstream,
                    f"const amrex::Real dwdot{num_idx}dsc{den_idx} = {cpp_str};",
                )

        cw.writer(fstream, cw.comment("Write dscqss_dsc terms..."))

        # Now write the chain rule terms
        for _, item in species_info.scqss_df.iterrows():
            for scnum in range(species_info.n_species):
                start_string = f"""dscqss{item["number"]}dsc{scnum}"""
                chain_string = []
                for scqss_dep in item["scqss_dep"]:
                    scqssdepnum = self.syms_to_specnum(scqss_dep)
                    chain_string.append(
                        f"""dscqss{item["number"]}dscqss{scqssdepnum} * """
                        f"""dscqss_dsc[{species_info.n_species*scqssdepnum + scnum}]"""
                    )

                if chain_string:
                    final_string = f"{start_string} + {chain_string[0]}"
                    for ics in range(len(chain_string) - 1):
                        final_string += f" + {chain_string[ics+1]}"
                else:
                    final_string = start_string

                cw.writer(
                    fstream,
                    f"dscqss_dsc[{species_info.n_species * item['number'] + scnum}]"
                    f" = {final_string};",
                )

        # Now write the full jacobian expression
        cw.writer(fstream, cw.comment("Write the full Jacobian expression..."))

        for idx, item in species_info.wdot_df.iterrows():
            for scnum in range(species_info.n_species):
                start_string = f"""dwdot{idx}dsc{scnum}"""
                chain_string = []
                for scqssnum in species_info.scqss_df["number"]:
                    chain_string.append(
                        f"""dwdot{idx}dscqss{scqssnum} * """
                        f"""dscqss_dsc[{species_info.n_species*scqssnum + scnum}]"""
                    )

                final_string = f"{start_string} + {chain_string[0]}"
                for ics in range(len(chain_string) - 1):
                    final_string += f" + {chain_string[ics+1]}"

                cw.writer(
                    fstream,
                    f"J[{(species_info.n_species + 1) * scnum + item['number']}]"
                    f" = {final_string};",
                )

    def write_symjac_to_cpp_gpu(self, species_info, cw, fstream):
        """Write species jacobian terms as functions of common subexpressions.

        As little as possible intermediate variables are declared which
        negatively affects readability.
        The memory efficiency makes the format useful for GPU.
        """
        n_dscqssdscqss = len(self.dscqssdscqss)
        n_dscqssdsc = len(self.dscqssdsc)
        n_dwdotdscqss = len(self.dwdotdscqss)
        n_dwdotdsc = len(self.dwdotdsc)

        list_smp = (
            list(self.dscqssdscqss.values())
            + list(self.dscqssdsc.values())
            + list(self.dwdotdscqss.values())
            + list(self.dwdotdsc.values())
        )

        dscqssdscqss_tuples = list(self.dscqssdscqss.keys())
        dscqssdsc_tuples = list(self.dscqssdsc.keys())
        dwdotdscqss_tuples = list(self.dwdotdscqss.keys())
        dwdotdsc_tuples = list(self.dwdotdsc.keys())
        tuple_list = (
            dscqssdscqss_tuples
            + dscqssdsc_tuples
            + dwdotdscqss_tuples
            + dwdotdsc_tuples
        )
        # Create Pandas dataframe to store CSE information
        term_type = (
            ["dscqssdscqss"] * n_dscqssdscqss
            + ["dscqssdsc"] * n_dscqssdsc
            + ["dwdotdscqss"] * n_dwdotdscqss
            + ["dwdotdsc"] * n_dwdotdsc
        )
        jac_df = pd.DataFrame({"tuples": tuple_list, "type": term_type})

        # Write common expressions
        print("Starting jacobian common subexpressions", end="...", flush=True)
        array_cse = sme.cse(list_smp)
        print("Done!", flush=True)

        (
            common_expr_lhs,
            common_expr_rhs,
            final_expr,
            to_replace,
            replace_with,
        ) = self.reduce_expr(array_cse)
        for cse_idx in range(len(common_expr_lhs)):
            if common_expr_lhs[cse_idx] in to_replace:
                ind = to_replace.index(common_expr_lhs[cse_idx])
                left_cse = self.convert_to_cpp(replace_with[ind])
                right_cse = self.convert_to_cpp(common_expr_rhs[cse_idx])
                if not left_cse == right_cse:
                    cw.writer(
                        fstream,
                        f"{left_cse} = {right_cse};",
                    )
            elif common_expr_lhs[cse_idx] in replace_with:
                left_cse = self.convert_to_cpp(common_expr_lhs[cse_idx])
                right_cse = self.convert_to_cpp(common_expr_rhs[cse_idx])
                cw.writer(
                    fstream,
                    f"amrex::Real {left_cse} = {right_cse};",
                )
            else:
                left_cse = self.convert_to_cpp(common_expr_lhs[cse_idx])
                right_cse = self.convert_to_cpp(common_expr_rhs[cse_idx])
                cw.writer(
                    fstream,
                    f"const amrex::Real {left_cse} = {right_cse};",
                )

        # Compute dscqss_dsc strings from CSEs
        dscqss_dsc = [""] * (species_info.n_species * species_info.n_qssa_species)

        for _, scqss_item in species_info.scqss_df.iterrows():
            for _, sc_item in species_info.sc_df.iterrows():
                dscqss_dsc_idx = (
                    species_info.n_species * scqss_item["number"] + sc_item["number"]
                )

                # Identify the CSE index of dscqssdsc term
                dscqssdsc_cse_idx = self.get_cse_idx(
                    jac_df,
                    "dscqssdsc",
                    (scqss_item["number"], sc_item["number"]),
                )

                # Get the dscqssdsc CSE string to start
                start_string = (
                    f"""{self.convert_to_cpp(final_expr[dscqssdsc_cse_idx])}"""
                )
                # Loop through the chain terms
                chain_string = []
                for scqss_dep in scqss_item["scqss_dep"]:
                    scqssdepnum = self.syms_to_specnum(scqss_dep)

                    # Identify the CSE index of the dscqssdscqss term
                    dscqssdscqss_cse_idx = self.get_cse_idx(
                        jac_df,
                        "dscqssdscqss",
                        (scqss_item["number"], scqssdepnum),
                    )
                    # Append the dscqssdscqss CSE to chain list
                    cse_string = self.convert_to_cpp(final_expr[dscqssdscqss_cse_idx])
                    dscqssdsc_string = dscqss_dsc[
                        species_info.n_species * scqssdepnum + sc_item["number"]
                    ]

                    # only append the term if it is NOT multiplied by 0
                    if not dscqssdsc_string == "0" and not str(cse_string) == "0":
                        chain_string.append(
                            f"""({cse_string}) * ({dscqssdsc_string})"""
                        )

                if chain_string:
                    final_string = f"""{start_string} + {chain_string[0]}"""
                    for ics in range(len(chain_string) - 1):
                        final_string += f""" + {chain_string[ics+1]}"""
                else:
                    final_string = start_string
                dscqss_dsc[dscqss_dsc_idx] = final_string

        # Now write the full jacobian expression
        cw.writer(fstream, cw.comment("Write the full Jacobian expression..."))
        for _, wdot_item in species_info.wdot_df.iterrows():
            for _, sc_item in species_info.sc_df.iterrows():
                # Find the CSE index for dwdotdsc
                dwdotdsc_cse_idx = self.get_cse_idx(
                    jac_df,
                    "dwdotdsc",
                    (wdot_item["number"], sc_item["number"]),
                )

                # Get the dwdotdsc CSE string to start
                start_string = (
                    f"""{self.convert_to_cpp(final_expr[dwdotdsc_cse_idx])}"""
                )
                # Loop through the chain terms
                chain_string = []
                for scqss in wdot_item["scqss_dep"]:
                    scqssnum = self.syms_to_specnum(scqss)

                    # Find the CSE index for dwdotdscqss
                    dwdotdscqss_cse_idx = self.get_cse_idx(
                        jac_df, "dwdotdscqss", (wdot_item["number"], scqssnum)
                    )
                    # Append the dwdotdscqss * dscqssdsc term to the chain lise
                    cse_string = self.convert_to_cpp(final_expr[dwdotdscqss_cse_idx])
                    dscqssdsc_string = dscqss_dsc[
                        species_info.n_species * scqssnum + sc_item["number"]
                    ]
                    # only append the term if it is NOT multiplied by 0
                    if not dscqssdsc_string == "0" and not str(cse_string) == "0":
                        chain_string.append(f"""({cse_string})*({dscqssdsc_string})""")
                if chain_string:
                    final_string = f"""{start_string}+{chain_string[0]}"""
                    for ics in range(len(chain_string) - 1):
                        final_string += f"""+{chain_string[ics+1]}"""
                else:
                    final_string = start_string

                idx = (species_info.n_species + 1) * sc_item["number"] + wdot_item[
                    "number"
                ]
                cw.writer(
                    fstream,
                    f"J[{idx}] = {final_string};",
                )

    def write_array_to_cpp_no_cse(
        self, list_smp, array_str, cw, fstream, index_list=None
    ):
        """Convert sympy array to C code compatible string."""
        n = len(list_smp)

        # Write all the entries
        for i in range(n):
            # The full expression is stored in array_cse index 1
            try:
                cpp_str = self.convert_to_cpp(list_smp[i])
            except RecursionError:
                if index_list is None:
                    print("Recursion error for index = ", i)
                else:
                    print("Recursion error for index = ", index_list[i])
                cpp_str = "0"
            if index_list is None:
                cw.writer(
                    fstream,
                    f"{array_str}[{str(i)}] = {cpp_str};",
                )
            else:
                cw.writer(
                    fstream,
                    f"{array_str}[{str(index_list[i])}] = {cpp_str};",
                )

    def get_cse_idx(self, df, type_name, tuple_val):
        """Extract id of common expression from the symengine array."""
        tmp_df = df[df.type == type_name]
        cse_idx = tmp_df.index[tmp_df.tuples == tuple_val].tolist()[0]

        return cse_idx

    def compute_spec_dependencies(self, sym_smp):
        """Routine to compute the sc and sc_qss dependencies."""
        free_symb = sorted(sym_smp.free_symbols, key=str)
        sc_depend = []
        scqss_depend = []
        for symbol in free_symb:
            if "sc_qss" in str(symbol):
                scqss_depend.append(symbol)
            elif "sc" in str(symbol):
                sc_depend.append(symbol)
            else:
                pass

        return sc_depend, scqss_depend

    def compute_dscqss_dscqss(self, species_info):
        """Routine that computes the dscqss i / dscqss j."""
        # Loop over the species info dataframe and compute dependencies
        for _, item in species_info.scqss_df.iterrows():
            # Loop over the dependencies of scqss
            for scqss in item["scqss_dep"]:
                scqssnum = self.syms_to_specnum(scqss)
                self.dscqssdscqss[(item["number"], scqssnum)] = sme.diff(
                    self.sc_qss_smp[item["number"]],
                    sme.symbols(f"sc_qss[{scqssnum}]"),
                )

            # # Loop over the dependencies of scqss
            # for scqssnum in species_info.scqss_df["number"]:
            #     # scqssnum = self.syms_to_specnum(scqss)
            #     self.dscqssdscqss[(item["number"], scqssnum)] = sme.diff(
            #         self.sc_qss_smp[item["number"]],
            #         sme.symbols(f"sc_qss[{scqssnum}]"),
            #     )

    def compute_dscqss_dsc(self, species_info):
        """Routine that computes the dscqss i / dsc j."""
        # Loop over the species info dataframe and compute all dsc derivatives
        for _, item in species_info.scqss_df.iterrows():
            # Loop over all sc terms
            for scnum in range(species_info.n_species):
                self.dscqssdsc[(item["number"], scnum)] = sme.diff(
                    self.sc_qss_smp[item["number"]],
                    sme.symbols(f"sc[{scnum}]"),
                )

                # Only do the derivative if there is an sc dependence explicitly included
                # if f"sc[{scnum}]" in str(item["sc_dep"]):
                #     self.dscqssdsc[(item["name"], f"sc[{scnum}]")] = sme.diff(
                #         self.sc_qss_smp[item["number"]],
                #         sme.symbols(f"sc[{scnum}]"),
                #     )
                # else:
                #     self.dscqssdsc[(item["name"], f"sc[{scnum}]")] = 0

    def compute_dwdot_dsc(self, species_info):
        """Routine that computes the dwdot i / dsc j."""
        # Loop over all wdots and sc terms
        for wdot_idx, _ in species_info.wdot_df.iterrows():
            # Loop over all sc terms
            for sc_idx in range(species_info.n_species):
                self.dwdotdsc[(wdot_idx, sc_idx)] = sme.diff(
                    self.wdot_smp[wdot_idx], sme.symbols(f"sc[{sc_idx}]")
                )

    def compute_dwdot_dscqss(self, species_info):
        """Routine that computes the dwdot i / dscqss j."""
        # Loop over all wdots and sc terms
        for wdot_idx, _ in species_info.wdot_df.iterrows():
            # Loop over all scqss terms
            for scqss in species_info.scqss_df["name"]:
                scqssnum = self.syms_to_specnum(scqss)
                self.dwdotdscqss[(wdot_idx, scqssnum)] = sme.diff(
                    self.wdot_smp[wdot_idx], sme.symbols(f"sc_qss[{scqssnum}]")
                )

    def compute_jacobian(self, species_info):
        """Routine that computes the Jacobian without chain ruling."""
        # Create intermediate vectors
        dscqss_dsc = [0.0] * (species_info.n_species * species_info.n_qssa_species)

        for _, scqss_item in species_info.scqss_df.iterrows():
            for _, sc_item in species_info.sc_df.iterrows():
                dscqss_dsc_idx = (
                    species_info.n_species * scqss_item["number"] + sc_item["number"]
                )

                for scqss_dep in scqss_item["scqss_dep"]:
                    scqssdepnum = self.syms_to_specnum(scqss_dep)
                    dscqss_dsc[dscqss_dsc_idx] += (
                        self.dscqssdscqss[(scqss_item["number"], scqssdepnum)]
                        * dscqss_dsc[
                            species_info.n_species * scqssdepnum + sc_item["number"]
                        ]
                    )

                dscqss_dsc[dscqss_dsc_idx] += self.dscqssdsc[
                    (scqss_item["number"], sc_item["number"])
                ]

        # Loop over all wdots and sc terms
        for wdot_idx, _ in species_info.wdot_df.iterrows():
            # Loop over all sc terms
            for sc_idx, sc_item in species_info.sc_df.iterrows():
                self.jacobian[(wdot_idx, sc_idx)] = 0.0

                for scqssnum in species_info.scqss_df["number"]:
                    self.jacobian[(wdot_idx, sc_idx)] += (
                        self.dwdotdscqss[(wdot_idx, scqssnum)]
                        * dscqss_dsc[
                            species_info.n_species * scqssnum + sc_item["number"]
                        ]
                    )

                # add in the non-chain derivative term
                self.jacobian[(wdot_idx, sc_idx)] += self.dwdotdsc[(wdot_idx, sc_idx)]
