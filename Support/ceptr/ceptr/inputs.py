"""Classes for toml input files."""

import toml


class Parameter:
    """Parameter class to hold inputs from file."""

    def __init__(self, default, helper, typer, choices=None):
        self.default = default
        self.helper = helper
        self.typer = typer
        self.choices = choices
        self.set_value(self.default)

    def __repr__(self):
        """Return description."""
        return self.describe()

    def __str__(self):
        """Return instance string."""
        return f"""Instance of {self.describe()}"""

    def describe(self):
        """Describe parameter class."""
        return (
            f"""Parameter({self.default}, {self.helper},"""
            f""" {self.typer}, choices={self.choices})"""
        )

    def set_value(self, value):
        """Set value of parameter."""
        if isinstance(value, self.typer):
            self.value = value
        elif value is None:
            self.value = value
        else:
            raise TypeError(
                f"Type does not match for {self.name} ({self.typer} is not"
                f" {type(value)})"
            )

        if (self.choices is not None) and (value not in self.choices):
            raise ValueError(
                f"Value {value} is not part of the available choices {self.choices}"
            )


class Input:
    """Input class to parse the toml file."""

    def __init__(self):
        self.inputs = {
            "Readability": {
                "hformat": Parameter(
                    "cpu",
                    "Style format for .H file output\nCPU: will print"
                    " intermediate variables used for chainruling. This"
                    " gives a readable version of the Jacobian entries,"
                    " albeit memory consuming.\nGPU: will not print"
                    " intermediate variables used for chainruling, and"
                    " instead will replace them directly in the Jacobian"
                    " entries. This gives a less readable version of the"
                    " Jacobian, but more memory efficient.",
                    str,
                    choices=["gpu", "cpu"],
                )
            },
            "Arithmetic": {
                "remove_1": Parameter(
                    False, "Remove factor 1.0 in printed expressions", bool
                ),
                "remove_pow": Parameter(
                    False,
                    "Replace pow(...,n) with multiplications or divisions"
                    " if n<=3 and n>=-3 in printed expressions.",
                    bool,
                ),
                "remove_pow10": Parameter(
                    False,
                    "Remove pow(10,x) in printed expressions and replace"
                    " with exp(ln(10)*x).",
                    bool,
                ),
            },
            "Replacement": {
                "min_op_count": Parameter(
                    0,
                    "Counts number operations used to construct each"
                    " common subexpression and replace the common"
                    " subexpression if the number of operations is less or"
                    " equal to the value",
                    int,
                ),
                "min_op_count_all": Parameter(
                    0,
                    "Similar to --min_op_count but also counts how many"
                    " times that common subexpression is used later.\nThe"
                    " meaning of the value passed is how many more"
                    " operations will be done if the common subexpression"
                    " is eliminated.\nThis option only marginally increase"
                    " the file size (therefore compile time), while still"
                    " being memory efficient.",
                    int,
                ),
                "gradual_op_count": Parameter(
                    False,
                    "Gradual elimination of common subexpressions.\nUseful"
                    " if --min_op_count or --min_op_count_all are"
                    " active.\nLoops from 1 to the min_op_count and"
                    " min_op_count_all values and gradually eliminate the"
                    " common subexpressions.\nThis has the advantage of"
                    " ensuring that the memory footprint is strictly"
                    " monotonically decreasing as min_op_count and"
                    " min_op_count_all are increased.",
                    bool,
                ),
                "remove_single_symbols_cse": Parameter(
                    False,
                    "Remove common subexpressions that are made of 1"
                    " symbol.\nThose common subexpressions are typically"
                    " `-xxx` and may not appear as worth replacing because"
                    " they save 1 operations and are reused multiple"
                    " times.\nHowever, when replaced in the later"
                    " expressions, the `-` operations typically disappear"
                    " or is merged into another operations which actually"
                    " does not increase the total number of operations.",
                    bool,
                ),
            },
            "Recycle": {
                "store_in_jacobian": Parameter(
                    False,
                    "Use the Jacobian array as a temporary space to store"
                    " intermediate variables.\nIn particular, the last row of"
                    " the Jacobian (dependence with respect to temperature) is"
                    " done by finite difference which requires storing"
                    " intermediate variables  (production rate, forward and"
                    " backward reactions)."
                    + "\nWhen the option is active, the `productionRate`"
                    " function used to compute the finite difference is"
                    " replaced with a `productionRate_light` function where"
                    " references to different parts of the Jacobian are used"
                    " in place of allocating new arrays.",
                    bool,
                ),
                "recycle_cse": Parameter(
                    False,
                    "Reuse common subexpressions that are not used later"
                    " to avoid declaring new temporary reals",
                    bool,
                ),
            },
            "Characters": {
                "round_decimals": Parameter(
                    False,
                    "Round decimal numbers when possible to minimize character count",
                    bool,
                ),
            },
        }

    def write_toml(self):
        """Write inputs as TOML format."""
        for section in self.inputs.keys():
            print(f"""[{section}]""")
            for name, param in self.inputs[section].items():
                if type(param.value) is str:
                    print(f"""{name} = "{param.value}" """)
                else:
                    print(f"""{name} = {param.value}""")

    def print_help(self):
        """Print the defaults and help."""
        for section in self.inputs.keys():
            print(f"""[{section}]""")
            for name, param in self.inputs[section].items():
                if type(param.value) is str:
                    print(f"""{name} = "{param.default}" \n\n # {param.helper} \n\n""")
                else:
                    print(f"""{name} = {param.default} \n\n # {param.helper} \n\n""")

    def from_toml(self, fname):
        """Read TOML file for inputs."""
        parsed = toml.load(fname)
        for section in parsed.keys():
            for key, value in parsed[section].items():
                self.inputs[section][key].set_value(value)
