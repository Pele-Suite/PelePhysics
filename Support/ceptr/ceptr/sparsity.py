"""Sparsity patterns."""

import ceptr.writer as cw


def sparsity(fstream, species_info):
    """Write sparsity pattern of Jacobian."""
    n_species = species_info.n_species

    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("compute the sparsity pattern of the chemistry Jacobian"),
    )
    cw.writer(
        fstream,
        "void SPARSITY_INFO( int * nJdata, const int * consP, int NCELLS)",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        f"amrex::GpuArray<amrex::Real,{(n_species + 1) ** 2}> Jac = {{0.0}};",
    )
    cw.writer(fstream, f"amrex::GpuArray<amrex::Real,{n_species}> conc = {{0.0}};")
    cw.writer(fstream, f"for (int n=0; n<{n_species}; n++) {{")
    cw.writer(fstream, f"    conc[n] = 1.0/ {n_species:f} ;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "aJacobian(Jac.data(), conc.data(), 1500.0, *consP);")
    cw.writer(fstream)

    cw.writer(fstream, "int nJdata_tmp = 0;")
    cw.writer(fstream, f"for (int k=0; k<{n_species + 1}; k++) {{")

    cw.writer(fstream, f"for (int l=0; l<{n_species + 1}; l++) {{")

    cw.writer(fstream, f"if(Jac[ {n_species + 1} * k + l] != 0.0){{")

    cw.writer(fstream, "nJdata_tmp = nJdata_tmp + 1;")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream)
    cw.writer(fstream, "*nJdata = NCELLS * nJdata_tmp;")

    cw.writer(fstream, "}")
    cw.writer(fstream)
    cw.writer(fstream)

    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment("compute the sparsity pattern of the system Jacobian"),
    )
    cw.writer(
        fstream,
        "void SPARSITY_INFO_SYST( int * nJdata, const int * consP, int NCELLS)",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        f"amrex::GpuArray<amrex::Real,{(n_species + 1) ** 2}> Jac = {{0.0}};",
    )
    cw.writer(fstream, f"amrex::GpuArray<amrex::Real,{n_species}> conc = {{0.0}};")
    cw.writer(fstream, f"for (int n=0; n<{n_species}; n++) {{")
    cw.writer(fstream, f"    conc[n] = 1.0/ {n_species:f} ;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "aJacobian(Jac.data(), conc.data(), 1500.0, *consP);")
    cw.writer(fstream)

    cw.writer(fstream, "int nJdata_tmp = 0;")
    cw.writer(fstream, f"for (int k=0; k<{n_species + 1}; k++) {{")

    cw.writer(fstream, f"for (int l=0; l<{n_species + 1}; l++) {{")

    cw.writer(fstream, "if(k == l){")

    cw.writer(fstream, "nJdata_tmp = nJdata_tmp + 1;")

    cw.writer(fstream, "} else {")

    cw.writer(fstream, f"if(Jac[ {n_species + 1} * k + l] != 0.0){{")

    cw.writer(fstream, "nJdata_tmp = nJdata_tmp + 1;")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream)
    cw.writer(fstream, "*nJdata = NCELLS * nJdata_tmp;")

    cw.writer(fstream, "}")
    cw.writer(fstream)
    cw.writer(fstream)

    cw.writer(fstream)
    cw.writer(
        fstream,
        cw.comment(
            "compute the sparsity pattern of the simplified (for"
            " preconditioning) system Jacobian"
        ),
    )
    cw.writer(
        fstream,
        "void SPARSITY_INFO_SYST_SIMPLIFIED( int * nJdata, const int * consP)",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        f"amrex::GpuArray<amrex::Real,{(n_species + 1) ** 2}> Jac = {{0.0}};",
    )
    cw.writer(fstream, f"amrex::GpuArray<amrex::Real,{n_species}> conc = {{0.0}};")
    cw.writer(fstream, f"for (int n=0; n<{n_species}; n++) {{")
    cw.writer(fstream, f"    conc[n] = 1.0/ {n_species:f} ;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "aJacobian_precond(Jac.data(), conc.data(), 1500.0, *consP);")
    cw.writer(fstream)

    cw.writer(fstream, "int nJdata_tmp = 0;")
    cw.writer(fstream, f"for (int k=0; k<{n_species + 1}; k++) {{")

    cw.writer(fstream, f"for (int l=0; l<{n_species + 1}; l++) {{")

    cw.writer(fstream, "if(k == l){")

    cw.writer(fstream, "nJdata_tmp = nJdata_tmp + 1;")

    cw.writer(fstream, "} else {")

    cw.writer(fstream, f"if(Jac[ {n_species + 1} * k + l] != 0.0){{")

    cw.writer(fstream, "nJdata_tmp = nJdata_tmp + 1;")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream)
    cw.writer(fstream, "nJdata[0] = nJdata_tmp;")

    cw.writer(fstream, "}")
    cw.writer(fstream)
    cw.writer(fstream)

    cw.writer(
        fstream,
        cw.comment(
            "compute the sparsity pattern of the chemistry Jacobian in CSC"
            " format -- base 0"
        ),
    )
    cw.writer(
        fstream,
        "void SPARSITY_PREPROC_CSC(int *  rowVals, int *  colPtrs, const"
        " int * consP, int NCELLS)",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        f"amrex::GpuArray<amrex::Real,{(n_species + 1) ** 2}> Jac = {{0.0}};",
    )
    cw.writer(fstream, f"amrex::GpuArray<amrex::Real,{n_species}> conc = {{0.0}};")
    cw.writer(fstream, f"for (int n=0; n<{n_species}; n++) {{")
    cw.writer(fstream, f"    conc[n] = 1.0/ {n_species:f} ;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "aJacobian(Jac.data(), conc.data(), 1500.0, *consP);")
    cw.writer(fstream)

    cw.writer(fstream, "colPtrs[0] = 0;")
    cw.writer(fstream, "int nJdata_tmp = 0;")
    cw.writer(fstream, "for (int nc=0; nc<NCELLS; nc++) {")

    cw.writer(fstream, f"int offset_row = nc * {n_species + 1};")
    cw.writer(fstream, f"int offset_col = nc * {n_species + 1};")
    cw.writer(fstream, f"for (int k=0; k<{n_species + 1}; k++) {{")

    cw.writer(fstream, f"for (int l=0; l<{n_species + 1}; l++) {{")

    cw.writer(fstream, f"if(Jac[{n_species + 1}*k + l] != 0.0) {{")

    cw.writer(fstream, "rowVals[nJdata_tmp] = l + offset_row; ")
    cw.writer(fstream, "nJdata_tmp = nJdata_tmp + 1; ")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")
    cw.writer(fstream, "colPtrs[offset_col + (k + 1)] = nJdata_tmp;")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")
    cw.writer(fstream)

    cw.writer(
        fstream,
        cw.comment(
            "compute the sparsity pattern of the chemistry Jacobian in CSR"
            " format -- base 0"
        ),
    )
    cw.writer(
        fstream,
        "void SPARSITY_PREPROC_CSR(int * colVals, int * rowPtrs, const int"
        " * consP, int NCELLS, int base)",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        f"amrex::GpuArray<amrex::Real,{(n_species + 1) ** 2}> Jac = {{0.0}};",
    )
    cw.writer(fstream, f"amrex::GpuArray<amrex::Real,{n_species}> conc = {{0.0}};")
    cw.writer(fstream, f"for (int n=0; n<{n_species}; n++) {{")
    cw.writer(fstream, f"    conc[n] = 1.0/ {n_species:f} ;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "aJacobian(Jac.data(), conc.data(), 1500.0, *consP);")
    cw.writer(fstream)

    cw.writer(fstream, "if (base == 1) {")

    cw.writer(fstream, "rowPtrs[0] = 1;")
    cw.writer(fstream, "int nJdata_tmp = 1;")
    cw.writer(fstream, "for (int nc=0; nc<NCELLS; nc++) {")

    cw.writer(fstream, f"int offset = nc * {n_species + 1};")
    cw.writer(fstream, f"for (int l=0; l<{n_species + 1}; l++) {{")

    cw.writer(fstream, f"for (int k=0; k<{n_species + 1}; k++) {{")

    cw.writer(fstream, f"if(Jac[{n_species + 1}*k + l] != 0.0) {{")

    cw.writer(fstream, "colVals[nJdata_tmp-1] = k+1 + offset; ")
    cw.writer(fstream, "nJdata_tmp = nJdata_tmp + 1; ")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")
    cw.writer(fstream, "rowPtrs[offset + (l + 1)] = nJdata_tmp;")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream, "} else {")

    cw.writer(fstream, "rowPtrs[0] = 0;")
    cw.writer(fstream, "int nJdata_tmp = 0;")
    cw.writer(fstream, "for (int nc=0; nc<NCELLS; nc++) {")

    cw.writer(fstream, f"int offset = nc * {n_species + 1};")
    cw.writer(fstream, f"for (int l=0; l<{n_species + 1}; l++) {{")

    cw.writer(fstream, f"for (int k=0; k<{n_species + 1}; k++) {{")

    cw.writer(fstream, f"if(Jac[{n_species + 1}*k + l] != 0.0) {{")

    cw.writer(fstream, "colVals[nJdata_tmp] = k + offset; ")
    cw.writer(fstream, "nJdata_tmp = nJdata_tmp + 1; ")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")
    cw.writer(fstream, "rowPtrs[offset + (l + 1)] = nJdata_tmp;")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")
    cw.writer(fstream)

    cw.writer(
        fstream,
        cw.comment("compute the sparsity pattern of the system Jacobian"),
    )
    cw.writer(fstream, cw.comment("CSR format BASE is user choice"))
    cw.writer(
        fstream,
        "void SPARSITY_PREPROC_SYST_CSR(int * colVals, int * rowPtr, const"
        " int * consP, int NCELLS, int base)",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        f"amrex::GpuArray<amrex::Real,{(n_species + 1) ** 2}> Jac = {{0.0}};",
    )
    cw.writer(fstream, f"amrex::GpuArray<amrex::Real,{n_species}> conc = {{0.0}};")
    cw.writer(fstream, f"for (int n=0; n<{n_species}; n++) {{")
    cw.writer(fstream, f"    conc[n] = 1.0/ {n_species:f} ;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "aJacobian(Jac.data(), conc.data(), 1500.0, *consP);")
    cw.writer(fstream)

    cw.writer(fstream, "if (base == 1) {")

    cw.writer(fstream, "rowPtr[0] = 1;")
    cw.writer(fstream, "int nJdata_tmp = 1;")
    cw.writer(fstream, "for (int nc=0; nc<NCELLS; nc++) {")

    cw.writer(fstream, f"int offset = nc * {n_species + 1};")
    cw.writer(fstream, f"for (int l=0; l<{n_species + 1}; l++) {{")

    cw.writer(fstream, f"for (int k=0; k<{n_species + 1}; k++) {{")

    cw.writer(fstream, "if (k == l) {")

    cw.writer(fstream, "colVals[nJdata_tmp-1] = l+1 + offset; ")
    cw.writer(fstream, "nJdata_tmp = nJdata_tmp + 1; ")

    cw.writer(fstream, "} else {")

    cw.writer(fstream, f"if(Jac[{n_species + 1}*k + l] != 0.0) {{")

    cw.writer(fstream, "colVals[nJdata_tmp-1] = k+1 + offset; ")
    cw.writer(fstream, "nJdata_tmp = nJdata_tmp + 1; ")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")
    cw.writer(fstream, "rowPtr[offset + (l + 1)] = nJdata_tmp;")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream, "} else {")

    cw.writer(fstream, "rowPtr[0] = 0;")
    cw.writer(fstream, "int nJdata_tmp = 0;")
    cw.writer(fstream, "for (int nc=0; nc<NCELLS; nc++) {")

    cw.writer(fstream, f"int offset = nc * {n_species + 1};")
    cw.writer(fstream, f"for (int l=0; l<{n_species + 1}; l++) {{")

    cw.writer(fstream, f"for (int k=0; k<{n_species + 1}; k++) {{")

    cw.writer(fstream, "if (k == l) {")

    cw.writer(fstream, "colVals[nJdata_tmp] = l + offset; ")
    cw.writer(fstream, "nJdata_tmp = nJdata_tmp + 1; ")

    cw.writer(fstream, "} else {")

    cw.writer(fstream, f"if(Jac[{n_species + 1}*k + l] != 0.0) {{")

    cw.writer(fstream, "colVals[nJdata_tmp] = k + offset; ")
    cw.writer(fstream, "nJdata_tmp = nJdata_tmp + 1; ")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")
    cw.writer(fstream, "rowPtr[offset + (l + 1)] = nJdata_tmp;")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")
    cw.writer(fstream)

    cw.writer(
        fstream,
        cw.comment(
            "compute the sparsity pattern of the simplified (for precond)"
            " system Jacobian on CPU"
        ),
    )
    cw.writer(fstream, cw.comment("BASE 0"))
    cw.writer(
        fstream,
        "void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(int * rowVals, int *"
        " colPtrs, int * indx, const int * consP)",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        f"amrex::GpuArray<amrex::Real,{(n_species + 1) ** 2}> Jac = {{0.0}};",
    )
    cw.writer(fstream, f"amrex::GpuArray<amrex::Real,{n_species}> conc = {{0.0}};")
    cw.writer(fstream, f"for (int n=0; n<{n_species}; n++) {{")
    cw.writer(fstream, f"    conc[n] = 1.0/ {n_species:f} ;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "aJacobian_precond(Jac.data(), conc.data(), 1500.0, *consP);")
    cw.writer(fstream)

    cw.writer(fstream, "colPtrs[0] = 0;")
    cw.writer(fstream, "int nJdata_tmp = 0;")
    cw.writer(fstream, f"for (int k=0; k<{n_species + 1}; k++) {{")

    cw.writer(fstream, f"for (int l=0; l<{n_species + 1}; l++) {{")

    cw.writer(fstream, "if (k == l) {")

    cw.writer(fstream, "rowVals[nJdata_tmp] = l; ")
    cw.writer(fstream, f"indx[nJdata_tmp] = {n_species + 1}*k + l;")
    cw.writer(fstream, "nJdata_tmp = nJdata_tmp + 1; ")

    cw.writer(fstream, "} else {")

    cw.writer(fstream, f"if(Jac[{n_species + 1}*k + l] != 0.0) {{")

    cw.writer(fstream, "rowVals[nJdata_tmp] = l; ")
    cw.writer(fstream, f"indx[nJdata_tmp] = {n_species + 1}*k + l;")
    cw.writer(fstream, "nJdata_tmp = nJdata_tmp + 1; ")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")
    cw.writer(fstream, "colPtrs[k+1] = nJdata_tmp;")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")
    cw.writer(fstream)

    cw.writer(
        fstream,
        cw.comment(
            "compute the sparsity pattern of the simplified (for precond)"
            " system Jacobian"
        ),
    )
    cw.writer(fstream, cw.comment("CSR format BASE is under choice"))
    cw.writer(
        fstream,
        "void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(int * colVals, int *"
        " rowPtr, const int * consP, int base)",
    )
    cw.writer(fstream, "{")

    cw.writer(
        fstream,
        f"amrex::GpuArray<amrex::Real,{(n_species + 1) ** 2}> Jac = {{0.0}};",
    )
    cw.writer(fstream, f"amrex::GpuArray<amrex::Real,{n_species}> conc = {{0.0}};")
    cw.writer(fstream, f"for (int n=0; n<{n_species}; n++) {{")
    cw.writer(fstream, f"    conc[n] = 1.0/ {n_species:f} ;")
    cw.writer(fstream, "}")
    cw.writer(fstream, "aJacobian_precond(Jac.data(), conc.data(), 1500.0, *consP);")
    cw.writer(fstream)

    cw.writer(fstream, "if (base == 1) {")

    cw.writer(fstream, "rowPtr[0] = 1;")
    cw.writer(fstream, "int nJdata_tmp = 1;")
    cw.writer(fstream, f"for (int l=0; l<{n_species + 1}; l++) {{")

    cw.writer(fstream, f"for (int k=0; k<{n_species + 1}; k++) {{")

    cw.writer(fstream, "if (k == l) {")

    cw.writer(fstream, "colVals[nJdata_tmp-1] = l+1; ")
    cw.writer(fstream, "nJdata_tmp = nJdata_tmp + 1; ")

    cw.writer(fstream, "} else {")

    cw.writer(fstream, f"if(Jac[{n_species + 1}*k + l] != 0.0) {{")

    cw.writer(fstream, "colVals[nJdata_tmp-1] = k+1; ")
    cw.writer(fstream, "nJdata_tmp = nJdata_tmp + 1; ")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")
    cw.writer(fstream, "rowPtr[l+1] = nJdata_tmp;")

    cw.writer(fstream, "}")

    cw.writer(fstream, "} else {")

    cw.writer(fstream, "rowPtr[0] = 0;")
    cw.writer(fstream, "int nJdata_tmp = 0;")
    cw.writer(fstream, f"for (int l=0; l<{n_species + 1}; l++) {{")

    cw.writer(fstream, f"for (int k=0; k<{n_species + 1}; k++) {{")

    cw.writer(fstream, "if (k == l) {")

    cw.writer(fstream, "colVals[nJdata_tmp] = l; ")
    cw.writer(fstream, "nJdata_tmp = nJdata_tmp + 1; ")

    cw.writer(fstream, "} else {")

    cw.writer(fstream, f"if(Jac[{n_species + 1}*k + l] != 0.0) {{")

    cw.writer(fstream, "colVals[nJdata_tmp] = k; ")
    cw.writer(fstream, "nJdata_tmp = nJdata_tmp + 1; ")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")
    cw.writer(fstream, "rowPtr[l+1] = nJdata_tmp;")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")

    cw.writer(fstream, "}")
