# Using multicomponent transport from PelePhysics in PeleC

*PelePhysics PR #709 and PeleC PR #1068 (merged October 2026); this note was written before the merge and updated for the merged state*

This note describes the multicomponent transport capabilities added to the
`Simple` transport model of PelePhysics and how PeleC uses them to form the
species and energy diffusion fluxes. Both sides are merged: PelePhysics
`development` (#709) and PeleC `development` (#1068, `pelec.use_multicomponent_diffusion`,
`pelec.use_soret`, `pelec.use_dufour`; kernel `Source/DifftermMC.H`; example inputs
`Exec/RegTests/PMF/pmf-lidryer-multicomp.inp` and `pmf-lidryer-matrixdiff.inp`).
Section 3 below is the original design sketch; the merged code follows it with
the differences listed in Section 7.

---

## 1. What PelePhysics now provides

All quantities are in CGS and are evaluated per cell from $(T, \rho, Y)$.

| Quantity | Symbol | Storage | Model / option |
|---|---|---|---|
| Mixture-averaged diffusivities (unchanged) | $\texttt{Ddiag}_k = \rho \, D_k^{\rm mix} \, W_k/\overline{W}$ | `Ddiag[N]` | always |
| Multicomponent flux diffusion matrix | $\texttt{Dmat}_{kj} = \rho\, Y_k D_{kj}$ | `Dmat[N*N]`, row-major `k*N + j` | on request |
| Thermal conductivity | $\lambda$ | scalar | `transport.conductivity_model = mixture` (default), `lct1`, `lct3` |
| Rescaled thermal diffusion ratios | $\tilde\chi_k$, with $\chi_k = X_k \tilde\chi_k$ | `chit[N]` | on request, needs `lct1` or `lct3` |
| Mixture-averaged Soret coefficients (unchanged) | $\texttt{chi}_k$ (CHEMKIN $\Theta_k$) | `chi[N]` | `transport.use_soret = 1` |

Here $D_{kj}$ is the symmetric multicomponent diffusion matrix of Ern and
Giovangigli and $\texttt{Dmat}$ is EGlib's "$\rho \times$ flux diffusion
matrix" (`EGSDR1`): pressure independent, and with every column summing to
zero,

$$\sum_k \texttt{Dmat}_{kj} = 0 \qquad \text{for every } j .$$

$\lambda$ and $\tilde\chi$ come from EGlib's `EGSLCT1` (reduced system
$\Lambda_{[e]}$, $N$ unknowns) or `EGSLCT3` (full system $\Lambda$ with
separate translational and internal energy modes, $2N$ unknowns), solved with a
few preconditioned conjugate-gradient iterations. The thermal diffusion ratios
sum to zero, $\sum_k \chi_k = 0$.

### 1.1 Run-time inputs (read only by `Simple` transport)

```
transport.trace_epsilon          = 1.e-15   # shift applied to vanishing species (the historical value)
transport.matrix_diffusion_niter = 2        # projected iterations for Dmat (2 = EGlib)
transport.conductivity_model     = mixture  # mixture | lct1 | lct3
transport.conductivity_niter     = 3        # CG iterations for lct1/lct3 (3 = EGlib)
transport.use_soret              = 0        # unchanged: mixture-averaged Soret fit
```

`conductivity_model` is a model choice: when `lct1` or `lct3` is selected the
$\lambda$ returned by **every** entry point, including the historical one PeleC
calls today, is the LCT value. The default `mixture` reproduces previous
results bitwise. Requesting `chit` with the `mixture` model aborts.

### 1.2 Extended API

The historical calls are unchanged. The new quantities are obtained through a
request/output struct pair (`Source/Transport/TransportTypes.H`):

```cpp
using namespace pele::physics::transport;

TransportRequest req;          // bools, all default false
req.mu = req.xi = req.lam = true;
req.Dmat = true;               // multicomponent flux diffusion matrix
req.chit = true;               // rescaled thermal diffusion ratios (needs lct1/lct3)

amrex::Real mu, xi, lam;
amrex::Real Dmat[NUM_SPECIES * NUM_SPECIES];
amrex::Real chit[NUM_SPECIES];

TransportOutputs out;          // pointers, all default nullptr
out.mu = &mu; out.xi = &xi; out.lam = &lam;
out.Dmat = Dmat; out.chit = chit;

auto trans = pele::physics::PhysicsType::transport();
trans.transport(req, T, rho, Y, out, ltransparm);   // T in K, rho in g/cm3, Y mass fractions
```

Only requested quantities are evaluated and only non-null outputs are written.
A box version exists as well:

```cpp
trans.get_transport_coeffs(bx, req, Y_in, T_in, Rho_in,
                           D_out, chi_out, Dmat_out, chit_out,
                           mu_out, xi_out, lam_out, ltransparm);
```

where `Dmat_out` needs `NUM_SPECIES*NUM_SPECIES` components and `chit_out`
`NUM_SPECIES`; unrequested outputs may be any `Array4`.

### 1.3 Conventions that matter for the flux

With the diffusion driving forces $d_j$ (Section 2), the species mass fluxes
and the Soret contribution are

$$\mathbf{F}_k = -\sum_j \texttt{Dmat}_{kj}\left( \mathbf{d}_j + \chi_j \frac{\nabla T}{T}\right),
\qquad \chi_j = X_j \tilde\chi_j .$$

**Sign convention.** $\tilde\chi$ follows EGlib: light species have
$\chi_k < 0$ and, through the minus sign above, drift towards hot regions. The
mixture-averaged `chi` of `transport.use_soret` follows CHEMKIN's $\Theta_k$,
which is positive for light species and is used by PeleLMeX as
$\mathbf F_k^T = +\rho D_k \Theta_k \nabla T/T$. Same physics, opposite sign of
the coefficient. Do not mix the two, and do not apply PeleLMeX's empirical
factors (0.664 for H$_2$, 0.58 for H) to the multicomponent result; those exist
to correct the mixture-averaged formula.

**Units.** $\texttt{Dmat}$ is in g/(cm s); $\tilde\chi$ is dimensionless;
$\lambda$ in erg/(cm s K). PeleC works in CGS, so no conversion is needed.

---

## 2. Fluxes: current PeleC versus multicomponent

PeleC's species and energy diffusion fluxes are assembled per face in
`Source/Diffterm.H`, functor `SpeciesEnergyFlux`, from the cell-centered
coefficients averaged to the face by `pc_move_transcoefs_to_ec`
(`Source/PeleCUtilities.H`).

**Current (mixture-averaged) form.** With face values of $X_k$, $Y_k$, $h_k$
and the baro-diffusion term,

$$d_k = \frac{\partial X_k}{\partial x} + (X_k - Y_k)\,\frac{\partial \ln p}{\partial x},$$

$$F_k = -\texttt{Ddiag}_k\, d_k, \qquad V_c = \sum_k F_k, \qquad F_k \leftarrow F_k - Y_k V_c,$$

$$F_E = \sum_k h_k F_k - \lambda \frac{\partial T}{\partial x} \;(+\;\text{viscous work}).$$

The correction velocity $V_c$ is required because the mixture-averaged fluxes
do not sum to zero.

**Multicomponent form.** Using the new quantities,

$$F_k = -\sum_j \texttt{Dmat}_{kj}\left( d_j + X_j \tilde\chi_j\, \frac{1}{T}\frac{\partial T}{\partial x}\right),$$

$$F_E = \sum_k h_k F_k - \lambda \frac{\partial T}{\partial x}
      + p \sum_k \chi_k V_k , \qquad V_k = \frac{F_k}{\rho Y_k}.$$

Properties:

* $\sum_k F_k = 0$ holds identically because $\sum_k \texttt{Dmat}_{kj} = 0$; the
  correction velocity step is dropped. This applies to the Soret and
  baro-diffusion parts as well.
* The driving force $d_j$ is already in the mole-fraction form the matrix
  expects, including the pressure term; no $W_k/\overline W$ conversion is
  needed (PeleC differs from PeleLMeX here).
* Soret becomes available in PeleC (it currently aborts if `use_soret` is set)
  and needs no empirical factors.
* The last term in $F_E$ is the Dufour heat flux, the thermodynamically
  consistent partner of Soret; it uses the same $\chi_k$ and is optional in a
  first implementation. $\lambda$ is the LCT conductivity when `lct1`/`lct3`
  is selected, which is the consistent choice when the Dufour term is included.

The current mixture-averaged form is recovered exactly from the matrix form
with `matrix_diffusion_niter = 1` if the projection is skipped; with the
projection it corresponds to the diagonal matrix plus the correction velocity.

---

## 3. Integration steps in PeleC

The changes are confined to the diffusion coefficient evaluation and the face
flux functor. Because PeleC treats diffusion explicitly there is no linear
solver to modify.

### 3.1 Coefficient storage (`Source/IndexDefines.H`)

Append the matrix and $\tilde\chi$ after the existing coefficients:

```cpp
#define dComp_rhoD   0
#define dComp_mu     (dComp_rhoD + NUM_SPECIES)
#define dComp_xi     (dComp_mu + 1)
#define dComp_lambda (dComp_xi + 1)
#define dComp_Dmat   (dComp_lambda + 1)                 // NUM_SPECIES*NUM_SPECIES entries, k*NUM_SPECIES + j
#define dComp_chit   (dComp_Dmat + NUM_SPECIES*NUM_SPECIES)
#define dComp_end    (dComp_chit + NUM_SPECIES)          // total number of coefficients
```

`coeff_cc` and the `GpuArray<Real, dComp_lambda + 1>` arguments of the flux
functors become `dComp_end` long. For 53 species this is 2809 extra doubles
per cell; see Section 5.

### 3.2 Coefficient evaluation (`Source/Diffusion.cpp`, `pc_transcoeff`)

Replace the 14-argument call with the struct API and store the new fields:

```cpp
const bool do_mc = PeleC::use_multicomponent_diffusion;   // new runtime flag, pelec.*
const bool do_soret = PeleC::use_soret;                   // requires transport.conductivity_model = lct1|lct3

amrex::ParallelFor(gbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
  amrex::Real Y[NUM_SPECIES], Ddiag[NUM_SPECIES], chit[NUM_SPECIES];
  amrex::Real Dmat[NUM_SPECIES * NUM_SPECIES];
  amrex::Real mu, xi, lam;
  for (int n = 0; n < NUM_SPECIES; ++n) Y[n] = qar_yin(i, j, k, n);

  pele::physics::transport::TransportRequest req;
  req.mu = req.xi = req.lam = true;
  req.Ddiag = !do_mc;
  req.Dmat  = do_mc;
  req.chit  = do_mc && do_soret;

  pele::physics::transport::TransportOutputs out;
  out.mu = &mu; out.xi = &xi; out.lam = &lam;
  out.Ddiag = Ddiag; out.Dmat = Dmat; out.chit = chit;

  auto trans = pele::physics::PhysicsType::transport();
  trans.transport(req, qar_Tin(i, j, k), qar_rhoin(i, j, k), Y, out, ltransparm);

  coe_mu(i, j, k) = mu; coe_xi(i, j, k) = xi; coe_lambda(i, j, k) = lam;
  if (do_mc) {
    for (int n = 0; n < NUM_SPECIES * NUM_SPECIES; ++n) coe_Dmat(i, j, k, n) = Dmat[n];
    for (int n = 0; n < NUM_SPECIES; ++n) coe_chit(i, j, k, n) = do_soret ? chit[n] : 0.0;
  } else {
    for (int n = 0; n < NUM_SPECIES; ++n) coe_rhoD(i, j, k, n) = Ddiag[n];
  }
});
```

Problem-specific coefficient modifications (`pc_transcoeff` forwards to a
problem hook) would need the same treatment if they touch `Ddiag`.

### 3.3 Face averaging (`pc_move_transcoefs_to_ec`)

Average every `Dmat` entry and `chit` to the face like any other coefficient,
**using the arithmetic mean**. The arithmetic mean preserves the zero column
sums of `Dmat` (and hence mass conservation of the face fluxes); the harmonic
option does not. `chit` can change sign, so it must not go through the
harmonic branch either.

### 3.4 Flux functor (`Source/Diffterm.H`, `SpeciesEnergyFlux`)

A multicomponent variant of the ideal-gas functor:

```cpp
struct SpeciesEnergyFluxMC
{
  AMREX_GPU_DEVICE void operator()(
    const amrex::IntVect iv, const amrex::IntVect ivm, const amrex::Real dxinv,
    const amrex::GpuArray<amrex::Real, dComp_end>& coef,
    const amrex::Array4<const amrex::Real>& q,
    const amrex::Array4<amrex::Real>& flx, const bool do_soret, const bool do_dufour)
  {
    auto eos = pele::physics::PhysicsType::eos();
    amrex::Real mass1[NUM_SPECIES], mass2[NUM_SPECIES], mole1[NUM_SPECIES], mole2[NUM_SPECIES];
    amrex::Real hi1[NUM_SPECIES], hi2[NUM_SPECIES];
    for (int n = 0; n < NUM_SPECIES; ++n) { mass1[n] = q(iv, QFS + n); mass2[n] = q(ivm, QFS + n); }
    eos.Y2X(mass1, mole1);
    eos.Y2X(mass2, mole2);
    eos.T2Hi(q(iv, QTEMP), hi1);
    eos.T2Hi(q(ivm, QTEMP), hi2);

    const amrex::Real pface = 0.5 * (q(iv, QPRES) + q(ivm, QPRES));
    const amrex::Real Tface = 0.5 * (q(iv, QTEMP) + q(ivm, QTEMP));
    const amrex::Real rhoface = 0.5 * (q(iv, QRHO) + q(ivm, QRHO));
    const amrex::Real dlnp = dxinv * (q(iv, QPRES) - q(ivm, QPRES)) / pface;
    const amrex::Real dlnT = dxinv * (q(iv, QTEMP) - q(ivm, QTEMP)) / Tface;

    // Generalized driving forces d_j (+ Soret part)
    amrex::Real d[NUM_SPECIES];
    for (int j = 0; j < NUM_SPECIES; ++j) {
      const amrex::Real Xf = 0.5 * (mole1[j] + mole2[j]);
      const amrex::Real Yf = 0.5 * (mass1[j] + mass2[j]);
      d[j] = dxinv * (mole1[j] - mole2[j]) + (Xf - Yf) * dlnp;
      if (do_soret) d[j] += Xf * coef[dComp_chit + j] * dlnT;   // chi_j = X_j chit_j
    }

    // F_k = - sum_j Dmat_kj d_j ; sum_k F_k = 0 by construction
    amrex::Real dufour = 0.0;
    for (int k = 0; k < NUM_SPECIES; ++k) {
      amrex::Real Fk = 0.0;
      for (int j = 0; j < NUM_SPECIES; ++j) Fk -= coef[dComp_Dmat + k * NUM_SPECIES + j] * d[j];
      flx(iv, UFS + k) = Fk;
      flx(iv, UEDEN) += Fk * 0.5 * (hi1[k] + hi2[k]);
      if (do_dufour) {
        const amrex::Real Xf = 0.5 * (mole1[k] + mole2[k]);
        const amrex::Real Yf = 0.5 * (mass1[k] + mass2[k]);
        if (Yf > 0.0) dufour += Xf * coef[dComp_chit + k] * Fk / (rhoface * Yf);   // chi_k V_k
      }
    }
    if (do_dufour) flx(iv, UEDEN) += pface * dufour;
  }
};
```

No correction-velocity loop. The `-lambda dT/dx` and viscous terms are added
by the caller as today. The embedded-boundary kernel `pc_diffusion_flux_eb`
uses the same functor, so it needs no separate flux logic beyond the enlarged
coefficient array. For the SRK specialization the same structure applies; the
matrix PelePhysics provides for SRK is built from the density-corrected binary
coefficients, so the face-interpolated non-ideal corrections in the current SRK
functor are already accounted for in `Dmat`.

A runtime switch (`pelec.use_multicomponent_diffusion = 1`) can select
`SpeciesEnergyFluxMC` versus the existing functor through `FluxTypes`.

### 3.5 Inputs and setup (`Source/Setup.cpp`, `Source/PeleC.cpp`)

* Add `pelec.use_multicomponent_diffusion` (default 0) and
  `pelec.use_dufour` (default 0).
* Remove the abort on `transport.use_soret` for the multicomponent mode and
  instead require `transport.conductivity_model = lct1` or `lct3` when the
  Soret term is enabled (chit is not available with the mixture model). The
  mixture-averaged `transport.use_soret` fit remains unused by PeleC.
* Typical input block:

```
pelec.use_multicomponent_diffusion = 1
pelec.use_dufour                   = 1
transport.conductivity_model       = lct1
transport.conductivity_niter       = 3
transport.matrix_diffusion_niter   = 2
```

---

## 4. Verification plan

1. **Consistency with the current model.** With
   `transport.matrix_diffusion_niter = 1` and Soret off, the new fluxes equal
   the current mixture-averaged fluxes with correction velocity up to the
   projection (TranEval check (e) verifies the coefficient-level identity).
   A 1D premixed H$_2$/air flame (`Exec/RegTests/PMF`, LiDryer) should give
   nearly identical flame speeds.
2. **Conservation.** $\sum_k F_k$ on every face is zero to round-off without
   the correction velocity; the total mass equation must not drift.
3. **Multicomponent effect.** Compare the H$_2$/air flame speed and the
   H$_2$ and H profiles for `matrix_diffusion_niter = 1, 2, 5` and Soret on/off
   against a Cantera multicomponent solution with Soret.
4. **Coefficient-level checks** are already in PelePhysics' TranEval
   (`inputs.2d_Simple_Matrix`, `inputs.2d_Simple_LCT`): mass conservation,
   symmetry of $\texttt{Dmat}_{kj}/Y_k$, comparison with a direct dense solve,
   antisymmetry of $\chi$, pure-species closed forms.

---

## 5. Costs, limits and caveats

* **Memory.** `coeff_cc` grows by $N^2 + N$ components per cell: 90 for
  LiDryer (9 species), 462 for drm19 (21), 2862 for dodecane_lu (53). Per cell
  the PelePhysics evaluation holds $\mathcal O(N^2)$ doubles on the stack
  (about 12 KB for 21 species, 63 KB for 53), which on GPUs is reserved for
  every resident thread. The matrix path is intended for small to moderate
  mechanisms (roughly $N \lesssim 30$ on GPU).
* **Work.** The flux functor is $\mathcal O(N^2)$ per face instead of
  $\mathcal O(N)$; the coefficient evaluation adds one $N^2$ pass per extra
  `matrix_diffusion_niter` beyond 2 and a few $N^2$ sweeps for `lct1`/`lct3`.
  For $N \le 21$ this is small next to chemistry.
* **Time step.** Diffusion remains explicit; the diffusive CFL applies to the
  coupled matrix as it does to the diagonal today, with the largest eigenvalue
  of `Dmat`/$\rho$ as the relevant diffusivity.
* **Soret sign.** See Section 1.3. The EGlib convention is built into the flux
  formula of Section 2; copying PeleLMeX's `rhotheta` logic would flip the
  sign.
* **Polar species.** The LCT pure-species conductivity differs from the
  mechanism's modified-Eucken fit by up to about 20 % for polar species
  (H$_2$O at 300 K) and by about 1 % for non-polar ones; this is a modeling
  difference between EGlib and CHEMKIN, not a transcription error.
* **Build-time opt-in for LCT.** The LCT conductivity models (and hence Soret
  and Dufour) are compiled only with `USE_LCT_TRANSPORT=TRUE` (GNU make, also
  honored by PeleC's `Make.PeleC`) or `PELE_PHYSICS_ENABLE_LCT_TRANSPORT=ON`
  (CMake), i.e. `-DPELE_TRANSPORT_LCT`; without it `transport.conductivity_model =
  lct1|lct3` is rejected at initialization. The diffusion matrix is only compiled
  into kernels that call the extended API.
* **Downstream build lists.** PeleC's `CMake/BuildPelePhysicsLib.cmake` lists
  transport headers explicitly; add `TransportUtils.H` and
  `SimpleConductivity.H` there (GNUmake picks them up automatically through
  `Source/Transport/Make.package`).

---

## 6. References

* A. Ern and V. Giovangigli, *Multicomponent Transport Algorithms*, Lecture
  Notes in Physics m24, Springer (1994).
* A. Ern and V. Giovangigli, Fast and accurate multicomponent transport
  property evaluation, *J. Comput. Phys.* 120 (1995) 105-116.
* EGlib (Fortran reference used for this implementation): routines `EGSDR1`
  (diffusion matrix), `EGSLCT1`/`EGSLCT3` (conductivity and rescaled thermal
  diffusion ratios), `EGSEVCT`, `LEVEPS`.
* PelePhysics documentation: `Docs/sphinx/Transport.rst`, sections
  "Multicomponent diffusion", "Multicomponent thermal conductivity and thermal
  diffusion ratios" and "Extended transport API".

---

## 7. Differences between the design sketch and the merged code

* LCT models are a build-time opt-in (`USE_LCT_TRANSPORT=TRUE`); `pelec.use_soret`
  aborts at startup if PelePhysics was built without them.
* PeleC detects the extended API through the PelePhysics feature macro
  `PELE_PHYSICS_TRANSPORT_EXTENDED_API` (`TransportTypes.H`) rather than probing
  for a header.
* The species-diffusion time-step limit reuses a Gershgorin diffusivity bound
  stored with the transport coefficients (`dComp_mcDgersh`, reduced to
  `PeleC::mc_specdif_dmax` per level) instead of re-evaluating the matrix in
  `estTimeStep`; the direct evaluation remains as a fallback for the first step.
* `transport.trace_epsilon` defaults to 1e-15, so the default results are bitwise
  identical to the code before the extension.
* The LCT models use Parker's rotational relaxation factor with the
  `(2 + pi^2/4)` coefficient; the bulk viscosity keeps its historical expression
  (`parker_factor_legacy`) so that it is unchanged.
* Requesting `chit` with the mixture conductivity model yields zeros (asserted in
  DEBUG builds) instead of aborting; PeleC's `check_params` still rejects
  `pelec.use_soret` without an LCT model.
* The PMF example inputs no longer carry the 3D-only `DiagFramePlane` block; the
  tests are registered in CI (`add_test_r`) with measured costs.
