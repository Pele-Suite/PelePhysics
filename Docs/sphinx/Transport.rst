.. highlight:: rst

.. _sec:transport:

*********
Transport
*********

PelePhysics supports computation of several transport coefficients: dynamic viscosity (:math:`\mu`), bulk viscosity (:math:`\xi`), thermal conductivity (:math:`\lambda`), mixture-averaged species diffusion coefficients (:math:`D_i`), optionally the full multicomponent diffusion matrix (:math:`\tilde{D}_{ij}`, see :ref:`sec:transport:multicomponent`), and Soret coefficients/thermal diffusion ratio (:math:`\chi_i`). There are three choices of model for computing these transport coefficients:

* ``Constant`` with user-specified values
* ``Sutherland``, adding a simple temperature dependence to user-specified values
* ``Simple`` where transport coefficients are computed based on thermochemistry data for the multi-component mixture
  
The choice between these transport models is made at compile time. When using GNUmake, this is done by setting the ``Transport_Model`` parameter in the ``GNUmakefile``.

The appropriate choice of transport model depends on the EOS being used. For perfect gasses (GammaLaw), constant transport must be used. For reacting flow calculations using either the ideal gas (Fuego) or Soave-Redlich-Kwong equations of state, Simple transport is appropriate in most cases. Note that based on code implementation and physics considerations, the EOS/transport combinations of GammaLaw/Sutherland, GammaLaw/Simple, Soave-Redlich-Kwong/Constant and Soave-Redlich-Kwong/Sutherland are not supported and attempting to compile with any of those combinations will lead to an error message similar to this: ::

    error: static_assert failed due to requirement 'is_valid_physics_combination<pele::physics::eos::SRK,
          pele::physics::transport::ConstTransport>::value' "Invalid physics combination attempted"

.. note:: For the flow solvers in the Pele suite, Soret effects are only supported in PeleLMeX at present. The bulk viscosity is utilized in PeleC but not in the low-Mach algorithm used in PeleLM and PeleLMeX.
	  
Constant
========

In this model, all transport coefficients are user-specified constants. They can be set in a simulation input file using: ::

  transport.const_viscosity = 1.81e-4
  transport.const_bulk_viscosity = 0.0
  transport.const_conductivity = 2.623e2
  transport.const_diffusivity = 0.242
  transport.const_thermal_diffusion_ratio =  0.0
  transport.units = CGS
  
Any unspecified quantity is assumed to be 0.0. The ``transport.units`` parameter indicates the units of the parameters set in the input file and can be ``CGS`` or ``MKS``. If not specified, CGS units are assumed, and regardless of the value of this parameter the output of all transport functions is in CGS units.

Sutherland
==========

This is another minimal model, based on the temperature dependence of viscosity proposed by `Sutherland (1893) <https://doi.org/10.1080/14786449308620508>`_:

.. math::

   \mu = \mu_{ref} \left(\frac{T}{T_{ref}} \right)^{3/2} \frac{T_{ref} + S} {T + S},

where :math:`\mu_{ref}` is the dynamic viscosity at a reference temperature :math:`T_{ref}` and :math:`S` is a constant. In the PelePhysics implementation, the thermal conductivity is then computed based on a user-specified Prandtl number, :math:`\lambda = \mu c_p / Pr`, where the heat capacity :math:`c_p` is evaluated using the EOS model. The user may specify constant values for the bulk viscosity and diffusvity (a single value for all species). Soret effects are not supported for this Transport model (:math:`\chi_i = 0`).

For Sutherland transport, there are several runtime parameters that may be specified in the input file: ::

  transport.Prandtl_number = 0.7
  transport.viscosity_mu_ref = 17.16
  transport.viscosity_T_ref = 273.15
  transport.viscosity_S = 110.4
  transport.const_bulk_viscosity = 0.0
  transport.const_diffusivity = 1.0
  
The values listed above are the defaults for each parameter. Note that the temperature is specified in K and all other dimensional quantities must be specified in CGS units.
  
Simple
======
	  
In this model, transport coefficients are evaluated from data available in the chemical mechanisms (set at compilation using ``Chemistry_Model``). The implementation isbased on that in `EGlib <http://www.cmap.polytechnique.fr/www.eglib/>`_ (see `Ern and Giovangigli (1995) <https://doi.org/10.1006/jcph.1995.1151>`_) and simplified to compute, by default, only mixture-averaged diffusivities for each species. The full multicomponent diffusion matrix is also available on request (see :ref:`sec:transport:multicomponent`). The run-time options are: ::

  transport.use_soret = 0                 # compute Soret coefficients (1) or not (0)
  transport.trace_epsilon = 1.e-16        # shift applied to vanishing species, see below
  transport.matrix_diffusion_niter = 2    # iterations for the multicomponent diffusion matrix

The values listed are the defaults. These options are read only by the Simple model.

All Simple transport evaluations start by replacing the mass fractions with "trace" values that are strictly positive but still sum to one,

.. math::

   Y_i^{tr} = Y_i + \epsilon \left( \frac{1}{N} \sum_k Y_k - Y_i \right), \qquad \epsilon = \texttt{transport.trace\_epsilon},

and deriving the mole fractions from :math:`Y^{tr}`. This keeps every quantity finite for pure components and vanishing species (as in EGlib) and only affects species whose mole fraction is below roughly :math:`\epsilon`. Prior to the introduction of this parameter the shift was hard-coded to :math:`10^{-15}`; setting ``transport.trace_epsilon = 1.e-15`` reproduces the earlier results bitwise.

The mixture-averaged diffusivity returned in ``Ddiag`` is :math:`\rho \, (W_i / \overline{W}) \, D_i^{mix}` with :math:`D_i^{mix} = (1 - Y_i) / \sum_{j \neq i} X_j / \mathcal{D}_{ij}` the Hirschfelder-Curtiss approximation, so that the species mass flux is :math:`\mathbf{F}_i = -\texttt{Ddiag}_i \nabla X_i`.

When Simple transport is used with the Soave-Redlich-Kwong equation of state, additional corrections are used to modify the transport coefficients to account for real gas effects based on `Chung et al. (1988) <https://doi.org/10.1021/ie00076a024>`_. Soret effects are not supported for SRK.

.. _sec:transport:multicomponent:

Multicomponent diffusion
------------------------

In addition to the mixture-averaged diffusivities, the Simple model can return the full multicomponent *flux diffusion matrix*

.. math::

   \texttt{Dmat}_{ij} = \rho \, \tilde{D}_{ij} = \rho \, Y_i \, D_{ij},

where :math:`D_{ij}` is the symmetric multicomponent diffusion matrix of Ern and Giovangigli, such that the species mass fluxes are

.. math::

   \mathbf{F}_i = - \sum_j \texttt{Dmat}_{ij} \nabla X_j .

The matrix is stored row-major, ``Dmat[i*NUM_SPECIES + j]``, in CGS units (g/cm/s), is independent of pressure (like ``Ddiag``), and satisfies :math:`\sum_i \texttt{Dmat}_{ij} = 0` exactly so that the fluxes sum to zero.

The algorithm follows EGlib's ``EGSDR1``: the Stefan-Maxwell matrix :math:`L_{[00]}` is built from the trace-adjusted mole fractions and the binary diffusion coefficients, a projected standard iteration is applied with the splitting that makes the first iterate the Hirschfelder-Curtiss approximation (`Ern and Giovangigli (1995) <https://doi.org/10.1006/jcph.1995.1151>`_), the iterate is scaled by :math:`\rho Y_i` and finally projected, :math:`\tilde{D} \leftarrow P \tilde{D} P` with :math:`P = I - Y \otimes u / \sum_k Y_k`, which enforces mass conservation. The number of iterations is ``transport.matrix_diffusion_niter`` (default 2, the EGlib choice; 1 gives the projected mixture-averaged matrix; larger values converge to the exact matrix at a cost of one :math:`N^3` matrix product per additional iteration). The ``matrix_diffusion_niter = 1`` diagonal coincides with ``Ddiag`` before projection, so the two formulations are consistent.

The matrix is only computed when requested through the extended entry points ::

  transport(wtr_get_xi, wtr_get_mu, wtr_get_lam, wtr_get_Ddiag, wtr_get_chi, wtr_get_Dmat,
            T, rho, Y, Ddiag, chi_mix, Dmat, mu, xi, lam, tparm);
  get_transport_coeffs(bx, Y_in, T_in, Rho_in, D_out, chi_out, Dmat_out, mu_out, xi_out, lam_out, tparm);

where ``Dmat`` has ``NUM_SPECIES*NUM_SPECIES`` entries (``Dmat_out`` as many components). The historical entry points without these arguments are unchanged and carry no additional cost. The ``Constant`` and ``Sutherland`` models return the projected uniform-diffusivity matrix through the same entry points (which yields the same fluxes as their diagonal coefficients), and ``Manifold`` transport aborts. When Simple transport is used with the SRK equation of state, the matrix is built from the density-corrected binary coefficients.

.. note:: Each matrix evaluation holds :math:`\mathcal{O}(N^2)` doubles per thread (about 12 KB for 21 species, 63 KB for 53 species), which is acceptable on GPUs for small to moderate mechanisms (roughly :math:`N \lesssim 30`) but reserves a large amount of local memory for larger ones. The flow solvers in the Pele suite do not yet use the multicomponent matrix; the low-Mach diffusion algorithm in PeleLMeX in particular assumes a diagonal ``Ddiag`` and would require additional algorithm development.

The ``TranEval`` test (``Testing/Exec/TranEval``, input ``inputs.2d_Simple_Matrix``) exercises the matrix path and checks mass conservation, symmetry of :math:`\tilde{D}_{ij}/Y_i`, the exact binary-mixture result, and the error with respect to a direct dense solution of the transport linear system.
