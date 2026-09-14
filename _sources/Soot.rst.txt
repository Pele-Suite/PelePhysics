.. highlight:: rst

.. _Soot:

****
Soot
****

Soot Equations
==============

Soot formation, growth, and oxidation is modeled using the hybrid-method of moments (HMOM) model developed by Mueller et al. [#mueller]_. This approach combines the numerical ease of the method of moments with interpolative closure (MOMIC) with the ability to capture the bimodal nature of the soot number density function (NDF) provided by the direct quadrature method of moments (DQMOM). :math:`M_{x,y}` is the moment of the soot NDF, where :math:`x` is the order for volume and :math:`y` for surface area; these terms are modeled according to

.. math::
   M_{x,y} = N_0 V_0^x S_0^y + \exp{\left(\sum_{r=0}^R \sum_{k = 0}^r a_{r,k} x^k y^{r-k}\right)},

where :math:`R` is the order of the polynomial interpolation, :math:`a_{r,k}` are the interpolation coefficients, :math:`N_0` is the weight of the delta function, and :math:`V_0` and :math:`S_0` are the volume and surface area of the nucleated spherical soot particles, respectively. The location of the delta function is fixed at coordinates :math:`V_0` and :math:`S_0` and assumed to be equal to the nucleated particle size.
The nucleated particle volume and surface area are fixed according to :math:`V_0 = 2 W_C C_{\rm{dimer}} / \rho_{\rm{soot}}` and :math:`S_0 = (36 \pi)^{1/3} V_0^{2/3}`, where :math:`W_C` is the molar mass of carbon, :math:`C_{\rm{dimer}}` is the average number of carbon atoms per dimer, and :math:`\rho_{\rm{soot}}` is the density of soot (:math:`\rho_{\rm{soot}} = 1800 {\text{ kg/m}}^3`). If the first-order polynomial interpolation of the moments (:math:`R=1`) is used, the above equation reduces to

.. math::
   M_{x,y} = N_0 V_0^x S_0^y + N_L V_L^x S_L^y,

where :math:`V_L` and :math:`S_L` are the mean volume and surface area of the second mode (large particles). For first-order polynomial interpolation, four transport equations are solved: :math:`M_{0,0}` (number density), :math:`M_{1,0}` (volume fraction, also denoted as :math:`f_v`), :math:`M_{0,1}`, and :math:`N_0`.
The governing equations for the soot moments are [#bisetti]_

.. math::
   \frac{\partial M_{x,y}}{\partial t} + \frac{\partial M_{x,y} \mathbf{u}_g}{\partial \mathbf{X}} = -\frac{\partial \boldsymbol{J}_{M}}{\partial \mathbf{X}} + \dot{M}_{x,y},

where :math:`\boldsymbol{J}_{M}` is the soot mass flux and :math:`\dot{M}_{x,y}` is the soot source term.
The current formulation ignores the molecular diffusion and thermophoretic effects of the soot, making the first term of the right-hand side zero.

For more details regarding the HMOM model, users are encouraged to consult the references cited.

Soot Flags and Inputs
======================

* In the ``GNUmakefile``, specify ``USE_SOOT = TRUE`` and ``NUM_SOOT_MOMENTS = N`` where ``N`` is the number of moments to be used in the solution, either 3 or 6.

* Depending on the gas phase solver, soot solving functionality can be turned on in the input file using ``pelec.add_soot_src = 1`` or ``peleLM.do_soot_solve = 1``.

* The chemistry model specified with ``Chemistry_model =`` in  ``GNUmakefile`` must contain the PAH inception species, as well as ``H2``, ``H``, ``OH``, ``H2O``, ``CO``, ``C2H2``, and ``O2``.

* A PAH inception species must be provided in the input file using ``soot.incept_pah =``. Currently, only one of three inputs are accepted: ``A2``, ``A3``, or ``A4``; which correspond to naphthalene (C10H8), phenathrene (C14H10), or pyrene (C16H10).

  * If the inception species is named something other than ``A#`` in the chemistry model, a different name can be specified using ``soot.pah_name =``. However, ``soot.incept_pah`` must be set to ``A2``, ``A3``, or ``A4``.

.. [#mueller] "Hybrid Method of Moments for modeling soot formation and growth", M. E. Mueller and G. Blanquart and H. Pitsch, Comb. Flame, Vol. 156, No. 6, pp. 1143-1155 (2009)

.. [#bisetti] "On the formation and early evolution of soot in turbulent nonpremixed flames", F. Bisetti and G. Blanquart and M. E. Mueller and H. Pitsch, Comb. Flame, Vol. 159, No. 1, pp. 317-335 (2012)
