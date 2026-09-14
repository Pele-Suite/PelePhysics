.. highlight:: rst

.. _Spray:

=======
Spray
=======

Spray Equations
===============

This section outlines the modeling and mathematics for the spray routines.
Firstly, spray modeling relies on the following assumptions:

* Dilute spray: droplet volume inside an Eulerian cell is much smaller than the volume of the gas phase; the droplets can be modeled as Lagrangian point source terms relative to the Eulerian gas phase

* Infinite conductivity model: temperature within a droplet is temporally varying but spatially uniform

* One-third rule: the thermophysical properties in the film of an evaporating droplet can be approximated as a weighted average of the state at the droplet surface (weighted as 2/3) and the state of the surrounding gas (weighted as 1/3)

* Ideal equilibrium: the liquid and vapor state at the surface of the droplet are in equilibrium

* The radiation, Soret, and Dufour effects are neglected

Secondly, accurate spray modeling depends on accurate thermophysical and transport properties for both the gas and liquid phases.
Gas-phase properties are obtained directly from the mechanism files in `PelePhysics`, while liquid-phase properties are derived from user-provided inputs for each liquid fuel species.
Currently, two approaches are available for estimating liquid fuel properties:

* The original PeleMP [#owen]_ method, which utilizes a combination of constant values and temperature-based fits.

* A group contribution method (GCM) based on the work of Constantinou & Gani [#gani94]_ [#gani95]_ and Govindaraju & Ihme [#govindaraju]_, previously implemented and validated in `FuelLib <https://github.com/nrel/fuellib>`_.

Further details on the required inputs, as well as the formulations of both component-level and mixture-level liquid-phase properties, are provided in the :ref:`Liquid Spray Properties <SprayLiquidProperties>` section.

The evaporation models follow the work by Abramzon and Sirignano [#abram]_ and the multicomponent evaporation is based on work by Tonini. [#ton]_ Details regarding the energy balance are provided in Ge et al. [#Ge]_
The subscript notation for this section is: :math:`d` relates to the liquid droplet, :math:`v` relates to the vapor state that is in equilibrium with the liquid and gas phase, :math:`L` relates to the liquid phase, and :math:`g` relates to the gas phase. The subscript :math:`r` relates to the reference state with which to approximate the thermophysical and transport properties. This reference state is assumed to be in the evaporating film that surrounds the droplet state and is approximated as

.. math::
   T_r &= T_d + A (T_g - T_d)

   Y_{r,n} &= Y_{v,n} + A (Y_{g,n} - Y_{v,n})

where :math:`A = 1/3` according the the one-third rule.

Typically, the liquid state contains a subset of the species present in the gas phase. However, we also consider generalized capability where the representation of the composition in the liquid phase can be more detailed than in the gas phase. This would occur, for example, when using a gas phase chemical mechanism for a single component surrogate, but using a multi-component representation of the liquid to capture the effects of multi-component vaporization on the volatilization of droplets. In this case, multiple liquid species are modeled with a single gas-phase species, which requires additional assumption and approximation; note that it is still assumed that each liquid species relates to only a single gas-phase species. :math:`N_L` is the number of liquid species, and :math:`N_g` is total number of gas-phase species, and the :math:`N_g \times N_L` mapping from liquid-phase species to gas-phase species is denoted :math:`\mathbf{L}`. The :math:`i{\rm th}` row of :math:`\mathbf{L}` contains :math:`N_{L,i}` ones corresponding to the liquid species that contribute to gas-phase species :math:`i` and zeros elsewhere. There are :math:`N_{pc}` gas species for which :math:`N_{L,i} > 0`, indicating that they participate in phase change. For convenience, we define the following sets of indices:

.. math::
   {\rm Liquid\ species}&: \mathcal{S}_L = \{0, 1, \dots, N_L - 1\}, \\
   {\rm Gas\ species}&: \mathcal{S}_g = \{0, 1, 2, \dots , N_g - 1\}, \\
   {\rm Gas\ species\ w/\ phase\ change} &: \mathcal{S}_{pc} = \{i \in \mathcal{S}_g \; | \; N_{L,i} > 0 \}, \\
   {\rm Liquid\ interacting\ w/\ gas\ species\ } i \in \mathcal{S}_g&: \mathcal{S}_{L,i} = \{n \in \mathcal{S}_L \; | \; \mathbf{L}_{i,n} \neq 0 \}. \\

The :math:`i{\rm th}` row of :math:`\mathbf{L}` contains :math:`N_{L,i}` ones corresponding to the liquid species that contribute to gas-phase species :math:`i` and zeros elsewhere. For example, if we have :math:`N_L = 4` liquid species and :math:`N_g = 3` gas species, where liquid species 0, 1, and 3 contribute to gas-phase species 0, liquid species 2 contributes to gas-phase species 1, and no liquid species contribute to gas-phase species 2, then

.. math::
   \mathbf{L} = \begin{bmatrix}
   1 & 1 & 0 & 1 \\
   0 & 0 & 1 & 0 \\
   0 & 0 & 0 & 0
   \end{bmatrix},

with :math:`N_{L,0} = 3,`  :math:`N_{L,1} = 1,` :math:`N_{L,2} = 0,` and the :math:`N_{pc} = 2` gas-phase species that participate in phase change.
The corresponding sets are:

.. math::
   \mathcal{S}_{L} &= \{0, 1, 2, 3\}, \\
   \mathcal{S}_g &= \{0, 1, 2\}, \\
   \mathcal{S}_{pc} &= \{0, 1\}, \\
   \mathcal{S}_{L,0} &= \{0, 1, 3\}, \\
   \mathcal{S}_{L,1} &= \{2\}, \\
   \mathcal{S}_{L,2} &= \emptyset.

In the PelePhysics implementation, these sets are accessed through matrix operations using the mapping matrix :math:`\mathbf{L}`, which is stored in compressed sparse row (CSR) format.

Additional nomenclature: :math:`M_n` is the molar mass of species :math:`n`, :math:`\overline{M}` is the average molar mass of a mixture, and :math:`\mathcal{R}` is the universal gas constant.  :math:`Y_{g,i}` and :math:`X_{g,i}` are the mass and molar fractions of gas-phase species :math:`i`, and :math:`Y_{d,n}` and :math:`X_{d,n}` are the mass and molar fractions of liquid-phase droplet species :math:`n`, respectively.
The user is required to provide a reference temperature for the liquid properties, :math:`T^*`, the critical temperature for each liquid species, :math:`T_{c,n}`, the boiling temperature for each liquid species at atmospheric pressure, :math:`T^*_{b,n}`, the latent heat and liquid specific heat at the reference temperature, :math:`h_{L,n}(T^*)` and :math:`c_{p,L,n}(T^*)`, respectively.
Note: this reference temperature is a constant value for all species and is not related to the reference state denoted by the subscript :math:`r`.

The equations of motion, mass, momentum, and energy for the Lagrangian spray droplet are:

.. math::
   \frac{d \mathbf{X}_d}{d t} &= \mathbf{u}_d,

   \frac{d m_d}{d t} &= \sum_{n \in \mathcal{S}_L} \dot{m}_n,

   m_d \frac{d Y_{d,n}}{d t} &= \dot{m}_n - Y_{d,n} \frac{d m_d}{d t} \quad \forall\; n \in \mathcal{S}_L,

   m_d \frac{d \mathbf{u}_d}{d t} &= \mathbf{F}_d + m_d \mathbf{g},

   m_d c_{p,L} \frac{d T_d}{d t} &= \sum_{n \in \mathcal{S}_L} \dot{m}_n h_{L,n}(T_d) + \mathcal{Q}_d.

where :math:`\mathbf{X}_d` is the spatial vector, :math:`\mathbf{u}_d` is the velocity vector, :math:`T_d` is the droplet temperature, :math:`m_d` is the mass of the droplet, :math:`\mathbf{g}` is an external body force (like gravity), :math:`\dot{m}` is evaporated mass, :math:`\mathcal{Q}_d` is the heat transfer between the droplet and the surrounding gas, and :math:`\mathbf{F}_d` is the momentum source term.
The density of the liquid mixture, :math:`\rho_d`, depends on the liquid mass fractions of the dropet, :math:`Y_{d,n}`,

.. math::
   \rho_d = \left( \sum_{n \in \mathcal{S}_L} \frac{Y_{d,n}}{\rho_{L,n}} \right)^{-1}

The droplets are assumed to be spherical with diameter :math:`d_d`. Therefore, the mass is computed as

.. math::
   m_d = \frac{\pi}{6} \rho_d d_d^3

The procedure is as follows for updating the spray droplet:

#. Interpolate the gas phase state to the droplet location using a trilinear interpolation scheme.
#. Compute the boiling temperature for liquid species :math:`n` at the current gas phase pressure using the Clasius-Clapeyron relation

   .. math::
      T_{b,n} = \left(\log\left(\frac{p_{\rm{atm}}}{p_g}\right) \frac{\mathcal{R}}{M_n h_{L,n}(T^*_{b,n})} + \frac{1}{T^*_{b,n}}\right)

   The boiling temperature of the droplet is computed as

   .. math::
      T_{d,b} = \sum_{n\in \mathcal{S}_L} Y_{d,n} T_{b,n}

   Since we only have the latent heat at the reference condition temperature, we estimate the enthalpy at the boiling condition using Watson's law

   .. math::
      h_{L,n}(T^*_{b,n}) = h_{L,n}(T^*) \left(\frac{T_{c,n} - T^*}{T_{c,n} - T^*_{b,n}} \right)^{-0.38}

#. Compute the latent heat for liquid species :math:`n` in the droplet using

   .. math::
      h_{L,n}(T_d) = h_{g,i}(T_d) - h_{g,i}(T^*) + h_{L,n}(T^*) - c_{p,L,n}(T^*) (T_d - T^*) \,.

   where :math:`i` is the gas-phase species dependent on liquid species :math:`n` (usually the same, except when multiple liquid species are modeled using the same gas species).

#. Compute the saturation pressure, :math:`p_{{\rm{sat}}, n}`, using one of the methods described in the :ref:`Liquid Spray Properties <SprayLiquidProperties>` section.

#. Estimate the composition of the vapor state using Raoult's law. First, convert from liquid state mass fractions to mole fractions for all :math:`N_L` liquid species and apply Raoult's law to obtain the vapor mole fractions:

   .. math::
      X_{d,n} &= \frac{Y_{d,n}}{M_n}\left(\sum_{k \in \mathcal{S}_L} \frac{Y_{d,k}}{M_k}\right)^{-1} \quad \forall\; n \in \mathcal{S}_L,

      X_{v,n} &= \frac{X_{d,n} p_{{\rm{sat}},n}}{p_g} \quad \forall\; n \in \mathcal{S}_L.

   Then, collapse these mole fractions onto the species available in the gas phase, if needed:

   .. math::
      X_{v,i} = \sum_{n \in \mathcal{S}_{L}} \mathbf{L}_{i,n}X_{v,n} = \sum_{n \in \mathcal{S}_{L,i}} X_{v,n} \quad \forall\; i \in \mathcal{S}_{pc},

   and compute the mass fractions in the vapor state:

   .. math::
      \overline{M}_v &= \sum_{i \in \mathcal{S}_{pc}} X_{v,i} M_i

      X_{v,{\rm{sum}}} &= \sum_{i \in \mathcal{S}_{pc}} X_{v,i}

      Y_{v,i} &= \frac{X_{v,i} M_i}{\overline{M}_v + \overline{M}_g (1 - X_{v,{\rm{sum}}})} \quad \forall\; i \in \mathcal{S}_{pc}.

   If :math:`X_{g,n} p_g > p_{{\rm{sat}},n}`, then :math:`X_{v,n} = Y_{v,n} = 0` for that particular species in the equations above, since that means the gas phase is saturated. For cases with gas species that depend on multiple liquid species, we don't have access to the mole fraction of each liquid species in the gas phase (:math:`X_{g,n}`). Therefore, we estimate it by distributing the gas species mole fraction across all the liquid species on which depends in proportion to the liquid composition:

   .. math::
      X_{g,n} = \frac{X_{d,n}}{\sum_{k \in \mathcal{S}_{L,i}} X_{d,k}} X_{g,i} \quad \forall\; n \in \mathcal{S}_{L,i}, \; i \in \mathcal{S}_{pc} .

   The mass fractions in the reference state for the fuel are computed using the one-third rule and the remaining reference mass fractions are normalized gas phase mass fractions to ensure they sum to 1

   .. math::
      Y_{r,i} = \left\{\begin{array}{c l}
      \displaystyle Y_{v,i} + A (Y_{g,i} - Y_{v,i}) & {\text{if $i \in \mathcal{S}_{pc}$ and $Y_{v,i} > 0$}}, \\
      \displaystyle\frac{1 - \sum_{k \in \mathcal{S}_{pc} | Y_{v,k} > 0 } Y_{r,k}}{1 - \sum_{k \in \mathcal{S}_{pc} | Y_{v,k} > 0 } Y_{g,k}} Y_{g,i} & {\text{otherwise}},
      \end{array}\right. \quad \forall\; i \in \mathcal{S}_g.

#. The average molar mass, specific heat, and density of the reference state in the gas film are computed as

   .. math::
      \overline{M}_r &= \left(\sum_{i \in \mathcal{S}_g} \frac{Y_{r,i}}{M_i}\right)^{-1},

      c_{p,r} &= \sum_{i \in \mathcal{S}_g} Y_{r,i} c_{p,g,i}(T_r),

      \rho_r &= \frac{\overline{M}_r p_g}{\mathcal{R} T_r}.

#. Transport properties are computed using the reference state: dynamic viscosity, :math:`\mu_r`, thermal conductivity, :math:`\lambda_r`, and mass diffusion coefficient for species :math:`n`, :math:`D_{r,n}`.

#. It is important to note that `PelePhysics` provides mixture averaged mass diffusion coefficient :math:`\overline{(\rho D)}_{r,n}`, which is converted into the binary mass diffusion coefficient using

   .. math::
      (\rho D)_{r,i} = \overline{(\rho D)}_{r,i} \overline{M}_r / M_i \quad \forall\; i \in \mathcal{S}_{pc}.

   Mass diffusion coefficient is then normalized by the total fuel vapor molar fraction

   .. math::
      (\rho D)^*_{r,i} = \frac{X_{v,i} (\rho D)_{r,i}}{X_{v,{\rm{sum}}}} \; \forall\; i \in \mathcal{S}_{pc},

   which can be consistently distributed across liquid species in the many-to-one case:

   .. math::
      (\rho D)^*_{r,n} = \frac{X_{v,n}}{X_{v,i}} (\rho D)^*_{r,i}  \quad \forall n \in \mathcal{S}_{L,i} \quad \forall\; i \in \mathcal{S}_{pc}

   (further investigation needed to determine if molecular weight scaling is also needed here). The total is

   .. math::
      (\rho D)_r = \sum_{i \in \mathcal{S}_{pc}} (\rho D)_{r,i}^*.

#. The momentum source is a function of the drag force

   .. math::
      \mathbf{F}_d = \frac{1}{2} \rho_r C_D A_d \left\|\Delta \mathbf{u}\right\| \Delta \mathbf{u}

   where :math:`\Delta \mathbf{u} = \mathbf{u}_g - \mathbf{u}_d`, :math:`A_d = \pi d_d^2/4` is the frontal area of the droplet, and :math:`C_D` is the drag coefficient for a sphere, which is estimated using the standard drag curve for an immersed sphere

   .. math::
      C_D = \frac{24}{{\rm{Re}}_d}\left\{\begin{array}{c l}
      1 & {\text{If Re$_d$ < 1}}, \\
      \displaystyle 1 + \frac{{\rm{Re}}^{2/3}_d}{6} & {\text{Otherwise}}.
      \end{array}\right.

   The droplet Reynolds number is defined as

   .. math::
      {\rm{Re}}_d = \frac{\rho_r d_d \left\|\Delta \mathbf{u}\right\|}{\mu_r}


#. The mass source term is modeled according to Abramzon and Sirignano (1989). The following non-dimensional numbers and factors are used:

   .. math::
      F(B) &= (1 + B)^{0.7}\frac{\log(1 + B)}{B}

      F_2 &= \max(1, \min(400, {\rm{Re}}_d)^{0.077})

      {\rm{Pr}}_r &= \frac{\mu_r c_{p,r}}{\lambda_r}

      {\rm{Sc}}_r &= \frac{\mu_r}{(\rho D)_r}

      {\rm{Sh}}_0 &= 1 + (1 + {\rm{Re}}_d {\rm{Sc}}_r)^{1/3} F_2

      {\rm{Nu}}_0 &= 1 + (1 + {\rm{Re}}_d {\rm{Pr}}_r)^{1/3} F_2

      {\rm{Sh}}^* &= 2 + \frac{{\rm{Sh}}_0 - 2}{F(B_M)}

      {\rm{Nu}}^* &= 2 + \frac{{\rm{Nu}}_0 - 2}{F(B_T)}

   * The Spalding numbers for mass transfer, :math:`B_M`, and heat transfer, :math:`B_T`, are computed using

     .. math::
        B_M &= \displaystyle\frac{\sum_{i \in \mathcal{S}_{pc} | Y_{v,i} > 0 } Y_{v,i} - \sum_{i \in \mathcal{S}_{pc} | Y_{v,i} > 0 } Y_{g,i}}{1 - \sum_{i \in \mathcal{S}_{pc}| Y_{v,i} > 0}  Y_{v,i}}

        B_T &= \left(1 + B_M\right)^{\phi} - 1

     where

     .. math::
        \phi = \frac{c_{p,r}^f (\rho D)_r {\rm{Sh}}^*}{\lambda_r {\rm{Nu}}^*}

     and :math:`c_{p,r}^f` is the heat capacity of the fuel at the reference/skin state temperature. Note the dependence of :math:`{\rm{Nu}}^*` on :math:`B_T` means an iterative scheme is required to solve for both. The droplet vaporization rate and heat transfer become

     .. math::
        \dot{m}_n &= -\pi (\rho D)_{r,n}^* d_d {\rm{Sh}}^* \log(1 + B_M). \; \forall\; n \in \mathcal{S}_L,

        \mathcal{Q}_d &= \pi \lambda_r d_d (T_g - T_d) {\rm{Nu}}^* \frac{\log(1 + B_T)}{B_T}

   * If the gas phase is saturated for all liquid species, the equations for heat and mass transfer become

     .. math::
        \dot{m}_n &= 0

        \mathcal{Q}_d &= \pi \lambda_r d_d (T_g - T_d) {\rm{Nu}}_0

#. To alleviate conservation issues at AMR interfaces, each parcel only contributes to the gas phase source term of the cell containing it. The gas phase source terms for a single parcel to the cell are

    .. math::
       S_{\rho} &= \mathcal{C} \sum_{n \in \mathcal{S}_L} \dot{m}_n,

       S_{\rho Y_i} &= \mathcal{C} \sum_{n \in \mathcal{S}_{L,i}} \dot{m}_n \quad \forall\; i \in \mathcal{S}_{pc},

       \mathbf{S}_{\rho \mathbf{u}} &= \mathcal{C} \mathbf{F}_d,

       S_{\rho h} &= \mathcal{C}\left(\mathcal{Q}_d + \sum_{n \in \mathcal{S}_L} \dot{m}_n h_{g,n}(T_d)\right),

       S_{\rho E} &= S_{\rho h} + \frac{1}{2}\left\|\mathbf{u}_d\right\| S_{\rho} + \mathcal{C} \mathbf{F}_d \cdot \mathbf{u}_d

    where

    .. math::
       \mathcal{C} = -\frac{N_{d}}{V_{\rm{cell}}},

    :math:`N_{d}` is the number of droplets per computational parcel, and :math:`V_{\rm{cell}}` is the volume for the cell of interest. Note that the cell volume can vary depending on AMR level and if an EB is present.


.. warning::

   **Dimensional Assumptions in the Spray Model**

   The spray model assumes spherically symmetric droplets and is inherently three-dimensional.
   For two-dimensional simulations, the domain is treated as one cell wide in the :math:`z`-direction,
   with :math:`L_z = \Delta x`, so the volume in the gas-phase source terms is:

   .. math::

      V_{\rm{cell}} = \Delta x \, \Delta y \, \Delta x

   For nominally single droplet cases, this effectively places an infinite array of droplets spaced :math:`\Delta x` apart in :math:`z`,
   exaggerating Stefan flow in the :math:`x`- and :math:`y`-directions and omitting flow in :math:`z`. While this has minimal impact on
   droplet diameter or temperature, it can distort the surrounding gas-phase flow, especially for low :math:`\text{Re}_d`.

   For such cases, fully three-dimensional simulations are recommended.


Spray Equations with Manifold Models
====================================

The Spray implementation for Manifold-based combustion models (i.e., in the gas phase reduced-dimensional manifold parameters are transported rather than species) is meant to follow that of the implementation and assumptions described above for simulations using finite rate gas phase chemistry mechanisms. However, a major additional assumption introduced for the first pass at this implementation is to substantially neglect evolution of the energy equation. Instead, it is assumed that the enthalpy deficit in the gas phase due to enthalpy of vaporization is accounted for in the boundary conditions used to generate the reduced-order manifold that defines gas phase properties. For the manifold model implementation, the liquid phase remains represented by the same set of liquid species as in the finite rate chemistry implementation, and the liquid phase equations of motion, mass, momentum, and energy remain unchanged from those listed above. However, because the gas phase now involved transport of manifold parameters rather than species, the implementation must be adjusted to properly estimate the gas phase composition near the droplets for the purpose of computing phase quilibria and phase change rates. The source terms must also be appropriately transformed from species source terms to manifold parameter source terms for coupling to the gas phase.

The manifold parameters are denoted as :math:`\xi_k` and are defined as linear combinations of species  :math:`\xi_k = W_{ki} Y_i`. Gas phase properties may be obtained as functions of the Manifold, :math:`\phi = \mathcal{F}(\xi_k)` where :math:`\phi = (\bar{M}, Y_i, T, ...)`. Similarly to detailed chemistry, we allow that multiple liquid phase species may contribute to a single species that can be extracted from the manifold.

The modified procedure for Manifold-based gas phase chemistry is as follows for updating the spray droplet:

#. Unchanged from Step 1 above.

#. Unchanged from Step 2 above.

#. The Manifold model does not provide information about species enthalpies. However, the enthalpy of vaporization term is not directly required in the present implementation, so this step to compute :math:`h_{L,n}(T_d)` is omitted.

#. Only Antoine-based computation of :math:`p_{\text{sat}}` can be used for the gas phase, as the Clausius-Clapeyron relationship requires :math:`h_{L,n}(T_d)`, which was not computed in the previous step.

#. Estimate the vapor state using Raoult's law. Note that the vapor state must be estimated in terms of Manifold parameters, rather than species. First, vapor-liquid equilibrium calculations are used to compute values in terms of species in the same manner as for detailed chemistry, but these are then converted to values in terms of Manifold parameters.

   .. math::
      X_{d,n} &= \frac{Y_{d,n}}{M_n}\left(\sum_{k \in \mathcal{S}_L} \frac{Y_{d,k}}{M_k}\right)^{-1} \quad \forall\; n \in \mathcal{S}_L.

      X_{v,n} &= \frac{X_{d,n} p_{{\rm{sat}},n}}{p_g} \quad \forall\; n \in \mathcal{S}_L.

   Then, collapse these mole fractions onto the species available in the gas phase, if needed:

   .. math::
      X_{v,i} = \sum_{n \in \mathcal{S}_{L,i}} X_{v,n} \quad \forall\; i \in \mathcal{S}_{pc},

   and compute the mass fractions in the vapor state:

   .. math::
      \overline{M}_v &= \sum_{i \in \mathcal{S}_{pc}} X_{v,i} M_i

      X_{v,{\rm{sum}}} &= \sum_{i \in \mathcal{S}_{pc}} X_{v,i}

      Y_{v,i} &= \frac{X_{v,i} M_i}{\overline{M}_v + \overline{M}_g (1 - X_{v,{\rm{sum}}})} \quad \forall\; i \in \mathcal{S}_{pc},

   noting that :math:`\overline{M}_g`, the mean molecular weight in the gas phase, must be available as one of the :math:`\phi` that can be evaluated from the manifold, and that molecular weights
   for the gas phase species in :math:`\mathcal{S}_{pc}` must also be specified as part of the manifold model. With these vapor phase composition, we could compute the reference state compositions in the
   same manner as for detailed chemistry, but in order to evaluate properties at the reference state, we require the composition to be projected onto the manifold. To do this, we start from the
   detailed species representation of the reference state:

   .. math::
      Y_{r,i} = \left\{\begin{array}{c l}
      \displaystyle Y_{v,i} + A (Y_{g,i} - Y_{v,i}) & {\text{if $i \in \mathcal{S}_{pc}$ and $Y_{v,i} > 0$}}, \\
      \displaystyle\frac{1 - \sum_{k \in \mathcal{S}_{pc} | Y_{v,k} > 0 } Y_{r,k}}{1 - \sum_{k \in \mathcal{S}_{pc} | Y_{v,k} > 0 } Y_{g,k}} Y_{g,i} & {\text{otherwise}},
      \end{array}\right. \quad \forall\; i \in \mathcal{S}_g.

   and note that :math:`\xi_i = W_{ij} Y_j`. We note that :math:`Y_{g,i} \: \forall\; i \in \mathcal{S}_g` can be evaluated from the manifold, but to reduce the number of variables that must be
   stored/evaluated in the manifold model, we can make some simplifications so we only have to evaluate :math:`Y_{g,i} \: \forall\; i \in \mathcal{S}_{pc}`. We define

   .. math::
      Y^{pc}_{g,i} = \left\{\begin{array}{c l}
      \displaystyle Y_{g,i} & {\text{if $i \in \mathcal{S}_{pc}$ and $Y_{v,i} > 0$}}, \\
      \displaystyle 0.0 & {\text{otherwise}},
      \end{array}\right. \quad \forall\; i \in \mathcal{S}_g.

   and its complement :math:`Y^{nc}_{g,i}` such that :math:`Y^{pc}_{g,i} + Y^{nc}_{g,i} = Y_{g,i}`, as well as similar definitions for :math:`Y^{nc}_{r,i}, Y^{pc}_{r,i},
   Y^{nc}_{v,i}, \text{and } Y^{pc}_{v,i}`. With these definitions,

   .. math::
      Y^{pc}_{r,i} &=  Y^{pc}_{v,i} + A (Y^{pc}_{g,i} - Y^{pc}_{v,i}), \\
      Y^{nc}_{r,i} &= \theta Y^{nc}_{g,i}, \\
      \text{where } \theta &= \frac{1 - \sum_{k \in \mathcal{S}_{pc} | Y_{v,k} > 0 } Y_{r,k}}{1 - \sum_{k \in \mathcal{S}_{pc} | Y_{v,k} > 0 } Y_{g,k}}
                            = \frac{1 - \sum_{k \in \mathcal{S}_{pc} | Y_{v,k} > 0 } Y^{pc}_{r,k}}{1 - \sum_{k \in \mathcal{S}_{pc} | Y_{v,k} > 0 } Y^{pc}_{g,k}}


   Therefore, in order to compute :math:`\xi_{r,k} = W_{ki} Y_{r,i}` from the available data :math:`(\xi_{g,k}, Y^{pc}_{g,i}(\xi_{g,k}), \text{and } Y^{pc}_{v,i} )`, we note

   .. math::
      \xi_{r,k} &= W_{ki} (Y^{pc}_{r,i} + Y^{nc}_{r,i}) = W_{ki} Y^{pc}_{r,i} + W_{ki} \theta Y^{nc}_{g,i}, \text{and} \\
      \xi_{g,k} &= W_{ki} (Y^{pc}_{g,i} + Y^{nc}_{g,i}) \therefore W_{ki} Y^{nc}_{g,i} = \xi_{g,j} - W_{ki} Y^{pc}_{g,i}.

   We can therefore compute :math:`\xi_{r,k}` from only available data using:

   .. math::
      \xi_{r,k} =  W_{ki} Y^{pc}_{r,i} + \theta (\xi_{g,k} - W_{ki} Y^{pc}_{g,i} ).

   The present implementation specializes to the case (common in combustion modeling) where the manifold parameters :math:`\xi_k` are either progress variable like :math:`(W_{ki}=0 \; \forall\; i \in \mathcal{S}_{pc})`
   or mixture fraction like (:math:`W_{ki}=1` for one :math:`i \in \mathcal{S}_{pc}` and :math:`0` for all others).

#. Proceeds as in the ideal gas implementation, but :math:`\bar{M}_r` and :math:`\rho_r` are computed as functions of the manifold :math:`\phi(\xi_k))`. However, we note that the computed reference temperature
   :math:`T_r^* = \phi(\xi_{k,r})` may deviate significantly from the nominal reference temperature :math:`T_r = T_d + A (T_g - T_d)`. To correct for this, the density obtained from the table is rescaled
   using an ideal gas law assumption:

   .. math::

      \rho_r = \rho_r^* \frac{T_r^*}{T_r}

   where :math:`\rho_r^*` is the value obtained from the manifold model. Heat capacities :math:`c_{P,r}` and :math:`c^f_{P,r}` are obtained from the manifold model without correction due to the relatively week dependence on temperature for heat capacity.

#. Again, :math:`\mu_r`, :math:`\lambda_r`, and :math:`\rho D_{r,n}`  are computed as functions of the manifold. Due to the temperature discrepancy noted in the previous step, a rescaling based on Sutherland's law is performed for all the transport coefficients, shown here for :math:`\mu_r`:

   .. math::

      \mu_r = \mu_r^* \left(\frac{T_r}{T_r^*}\right)^{3/2} \frac{T_r^* + S}{T_r + S}

   where :math:`{}^*` indicates a quantity obtained from the manifold model :math:`\phi(\xi_k)`.


#. Diffusion coefficient modification proceeds as in the detailed chemistry case.

#. Momentum source term computation proceeds as in the detailed chemistry case.

#. Mass source term proceeds as in the detailed chemistry case. Energy source term is ignored.

#. Gas phase source terms follow the detailed chemistry implementation, except that the energy/enthalpy equations are ignored and

   .. math::
      S_{\rho \xi_k} = W_{ki} S_{\rho Y_i} = W_{ki} \mathcal{C} \sum_{n \in \mathcal{S}_{L,i}} \dot{m}_n \quad \forall\; i \in \mathcal{S}_{pc},



Spray Flags and Inputs
======================

* In the ``GNUmakefile``, specify ``USE_PARTICLES = TRUE`` and ``SPRAY_FUEL_NUM = N`` where ``N`` is the number of liquid species being used in the simulation.

* Depending on the gas phase solver, spray solving functionality can be turned on in the input file using ``pelec.do_spray_particles = 1`` or ``peleLM.do_spray_particles = 1``.

* There are many required ``particles.`` flags in the input file. For demonstration purposes, 2 liquid species of ``NC7H16`` and ``NC10H22`` will be used.

  * The liquid fuel species names are specified using ``particles.fuel_species = NC7H16 NC10H22``. The number of fuel species listed must match ``SPRAY_FUEL_NUM``.

  * Although this is not required or typical, if the evaporated mass should contribute to a different gas-phase species than what is modeled in the liquid phase, use ``particles.dep_fuel_species``. For example, if we wanted the evaporated mass from both liquid species to contribute to a different species called ``SP3``, we would put ``particles.dep_fuel_species = SP3 SP3``. All species specified must be present in the chemistry transport and thermodynamic data.

* The following table lists other inputs related to ``particles.``

.. table::

   +------------------------+-------------------------------+-------------+-------------------+
   |Input                   |Description                    |Required     |Default Value      |
   +========================+===============================+=============+===================+
   |``fuel_species``        |Names of liquid species        |Yes          |None               |
   +------------------------+-------------------------------+-------------+-------------------+
   |``dep_fuel_species``    |Name of gas-phase species to   |No           |Inputs to          |
   |                        |contribute                     |             |``fuel_species``   |
   +------------------------+-------------------------------+-------------+-------------------+
   |``dep_manifold_species``|Name of manifold parameter to  |Yes          |None               |
   |                        |contribute (Manifold EOS only) |             |                   |
   +------------------------+-------------------------------+-------------+-------------------+
   |``mom_transfer``        |Couple momentum with gas phase |No           |``1``              |
   |                        |                               |             |                   |
   +------------------------+-------------------------------+-------------+-------------------+
   |``mass_transfer``       |Evaporate mass and exchange    |No           |``1``              |
   |                        |heat with gas phase            |             |                   |
   +------------------------+-------------------------------+-------------+-------------------+
   |``fixed_parts``         |Fix particles in space         |No           |``0``              |
   +------------------------+-------------------------------+-------------+-------------------+
   |``parcel_size``         |:math:`N_{d}`; Number of       |No           |``1.``             |
   |                        |droplets per parcel            |             |                   |
   +------------------------+-------------------------------+-------------+-------------------+
   |``write_ascii_files``   |Output ascii files of spray    |No           |``0``              |
   |                        |data                           |             |                   |
   +------------------------+-------------------------------+-------------+-------------------+
   |``cfl``                 |Particle CFL number for        |No           |``0.5``            |
   |                        |limiting time step             |             |                   |
   +------------------------+-------------------------------+-------------+-------------------+
   |``init_file``           |Ascii file name to initialize  |No           |Empty              |
   |                        |droplets                       |             |                   |
   +------------------------+-------------------------------+-------------+-------------------+


.. _SprayLiquidProperties:

Liquid Spray Properties
-----------------------

The required inputs and corresponding correlations for the original *PeleMP* [#owen]_
and the *FuelLib-based GCM* are outlined in the subsections below. Please note the following details:

**Units:**

* `PeleLM` and `PeleLMeX` use MKS units, while `PeleC` uses CGS units. The Spray inputs follow the same convention.

* For example, when running a spray simulation coupled with `PeleC`, the values for ``particles.fuel_cp`` must be provided in erg/g.

**Input flags:**

* A number of ``particles.`` flags are required in the input file to define liquid fuel properties.

* For demonstration purposes, two liquid species will be used: ``NC7H16`` and ``NC10H22``.

* Many values must be specified on a per-species basis. In this example, one would need to specify:

   - ``particles.NC7H16_crit_temp = 540`` critical temperature of 540 K for ``NC7H16``

   - ``particles.NC10H22_crit_temp = 617`` critical temperatures of 617 K for ``NC10H22``.

**Additional method-specific inputs:**

   * The following tables list other required inputs related to ``particles.``,  where ``SP`` refers to a given fuel species name.

The source code for the liquid spray properties can be found in ``SprayProperties.H``.

PeleMP Implementation
~~~~~~~~~~~~~~~~~~~~~

.. table::

   +-----------------------+-------------------------------+-------------+-------------------+
   |Input                  |Description                    |Required     |Default Value      |
   +=======================+===============================+=============+===================+
   |``fuel_ref_temp``      |Liquid reference temperature   |Yes          |None               |
   +-----------------------+-------------------------------+-------------+-------------------+
   |``SP_crit_temp``       |Critical temperature           |Yes          |None               |
   +-----------------------+-------------------------------+-------------+-------------------+
   |``SP_boil_temp``       |Boiling temperature at         |Yes          |None               |
   |                       |atmospheric pressure           |             |                   |
   +-----------------------+-------------------------------+-------------+-------------------+
   |``SP_cp``              |Liquid :math:`c_p` at reference|Yes          |None               |
   |                       |temperature                    |             |                   |
   +-----------------------+-------------------------------+-------------+-------------------+
   |``SP_latent``          |Latent heat at reference       |Yes          |None               |
   |                       |temperature                    |             |                   |
   +-----------------------+-------------------------------+-------------+-------------------+
   |``SP_rho``             |Liquid density                 |Yes          |None               |
   |                       |                               |             |                   |
   +-----------------------+-------------------------------+-------------+-------------------+
   |``SP_lambda``          |Liquid thermal conductivity    |No           |0.                 |
   |                       |                               |             |                   |
   +-----------------------+-------------------------------+-------------+-------------------+
   |``SP_mu``              |Liquid dynamic viscosity       |No           |0.                 |
   |                       |                               |             |                   |
   +-----------------------+-------------------------------+-------------+-------------------+
   |``SP_sigma``           |Liquid surface tension         |No           |0.                 |
   |                       |                               |             |                   |
   +-----------------------+-------------------------------+-------------+-------------------+


* If an Antoine fit for saturation pressure is used, it must be specified for individual species, ::

    particles.SP_psat = 4.07857 1501.268 -78.67 1.E5

  where the numbers represent coefficients :math:`a_n`, :math:`b_n`, :math:`c_n`, and unit conversion :math:`d_n`, respectively in:

  .. math::
     p_{\rm{sat},n}(T) = d_n 10^{a_n - b_n / (T + c_n)}.

  * If no fit is provided, the saturation pressure is estimated using the Clasius-Clapeyron relation

      .. math::
         p_{{\rm{sat}}, n} = p_{\rm{atm}} \exp\left(\frac{h_{L,n}(T_d) M_n}{\mathcal{R}} \left(\frac{1}{T^*_{b,n}} - \frac{1}{T_d}\right)\right)

* Temperature based fits for liquid density, thermal conductivity, and dynamic viscosity can be used; these can be specified as ::

    particles.SP_rho = 10.42 -5.222 1.152E-2 4.123E-7
    particles.SP_lambda = 7.243 1.223 4.223E-8 8.224E-9
    particles.SP_mu = 7.243 1.223 4.223E-8 8.224E-9

  where the numbers represent :math:`a_n`, :math:`b_n`, :math:`c_n`, and :math:`d_n`, respectively in:

  .. math::
     \rho_{L,n} \,, \lambda_{L,n} = a_n + b_n T + c_n T^2 + d_n T^3

     \mu_{L,n} = a_n + b_n / T + c_n / T^2 + d_n / T^3

  If only a single value is provided, :math:`a_n` is assigned to that value and the other coefficients are set to zero, effectively using a constant value for the parameters.


FuelLib-Based GCM
~~~~~~~~~~~~~~~~~

Currently the *GCM* approach of estimating liquid fuel properties is only available in PeleLMeX and requires:

* Setting the compile-time flag ``SPRAY_GCM=TRUE`` in the case's ``GNUmakefile``

* Generating a liquid-fuel-specific GCM input file from FuelLib, and copying the input file into the case directory.

   - The process for generating this input file is provided in FuelLib's tutorial: `Exporting Properties for Pele <https://nrel.github.io/FuelLib/tutorials-export4pele.html>`_.

   - An example of using the GCM in Pele is provided in ``PeleLMeX/Exec/RegTests/SingleDropEvap``.

The following inputs are generated from FuelLib for each liquid fuel species.

.. table::

   +------------------------+-------------------------------+-------------+
   |Input                   |Description                    |Required     |
   +========================+===============================+=============+
   |``SP_family``           |Compound family                |Yes          |
   +------------------------+-------------------------------+-------------+
   |``SP_molar_weight``     |Molecular weight               |Yes          |
   +------------------------+-------------------------------+-------------+
   |``SP_crit_temp``        |Critical temperature           |Yes          |
   +------------------------+-------------------------------+-------------+
   |``SP_crit_press``       |Critical pressure              |Yes          |
   +------------------------+-------------------------------+-------------+
   |``SP_crit_vol``         |Critical volume                |Yes          |
   +------------------------+-------------------------------+-------------+
   |``SP_boil_temp``        |Boiling temperature at         |Yes          |
   |                        |atmospheric pressure           |             |
   +------------------------+-------------------------------+-------------+
   |``SP_accentric_factor`` |Critical temperature           |Yes          |
   +------------------------+-------------------------------+-------------+
   |``SP_molar_vol``        |Molecular volume               |Yes          |
   +------------------------+-------------------------------+-------------+
   |``SP_cp_a``,            |GCM coefficients for specific  |Yes          |
   |``SP_cp_c``,            |heat                           |             |
   |``SP_cp_c``             |                               |             |
   +------------------------+-------------------------------+-------------+
   |``SP_latent``           |Latent heat at 298.15 K        |Yes          |
   +------------------------+-------------------------------+-------------+

The specific equations, correlations and mixture rules used in the GCM implementation are detailed in the `Fuel Property Prediction Model <https://nrel.github.io/FuelLib/fuelprops.html>`_ section of FuelLib's documentation.


Spray Injection
---------------

Templates to facilitate and simplify spray injection are available. To use them, changes must be made to the input and ``SprayParticlesInitInsert.cpp`` files. Inputs related to injection use the ``spray.`` parser name. To create a jet in the domain, modify the ``InitSprayParticles()`` function in ``SprayParticleInitInsert.cpp``. Here is an example: ::

  void
  SprayParticleContainer::InitSprayParticles(
  const bool init_parts)
  {
    int num_jets = 1;
    m_sprayJets.resize(num_jets);
    std::string jet_name = "jet1";
    m_sprayJets[0] = std::make_unique<SprayJet>(jet_name, Geom(0));
    return;
  }


This creates a single jet that is named ``jet1``. This name will be used in the input file to reference this particular jet. For example, to set the location of the jet center for ``jet1``, the following should be included in the input file, ::

  spray.jet1.jet_cent = 0. 0. 0.

No two jets may have the same name. If an injector is constructed using only a name and geometry, the injection parameters are read from the input file. Here is a list of injection related inputs:

.. table::
   :widths: 20 40 20

   +--------------------+--------------------------------+--------------------+
   |Input               |Description                     |Required            |
   |                    |                                |                    |
   +====================+================================+====================+
   |``jet_cent``        |Jet center location             |Yes                 |
   +--------------------+--------------------------------+--------------------+
   |``jet_norm``        |Jet normal direction            |Yes                 |
   +--------------------+--------------------------------+--------------------+
   |``jet_vel``         |Jet velocity magnitude          |Yes                 |
   +--------------------+--------------------------------+--------------------+
   |``jet_dia``         |Jet diameter                    |Yes                 |
   +--------------------+--------------------------------+--------------------+
   |``spread_angle``    |:math:`\theta_J`; Full spread   |Yes                 |
   |                    |angle in degrees from the jet   |                    |
   |                    |normal direction; droplets vary |                    |
   |                    |from                            |                    |
   |                    |:math:`[-\theta_J/2,\theta_J/2]`|                    |
   +--------------------+--------------------------------+--------------------+
   |``T``               |Temperature of the injected     |Yes                 |
   |                    |liquid                          |                    |
   +--------------------+--------------------------------+--------------------+
   |``Y``               |Mass fractions of the injected  |Yes, if             |
   |                    |liquid based on                 |``SPRAY_FUEL_NUM`` >|
   |                    |``particles.fuel_species``      |1                   |
   +--------------------+--------------------------------+--------------------+
   |``mass_flow_rate``  |:math:`\dot{m}_{\rm{inj}}`; Mass|Yes                 |
   |                    |flow rate of the jet            |                    |
   +--------------------+--------------------------------+--------------------+
   |``hollow_spray``    |Sets hollow cone injection with |No (Default: 0)     |
   |                    |angle :math:`\theta_J/2`        |                    |
   +--------------------+--------------------------------+--------------------+
   |``hollow_spread``   |:math:`\theta_h`; Adds spread to|No (Default: 0)     |
   |                    |hollow cone :math:`\theta_J/2\pm|                    |
   |                    |\theta_h`                       |                    |
   +--------------------+--------------------------------+--------------------+
   |``swirl_angle``     |:math:`\phi_S`; Adds a swirling |No (Default: 0)     |
   |                    |component along azimuthal       |                    |
   |                    |direction                       |                    |
   +--------------------+--------------------------------+--------------------+
   |``start_time`` and  |Beginning and end time for jet  |No                  |
   |``end_time``        |                                |                    |
   +--------------------+--------------------------------+--------------------+
   |``dist_type``       |Droplet diameter distribution   |Yes                 |
   |                    |type: ``Uniform``, ``Normal``,  |                    |
   |                    |``LogNormal``, ``Weibull``,     |                    |
   |                    |``ChiSquared``                  |                    |
   +--------------------+--------------------------------+--------------------+


.. figure:: /Visualization/inject_transform.png
   :align: center
   :figwidth: 60%

   Demonstration of injection angles. :math:`\phi_J` varies uniformly from :math:`[0, 2 \pi]`


Care must be taken to ensure the amount of mass injected during a time step matches the desired mass flow rate. For smaller time steps, the risk of over-injecting mass increases. To mitigate this issue, each jet accounts for three values: :math:`N_{P,\min}`, :math:`m_{\rm{acc}}`, and :math:`t_{\rm{acc}}` (labeled in the code as ``m_minParcel``, ``m_sumInjMass``, and ``m_sumInjTime``, respectively). :math:`N_{P,\min}` is the minimum number of parcels that must be injected over the course of an injection event; this must be greater than or equal to one. :math:`m_{\rm{acc}}` is the amount of uninjected mass accumulated over the time period :math:`t_{\rm{acc}}`. The injection routine steps are as follows:

#. The injected mass for the current time step is computed using the desired mass flow rate, :math:`\dot{m}_{\rm{inj}}` and the current time step

   .. math::
      m_{\rm{inj}} = \dot{m}_{\rm{inj}} \Delta t + m_{\rm{acc}}

#. The time period for the current injection event is computed using

   .. math::
      t_{\rm{inj}} = \Delta t + t_{\rm{acc}}

#. Using the average mass of an injected parcel, :math:`N_{d} m_{d,\rm{avg}}`, the estimated number of injected parcels is computed

   .. math::
      N_{P, \rm{inj}} = m_{\rm{inj}} / (N_{d} m_{d, \rm{avg}})

  * If :math:`N_{P, \rm{inj}} < N_{P, \min}`, the mass and time is accumulated as :math:`m_{\rm{acc}} = m_{\rm{inj}}` and :math:`t_{\rm{acc}} = t_{\rm{inj}}` and no injection occurs this time step.

  * Otherwise, :math:`m_{\rm{inj}}` mass is injected and convected over time :math:`t_{\rm{inj}}` and :math:`m_{\rm{acc}}` and :math:`t_{\rm{acc}}` are reset.

4. If injection occurs, the amount of mass injected, :math:`m_{\rm{actual}}`, is summed and compared with the desired mass flow rate. If :math:`m_{\rm{actual}} / t_{\rm{inj}} - \dot{m}_{\rm{inj}} > 0.05 \dot{m}_{\rm{inj}}`, then :math:`N_{P,\min}` is increased by one to reduce the likelihood of over-injecting in the future. A balance is necessary: the higher the minimum number of parcels, the less likely to over-inject mass but the number of time steps between injections can potentially grow as well.

Spray data derived from ANSYS Fluent DPM solution files can also be used to inject spray particles into the fluid domain. To use this feature, initialize a spray jet (named ``jet_dpm``, say) by using the statement `spray.jetnames=jet_dpm` in the input file as previously mentioned. In order to use the ANSYS Fluent DPM capability, set the boolean input variable ``spray.jet_dpm.read_from_dpm_file`` to true and specify the DPM file name by ``spray.jet_dpm.dpm_filename=<filename>``.
The DPM data in the file will correspond to a specified time-gap, but it can be made to repeat periodically by specifying ``spray.jet_dpm.is_dpm_periodic=true``. Otherwise, spray particle injection will cease after the final DPM time is reached. The spray injection can be set to start from a specific DPM time stamp by specifying
``spray.jet_dpm.initial_injection_dpm_time=<DPM time stamp>``. Similarly, the spray injection can also be set to start from a specific flow time by setting ``spray.jet_dpm.initial_injection_flow_time`` to the appropriate value. If the coordinate system used in the ANSYS Fluent DPM file is different from that in Pele,
one can use ``spray.jet_dpm.trans_matrix`` and ``spray.jet_dpm.translation`` to specify the direction cosine matrix (3X3 matrix) and the translation vector (<dx,dy,dz>) to align both the coordinate systems.


  The DPM file has a specific format and a typical file is shown below:

.. figure:: /Visualization/DPMFileFormat.png
   :align: center
   :figwidth: 60%

.. _spray_validation:

Spray Validation
================

Single Droplet Tests
--------------------

Single droplet tests are performed in 2D with PeleLMeX and compared with experimental results published in literature. These tests are setup in ``PeleLMeX/Exec/RegTests/SingleDropEvap`` and can be run with the ``Validate.py`` script. To run a test case with the *PeleMP* or *GCM* liquid properties, simply run ``Validate.py`` with the desired variables for ``case_name``, ``LiqPropsType``, and the method for calculating :math:`p_{\rm sat}` for the PeleMP model. Using the build flag ``-b`` ensures the correct executable is built for each case. For example: ::

   # Build a new executable and run WongLin with GCM
   python Validate.py -b -c WongLin

   # Build a new executable and run WongLin with PeleMP and Clasius-Clapeyron psat
   python Validate.py -b -c WongLin -l mp -p CC

   # Build a new executable and run RungeJP8 many-to-one with GCM and Manifold EOS
   python Validate.py -b -c RungeJP8 -m --cmlm_path <local-path-to-cmlm>

   # Build a new executable and run RungeJP8 one-to-one with HyChem mechanism with GCM
   python Validate.py -b -c RungeJP8-H

Comparison plots can be generated after the simulations have completed by running: ::

   # Generate comparison plots for all RungeJP8 cases
   python CompareResults.py -c RungeJP8

The following table details the parameters of each test:

.. table::

   +---------------+-----------------+-----------------+-----------------+-----------------+-----------------------+-----------------+-----------------+
   |Test Name      | :math:`T_g` [K] |:math:`p_g` [bar]|:math:`T_d` [K]  |:math:`d_d` [um] | :math:`\Delta u` [m/s]|Fuel             |Ref              |
   |               |                 |                 |                 |                 |                       |                 |                 |
   +===============+=================+=================+=================+=================+=======================+=================+=================+
   |``Nomura``     |471              |1                |298              |700              |0.0                    |heptane          |[#nomura]_       |
   +---------------+-----------------+-----------------+-----------------+-----------------+-----------------------+-----------------+-----------------+
   |``WongLin``    |1000             |1.01325          |315              |1961             |0.385                  |decane           |[#wonglin]_      |
   +---------------+-----------------+-----------------+-----------------+-----------------+-----------------------+-----------------+-----------------+
   |``Daif``       |348              |1.01325          |291              |1334             |3.1                    |heptane/decane   |[#daif]_         |
   +---------------+-----------------+-----------------+-----------------+-----------------+-----------------------+-----------------+-----------------+
   |``RungeHep``,  |272              |1.01325          |272              |570-594          |2.5                    |heptane,         |[#runge]_        |
   |``RungeDec``,  |                 |                 |                 |                 |                       |decane,          |                 |
   |``RungeMix``   |                 |                 |                 |                 |                       |mix              |                 |
   +---------------+-----------------+-----------------+-----------------+-----------------+-----------------------+-----------------+-----------------+
   |``RungeJP8``   |294.15           |1.01325          |294.15           |636              |3.0                    |POSF10264        |[#runge]_        |
   +---------------+-----------------+-----------------+-----------------+-----------------+-----------------------+-----------------+-----------------+
   |``Burger``     |800              |1, 10, 50        |300              |100              |0.0                    |POSF10325        |[#burger]_       |
   +---------------+-----------------+-----------------+-----------------+-----------------+-----------------------+-----------------+-----------------+

.. figure:: /Visualization/nomura_res_2025.png
   :align: center
   :figwidth: 45%

   Heptane droplet diameter comparisons with Nomura et al. [#nomura]_

.. figure:: /Visualization/wonglin_res_2025.png
   :align: center
   :figwidth: 80%

   Decane droplet diameter and temperature comparisons with Wong & Lin [#wonglin]_

.. figure:: /Visualization/daif_res_2025.png
   :align: center
   :figwidth: 80%

   Binary mixture of heptane and decane droplet diameter and temperature comparisons with Daı̈f et al. [#daif]_

.. figure:: /Visualization/runge_simple_res_2025.png
   :align: center
   :figwidth: 80%

   Droplet evaporation of heptane, decane, and a binary mixture of heptane and decane compared to experimental measurements from with Runge et al. [#runge]_ All three PeleMP cases utilized an Antoine fit for estimating the saturated vapor pressure.

.. figure:: /Visualization/runge_jp8_res_2025.png
   :align: center
   :figwidth: 80%

   Droplet evaporation of POSF10264 (JP8) compared to experimental measurements from Runge et al. [#runge]_ This demonstrates three different approaches: (i) A detailed non-reacting mechanism with 67 species in the liquid and gas phases, (ii) a single liquid species and corresponding single Hychem gas species, where the thermophysical properties of the liquid species are estimated using FuelLib's mixture properties, and (iii) a many-to-one mapping where 67 liquid species evaporate to a single Hychem gas species with and without the Manifold EOS. The many-to-one cases for PeleGCM and PeleMP with Antoine fit are indistinguishable on this plot.

.. figure:: /Visualization/burger_jeta_res_2026.png
   :align: center
   :figwidth: 80%

   Droplet evaporation of POSF10325 (Jet-A) compared to numerical experiments from Burger et al. [#burger]_ The PeleGCM simulations are performed using a 67-to-1 mapping (GC-to-HyChem) and the PeleMP simulations are performed using a 1-to-1 mapping with a single species representing the liquid and gas phases. The 50 bar case is not shown here as further development is needed to capture high pressure effects on liquid properties in the PeleGCM implementation. 

.. [#owen] "PeleMP: The Multiphysics Solver for the Combustion Pele Adaptive Mesh Refinement Code Suite," L. D. Owen, W. Ge, M. Rieth, M. Arienti, L. Esclapez, B. S. Soriano, M. E. Mueller, M. Day, R. Sankaran, and J. H. Chen, J. Fluids Eng., vol. 146, no. 4, pp. 1-18 (2024), doi: `10.1115/1.4064494 <https://doi.org/10.1115/1.4064494>`_.

.. [#gani94] "New group contribution method for estimating properties of pure compounds", L. Constantinou, and R. Gani, AIChE J., Vol. 40, No. 10, pp.1697-1710 (1994), doi: `10.1002/aic.690401011 <https://doi.org/10.1002/aic.690401011>`_.

.. [#gani95] "Estimation of the acentric factor and the liquid molar volume at 298 K using a new group contribution method", L. Constantinou, and R. Gani, Fluid Phase Equilibria, Vol. 103, No. 1, pp.11-22 (1995), doi: `10.1016/0378-3812(94)02593-P. <https://doi.org/10.1016/0378-3812(94)02593-P.>`_.

.. [#govindaraju] "Group contribution method for multicomponent evaporation with application to transportation fuels", Int. J. of Heat and Mass Transfer, Vol. 102, pp.833–845 (2016), doi: `10.1016/j.ijheatmasstransfer.2016.06.079 <https://doi.org/10.1016/j.ijheatmasstransfer.2016.06.079>`_.

.. [#abram] "Droplet vaporization model for spray combustion calculations", B. Abramzon and W. A. Sirignano, Int. J. Heat Mass Transfer, vol. 32, no. 9, pp. 1605-1618 (1989), doi: `10.1016/0017-9310(89)90043-4 <https://doi.org/10.1016/0017-9310(89)90043-4>`_.

.. [#ton] "Fuel spray modeling in direct-injection diesel and gasoline engines", S. Tonini, Dissertation, City University London (2006), url: `https://openaccess.city.ac.uk/id/eprint/8486/ <https://openaccess.city.ac.uk/id/eprint/8486/>`_.

.. [#Ge] "Development of a CPU/GPU portable software library for Lagrangian-Eulerian simulations of liquid sprays", W. Ge and R. Sankaran and J. H. Chen, Int. J. Multiph. Flow, vol. 128 (2020), doi: `10.1016/j.ijmultiphaseflow.2020.103293 <https://doi.org/10.1016/j.ijmultiphaseflow.2020.103293>`_.

.. [#nomura] "Experimental study on high-pressure droplet evaporation using microgravity conditions", H. Nomura and Y. Ujiie and H. J. Rath and J. Sato and M. Kono, Symposium (International) on Combustion, vol. 26, no. 1, pp. 1267–1273 (1996), doi: `10.1016/S0082-0784(96)80344-4 <https://doi.org/10.1016/S0082-0784(96)80344-4>`_.

.. [#wonglin] "Internal temperature distributions of droplets vaporizing in high-temperature convective flows", S.-C. Wong and A.-C. Lin, J. Fluid Mech., vol. 237, pp. 671–687 (1992), doi: `10.1017/S0022112092003574 <https://doi.org/10.1017/S0022112092003574>`_.

.. [#daif] "Comparison of multicomponent fuel droplet vaporization experiments in forced convection with the Sirignano model", A. Daı̈f and M. Bouaziz and X. Chesneau and A. Ali Chérif, Exp. Therm. Fluid Sci., vol. 18, no. 4, pp. 282-290, Issn 0894-1777 (1998), doi: `10.1016/S0894-1777(98)10035-3 <https://doi.org/10.1016/S0894-1777(98)10035-3>`_.

.. [#runge] "Low-temperature vaporization of JP-4 and JP-8 fuel droplets", T. Runge and M. Teske and C. E. Polymeropoulos, At. Sprays, vol. 8, pp. 25-44 (1998), doi: `10.1615/AtomizSpr.v8.i1.20 <https://doi.org/10.1615/AtomizSpr.v8.i1.20>`_.

.. [#burger] "Droplet evaporation modeling by the distillation curve model: accounting for kerosene fuel and elevated pressures", M. Burger, R. Schmehl, K. Prommersberger, O. Schäfer, R. Koch, S. Wittig, Int. J. Heat Mass Transfer, 46.23 (2003), doi: `10.1016/S0017-9310(03)00286-2 <https://doi.org/10.1016/S0017-9310(03)00286-2>`_.