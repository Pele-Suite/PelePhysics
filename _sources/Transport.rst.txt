.. highlight:: rst

.. _sec:transport:

*********
Transport
*********

PelePhysics supports computation of several transport coefficients: dynamic viscosity (:math:`\mu`), bulk viscosity (:math:`\xi`), thermal conductivity (:math:`\lambda`), mixture-averaged species diffusion coefficients (:math:`D_i`), and Soret coefficients/thermal diffusion ratio (:math:`\chi_i`). There are three choices of model for computing these transport coefficients:

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
  transport.const_conducitivity = 2.623e2
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
	  
In this model, transport coefficients are evaluated from data available in the chemical mechanisms (set at compilation using ``Chemistry_Model``). The implementation isbased on that in `EGlib <http://www.cmap.polytechnique.fr/www.eglib/>`_ (see `Ern and Giovangigli (1995) <https://doi.org/10.1006/jcph.1995.1151>`_) and simplified to compute only mixture-averaged diffusivities for each species.  The only option that may be specified at run time is whether or not to compute Soret coefficients, which is done by setting the input file parameter ``transport.use_soret`` to 1 or 0, respectively (default: 0).

When Simple transport is used with the Soave-Redlich-Kwong equation of state, additional corrections are used to modify the transport coefficients to account for real gas effects based on `Chung et al. (1988) <https://doi.org/10.1021/ie00076a024>`_. Soret effects are not supported for SRK.
