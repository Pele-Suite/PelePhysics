.. highlight:: rst

A brief introduction to CVODE
=============================

CVODE is part of a software family called sundials for SUite of Nonlinear and DIfferential / ALgebraic equation Solver [LLNL2005]_. 

**The version used with PelePhysics for the tests presented in this document is v4.0.2. This does not mean this is the recommended version to link with PelePhysics. Always refer to** :ref:`sec:GetCVODE` **to know which versions of CVODE/SuiteSparse to use !!** 

In the following, details pertaining to the methods implemented in `PelePhysics` are provided. 
The interested user is referred to the very exhaustive `CVODE User-guide <https://computation.llnl.gov/sites/default/files/public/cv_guide.pdf>`_ for more information.

`Most of this section is adapted from the v4.1.0 Cvode Documentation.`

.. _sec:NMO: 

Numerical methods overview
--------------------------

The methods implemented in CVODE are variable-order, variable-step `multistep` methods, based on formulas of the form

.. math::

    \sum_{i=0}^{K_1} \alpha_{n,i} y^{n-i} + h_n \sum_{i=0}^{K_2} \beta_{n,i} \dot{y}^{n-i} = 0 

Here the :math:`y^n` are computed approximations to :math:`y(t_n)`, and :math:`h_n = t_n-t_{n-1}` is the step size. 
For stiff problems, CVODE includes the Backward Differentiation Formulas (BDF) in so-called fixed-leading coefficient (FLC) form, 
given by :math:`K_1=q` and :math:`K_2= 0`, with order :math:`q` varying between 1 and 5.  The coefficients are uniquely determined by the method type, 
its order, the recent history of the step sizes, and the normalization :math:`\alpha_{n,0}=-1` [BYRNE1975]_, [JAC1980]_.  

A nonlinear system must be solved (approximately) at each integration step.  This nonlinear system can be formulated as a root-finding problem

.. math::

    F(y^{n}) = y^n - h_n \beta_{n,0} f(t_n,y^{n}) - a_n = 0

where :math:`a_n = \sum_{i>0} (\alpha_{n,i} y^{n-i} + h_n\beta_{n,i} \dot{y}^{n-i})`. CVODE provides several non-linear solver choices. 
By default, CVODE solves this problem with a Newton iteration, which requires the solution of linear systems

.. math::  M[y^{n(m+1)} - y^{n(m)}] = -F(y^{n(m)})
    :label: eqc

in which

.. math::
    M \approx I-\gamma J, \; \; \; J = \frac{\partial f}{ \partial y}, \;\;\; and \;\;\; \gamma =  h_n \beta_{n,0}


.. _sec:subslinalg:

Linear Algebra
--------------

To find the solution of the linear system :eq:`eqc`; CVODE provides several linear solver choices. 
The linear solver modules distributed with Sundials are organized in two families, a `direct` family comprising direct linear solvers 
for dense, banded, or sparse matrices, and a `spils` family comprising scaled preconditioned iterative (Krylov) linear solvers.  
The solvers offered through these modules that are of interest to us are:

- a dense direct solver
- a sparse direct solver interface using the `KLU` sparse solver library
- SPGMR, a scaled -possibly preconditioned- GMRES (Generalized Minimal Residual method) solver [BROWN1990]_

When using a dense direct solver, the user has the option to specify an Analytical Jacobian. 
If none is provided, a difference quotients is performed. When a sparse direct solver is employed however, 
the user **must** specify an analytical Jacobian. All of these options have been enabled in `PelePhysics`.

For large stiff systems,  where direct methods are often not feasible, the combination of a BDF integrator and a `preconditioned` Krylov method 
yields a powerful tool. In this case, the linear solve is `matrix-free`, and the default Newton iteration is an 
`Inexact` Newton iteration, in which :math:`M` is applied with matrix-vector products :math:`Jv` obtained by either difference quotients 
or a user-supplied routine. In `PelePhysics`, it is possible to use either a non-preconditioned or a preconditioned GMRES solver. 
In the latter case, the preconditioner can be either specified in a dense or sparse format (if the KLU library is linked to CVODE), 
and it is provided in the form of a Jacobian approximation, based on the work of [McNenly2015]_.



Error control, step-sizing, order determination
-----------------------------------------------

In the process of controlling errors at various levels, CVODE uses a weighted root-mean-square norm, 
denoted :math:`|| \bullet ||_{WRMS}`, for all error-like quantities. The multiplicative weights used are based 
on the current solution and on the relative and absolute tolerances input by the user, namely

.. math::

    W_i= \frac{1}{[rtol |y_i|+atol_i]}

Because :math:`1/W_i` represents a tolerance in the component :math:`y_i`, a vector whose norm is 1 is regarded as small. 
In `PelePhysics`, both these tolerances are fixed to a value of :math:`1.0e-10`.

A critical part of CVODE - making it an ODE `solver` rather than just an ODE method, is its control
of the local error. At every step, the local error is estimated and required to satisfy tolerance conditions, 
and the step is redone with reduced step size whenever that error test fails. 
Note that in `PelePhysics`, the first time step is always forced to :math:`1.0e-9`.

In addition to adjusting the step size to meet the local error test, CVODE periodically adjusts the order, 
with the goal of maximizing the step size. The integration starts out at order 1 and varies the order dynamically after that. 
The basic idea is to pick the order :math:`q` for which a polynomial of order :math:`q` best fits the discrete data involved 
in the multistep method. In `PelePhysics`, the maximum order is limited to 2 for stability reasons.

The various algorithmic features of CVODE are inherited from VODE and VODPK, and are documented in [VODE1989]_ and [BROWN1990]_.  
They are also summarized in the `CVODE User-guide <https://computation.llnl.gov/sites/default/files/public/cv_guide.pdf>`_ as well as in [LLNL2005]_.


.. [LLNL2005] A. C. Hindmarsh, P. N. Brown, K. E. Grant, S. L. Lee, R. Serban, D. E. Shumaker, and C. S. Woodward. SUNDIALS: Suite of nonlinear and differential/algebraic-equation solvers. ACM Transactions on Mathematical Software (TOMS), 31(3):363-396, 2005.

.. [BYRNE1975] G. D. Byrne, A. C. Hindmarsh. A polyalgorithm for the numerical solution of ordinary differential equations. ACM Transactions on Mathematical Software (TOMS), 1(1):71-96, 1975.

.. [JAC1980] K. R Jackson and R. Sacks-Davis. An alternative implementation of variable step-size multistep formulas for stiff odes. ACM Transactions on Mathematical Software (TOMS), 6(3):295–318, 1980.

.. [BROWN1990] P. N. Brown and Y. Saad. Hybrid krylov methods for nonlinear systems of equations. SIAM Journal on Scientific and Statistical Computing, 11(3):450–481, 1990.

.. [McNenly2015] M. J. McNenly, R. A. Whitesides, and D. L. Flowers. Faster solvers for large kinetic mechanisms using adaptive preconditioners. Proceedings of the Combustion Institute, 35(1):581–587, 2015.

.. [VODE1989] P. N. Brown, G. D. Byrne, and A. C. Hindmarsh. VODE, a variable-coefficient ODE solver. SIAM journal on scientific and statistical computing, 10(5):1038-1051, 1989. 
