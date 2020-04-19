Developer's Guide
==============================

Getting the latest version
------------------------------
The latest version of ThemisPy can be obtained from https://github.com/aeb/ThemisPy.

ThemisPy uses the git-flow_ extensions to facilitate development.


Scope and goals
------------------------------
ThemisPy seeks to organize and structure postprocessing functions for Themis_ analyses.
As such, it is closely related to and dependent on the structure and development of
the Themis_ analysis framework.  Importantly, it is not intended to produce analyses
of its own.  If such an analysis is necessary, consider generating a separate script
and submit it to the `Themis repository`_.


Backward compatibility
------------------------------
ThemisPy follows the `Semantic Versioning 2.0`_ versioning system.  There is no current
plan to pursue additional major versions after 1.0.0.  Thus, ThemisPy must strictly
maintain backwards compatability.  If you add a feature, remember that you will be
responsible for ensuring its continued functionality.  Thus, think carefully about
the existing structure and how to build durable additions.


Portability
------------------------------
ThemisPy aspires to be as universally portable as possible.  While library
interdependencies are neither unavoidable nor always undesirable, and ThemisPy does
depend on Numpy, SciPy, and Matplotlib, such dependencies should be kept to a
minimum.  No absolute rules are possible for this.  Nevertheless:

1. Consider if additional libraries are required, or if the goal can be
   achieved within the confines existing dependencies.
2. Where additional libraries are required to enable specific functionality
   consider wrapping the necessary import statements in ``try``, ``except``
   blocks with warnings.

For example, ehtim_ is not included as a dependency, though some functionality is
dependent upon it.  Where such functions are present, the following import block is
included:

::

   try:
       import ehtim as eh
       ehtim_found = True
   except:
       warnings.warn("foo not found.", Warning)
       ehtim_found = False


Documenting with autodoc
------------------------------
All components should be documented with python docstrings that clearly specify the
purpose, function, inputs and outputs (if any).  For example, a new function might
be documented as:

::

   def myfunc(x,y) :
   """
   This function is an example.
   
   Args:
     - x (float): A first parameter.
     - y (float): A second parameter.

   Returns:
     - (float): The sum of the two inputs.
   """

   return x+y

If types are specified, they will be crosslinked with the relevant source documentation.

Mathematical espressions can be included inline using the ``:math:`` command:

::

   """
   :math:`x^2+y^2+\\alpha`
   """

and in displaystyle via:

::
   
   """
   .. math::

      f(x) = \\int_0^\\infty dx \\frac{\\cos(x^2)}{1+g(x)}
   """
   
AMSmath LaTex directives can be inserted, though escape characters must be
escaped themselves if they are imported via autodoc (i.e., the ``\\`` in ``\\alpha``).


Autodoc directives for new module files must be added to the appropriate documentation
file in docs/src, e.g.,

::

   .. automodule:: chain.mcmc_chain
   :members:









 

.. _Themis: https://perimeterinstitute.github.io/Themis
.. _`Themis repository`: https://github.com/PerimeterInstitute/Themis
.. _ehtim: https://achael.github.io/eht-imaging/array.html      
.. _`Semantic Versioning 2.0`: https://semver.org/
.. _git-flow: https://danielkummer.github.io/git-flow-cheatsheet/
