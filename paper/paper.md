title: 'ThemisPy: A Python analysis layer for post-processing Themis projects'
tags:
  - Python
  - astronomy
authors:
  - name: Avery E. Broderick
    orcid: 0000-0002-3351-760X
    affiliation: "1, 2, 3"
  - name: Paul Tiede
    orcid: 0000-0003-3826-5648
    affiliation: "1, 2, 3"
  - name: Britton Jeter
    orcid: 0000-0003-2847-1712
    affiliation: "1, 2, 3"
  - name: Boris Georgiev
    orcid: 0000-0002-3586-6424
    affiliation: "1, 2, 3"
affiliations:
  - name: Perimeter Institute for Theoretical Physics, 31 Caroline Street North, Waterloo, ON, N2L 2Y5, Canada
    index: 1
  - name: Department of Physics and Astronomy, University of Waterloo, 200 University Avenue West, Waterloo, ON, N2L 3G1, Canada
    index: 2
  - name: Waterloo Centre for Astrophysics, University of Waterloo, Waterloo, ON N2L 3G1 Canada
    index: 3
date: 21 August 2021
bibliography: paper.bib


# Summary

Themis is a growing and already an extensive C++ library of models, data structures, samplers and example analysis drivers for performing analysis of Very Long Baseline Interferometric data and ancillary observations of the primary Event Horizon Telescope (EHT) science targets [@Broderick:2020].  While computationally efficient, extensible, and portable to many high-performance computational resources, Themis lacks intrinsic tools to perform many important diagnostic and visualization tasks generic to likelihood exploration, model selection and parameter estimation.  Within the Themis software project, these crucial elements have been implemented in a heterogeneous language environment with project-specific and/or task-specific scripts or programs.  With the growth of Themis capabilities and the complexity of Themis analysis results, a more coherent and extensible approach is required.

ThemisPy is a Python package that provides a library of utilities for manipulating, assessing, and visualizing Themis analysis outputs.  Because the primary Themis outputs are Markov Chain Monte Carlo (MCMC) chains and ancillary data, the ThemisPy tools are likely to be broadly applicable to any analysis method capable of producing chains of samples distributed by their likelihood.

The ThemisPy package consists of three major and two minor components, each collected into subpackages.  The major components are:

* `chain` -- Objects and utilities for importing, manipulating, and writing MCMC chains.
* `diag` -- Utilities for generating MCMC chain convergence diagnostics, fit diagnostics.
* `vis` -- Functions for generating visualizations of model fits, parameter distributions, and a unified Themis model interface.

In addition, two minor components that more specific to the Themis representations of EHT data and EHT specific elements are encapsulated in the `data` and `utils` packages.

In addition, ThemisPy provides a number of command-line utilities for executing pre-scripted ThemisPy tasks on Themis analysis outputs.  These include the basic tasks of plotting samples from an MCMC chain, generating triangle plots of parameters (including double triangles), and computing diagnostics (e.g., split-$\hat{R}$, integrated autocorrelation times, etc.).

Detailed information about the API and installation can be found in the packaged documentation.  


# Acknowledgements

This work was supported in part by Perimeter Institute for Theoretical Physics.  Research at Perimeter Institute is supported by the Government of Canada through the Department of Innovation, Science and Economic Development Canada and by the Province of Ontario through the Ministry of Economic Development, Job Creation and Trade.
A.E.B. thanks the Delaney Family for their generous financial support via the Delaney Family John A. Wheeler Chair at Perimeter Institute.
A.E.B., P.T., and M.K. receive additional financial support from the Natural Sciences and Engineering Research Council of Canada through a Discovery Grant.

# References