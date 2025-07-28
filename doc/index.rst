.. INTEGRATE documentation master file, created by
   sphinx-quickstart on Thu Jan 24 09:18:42 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


INTEGRATE: Fast Probabilistic inversion of EM data using informed prior models
------------------------------------------------------------------------------
Last updated: |today|.

INTEGRATE provides a python module and methods for fast probabilistic inversion of local information (e.g. electromagnetic data (EM), well log data, ...) using informed prior models. 

The aim is to provide methods for the following tasks, that together represent a probabilistic workflow: 

Prior modeling
   Tools will be developed to quantify (through forward simulation) as much information as possible about the subsurface, such as the expected distribution of lithological layers and a model that links resistivity to lithology.
   See, for example, [MADSEN2023]_.
  
Forward modeling
   For each type of data considered a forward model must be available. 

   For EM type data we consider using GA-AEM (https://github.com/GeoscienceAustralia/ga-aem), simPEG [SimPEG]_, and AarhusInv [AarhusInv]_ as forward modeling engines.

   
Probabilistic Inversion
   An implementation of the 1D probabilistic localized inversion using the **Localized Rejection Sampler** [HANSEN2021]_ and **Machine Learning** [HANSENFINLAY2022]_.

   Features
      - Fast probabilistic inversion with informed prior models
      - Multiple Data Types
      - Multiple Forward Models
      - Joint inversion 


Analysis
   Tools for visual illustrations of the results will be developed, such as 1D, 2D cross-sections, 3D rendering, as well as uncertainty quantification.
   

Getting started
===============
Refer to the documentation in :doc:`install` for installation instructions.

Examples of using the module can be found in the :doc:`notebooks`.



The INTEGRATE project
=====================
The project is developed as part of the INTEGRATE project, where the goal is to develop probabilistic support tools that allow quantifying the potential for finding raw material resources close to where it is to be utilized.

For more information, please visit the INTEGRATE website (https://integrate.nu/).


Source Code
===========

   The current development version is available through GitHub at
   https://github.com/cultpenguin/integrate_mockup/.

.. 
   The latest stable code can be downloaded from
   http://cultpenguin.github.io/integrate/.



License (LGPL)
==============

This library is free software; you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 2.1 of the License, or (at
your option) any later version. This program is distributed in the hope
that it will be useful, but WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details. You should
have received a copy of the GNU Lesser General Public License along with
this program; if not, write to the Free Software Foundation, Inc., 59
Temple Place - Suite 330, Boston, MA 02111-1307, USA.

The manual
----------

.. toctree::
   :maxdepth: 3
	      
   install
   gettingstarted
   format
   workflow
   notebooks
   contributions
   references
   modules


   
