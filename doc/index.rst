.. ot-AK documentation master file, created by
   sphinx-quickstart on Mon Feb 14 09:17:40 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ot-AK documentation
===================

.. image:: _static/AK_methods.png
     :align: left
     :scale: 50%
	 
otAK is a module of OpenTURNS implementing some methods for combining Kriging and reliability methods. 
The different methods are: AK-MCS for Monte-Carlo Simulation, AK-IS for Importance Sampling and AK-SS 
for Subset Simulation. The AK methods are built on top of OpenTURNS and consist in training a Kriging metamodel
to avoid expensive simulations. 


AK methods involve dedicated infill criterion to refine the Kriging models only in
the area that are relevant to the reliability problem (in the vicinity of limit state and on area with
high probabilistic content).

Theory
------


.. toctree::
   :maxdepth: 1  
   
   principle/principle


User documentation
------------------

.. toctree::
   :maxdepth: 2  
   
   user_manual/user_manual



Examples 
--------

.. toctree::
   :maxdepth: 2  
   
   examples/examples


References
----------
- Echard, B., Gayton, N., & Lemaire, M. (2011). AK-MCS: an active learning reliability method combining Kriging and Monte Carlo simulation. Structural Safety, 33(2), 145-154.
- Echard, B. (2012). Assessment by kriging of the reliability of structures subjected to fatigue stress, Université Blaise Pascal, PhD thesis 
- Huan (2016). Assessing small failure probabilities by AK–SS: An active learning method combining Kriging and Subset Simulation, Structural Safety 59 (2016) 86–95

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
