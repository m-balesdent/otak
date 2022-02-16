"""
Example 2: Use of AK - IS on Non linear oscillator
--------------------------------------------------
"""

# %%

import openturns as ot
from openturns.viewer import View
import numpy as np
import otak

ot.RandomGenerator.SetSeed(1)


# %%
# | Definition of Input distribution

dist_c1=ot.Normal(1, 0.1)
dist_c2=ot.Normal(0.1, 0.01)
dist_m=ot.Normal(1.,0.05 )
dist_r =ot.Normal(0.5, 0.05)
dist_t1 =ot.Normal(1., 0.2)
dist_F1=ot.Normal(1., 0.2)
std_dev = [dist_c1.getStandardDeviation()[0],dist_c2.getStandardDeviation()[0],dist_m.getStandardDeviation()[0],
           dist_r.getStandardDeviation()[0],dist_t1.getStandardDeviation()[0],dist_F1.getStandardDeviation()[0]]
dim_inputs=6
marginals = [dist_c1,dist_c2,dist_m,dist_r,dist_t1,dist_F1]
dist_x = ot.ComposedDistribution(marginals)

# %%
# | Definition of limit state function

def non_lin_osc(x):
    c1 = x[0]
    c2  = x[1]
    m = x[2]
    r = x[3]
    t1 = x[4]
    F1= x[5]
    
    omega = np.sqrt((c1+c2)/m)
    
    G = 3*r - np.abs(2*F1/(m*omega**2)*np.sin(omega*t1/2))
    return [G]

# Definition of pythonfunction
non_lin_osc = ot.PythonFunction(6, 1, non_lin_osc)

# %%
# | Definition of event

vect = ot.RandomVector(dist_x)
G = ot.CompositeRandomVector(non_lin_osc, vect)
event_osc = ot.ThresholdEvent(G, ot.Less(), 0.0)


# %%
# | Run of AK IS

dim_osc = 6
basis = ot.ConstantBasisFactory(dim_inputs).build()
covarianceModel = ot.SquaredExponential([0.1]*dim_osc, [1.0])

n_IS_osc = 1000
n_DoE_osc = 40
sim_budget_osc = 200
verbose = False
my_AK_non_lin_osc = otak.AK_ISAlgorithm(event_osc,
                                    n_IS_osc,
                                    n_DoE_osc,
                                    sim_budget_osc,                                                                   
                                    basis,
                                    covarianceModel,
                                    ot.Cobyla(),
                                    2,
                                    verbose)
									
									
#computation of probability with AK-IS
my_AK_non_lin_osc.compute_proba()


print('Probability of failure:',my_AK_non_lin_osc.getFailureProbability())
print('Simulation budget:',my_AK_non_lin_osc.getSimBudget())