"""
Example 3: Use of AK - SS on four branch function
-------------------------------------------------
"""

# %%

import openturns as ot
from openturns.viewer import View
import numpy as np
import otak

ot.RandomGenerator.SetSeed(1)


# %%
# | Definition of Input distribution

dim_inputs = 2
dist_x = ot.Normal([0.0, 0.0], [1., 1.], ot.CorrelationMatrix(dim_inputs))
inputVector = ot.RandomVector(dist_x)

# %%
# | Definition of limit state function

def four_branch(x):
    x1 = x[0]
    x2  = x[1]
    k = x[2]
    g1 = 3+0.1*(x1-x2)**2-(x1+x2)/np.sqrt(2)
    g2 = 3+0.1*(x1-x2)**2+(x1+x2)/np.sqrt(2)
    g3 = (x1-x2)+k/np.sqrt(2)
    g4 =(x2-x1)+k/np.sqrt(2)
    return [min((g1,g2,g3,g4))]

# Definition of python function
my_four_branch = ot.PythonFunction(3, 1, four_branch)

# Transformation of python function to parametric function
index_frozen = [2]
my_four_branch_6 = ot.ParametricFunction(my_four_branch, index_frozen, [6])

# %%
# | Definition of event

Y = ot.CompositeRandomVector(my_four_branch_6, inputVector)
my_event4b = ot.ThresholdEvent(Y,ot.Less(),0.0)


# %%
# | Run of AK SS

dim_4b = 2
basis = ot.ConstantBasisFactory(dim_4b).build()
covarianceModel = ot.MaternModel(dim_4b)
n_MC_4b = 1000
n_DoE_4b = 10
sim_budget_4b = 200
verbose = False
criterion = 2

proposal_range = 1.
target_proba=0.1
cv_target = 0.05

my_AK_four_branch = otak.AK_SSAlgorithm(my_event4b,
                                    n_MC_4b,
                                    n_DoE_4b,
                                    sim_budget_4b,
                                    basis,
                                    covarianceModel,
                                   proposal_range,
                                   target_proba,
                                    cv_target,
                                    criterion,
                                    verbose)

#computation of probability with AK-SS
my_AK_four_branch.compute_proba()

print('Probability of failure',my_AK_four_branch.getFailureProbability())

print('Simulation budget',my_AK_four_branch.getSimBudget())

print('Coefficient of variation',my_AK_four_branch.getCoefficientOfVariation())
