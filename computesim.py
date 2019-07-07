# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 19:11:30 2019

@author: BACHRI Walid

convert data into similarity between drug or protein network
using jaccard distance 
"""


import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

Nets = ['mat_drug_drug', 'mat_drug_disease', 'mat_drug_se','mat_protein_protein', 'mat_protein_disease']

for net in Nets:
    M=np.loadtxt('data/'+net+'.txt',dtype='int')
    
    # jaccard distance
    Sim=1-pdist(M,'jaccard')
    Sim=squareform(Sim)
    Sim=Sim+np.eye(len(Sim))
    Sim=np.nan_to_num(Sim)
    np.savetxt('data/similarity/Sim_'+net+'.txt',Sim,fmt="%1.3f")

    


np.loadtxt('data/mat_drug_drug.txt',dtype='int')[0]