# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 10:21:52 

Learning edge features using binary operators

@author: BACHRI Walid
@email: bachriwalid@gmail.com
"""

import numpy as np
import pandas as pd


def average(Drug,Protein):
    #avg = np.array((len(Drug),Drug.shape[1]))
    ll=[]
    ii=0
    for i in Drug:
        for j in Protein:
            sm = np.add(i,j)
            ll.append(sm/2)
            #ii+=1


    return ll



drug = np.loadtxt('data/feature/drug_vector_d.txt')
protein = np.loadtxt('data/feature/protein_vector_d.txt')


dd = average(drug,protein)

dd.shape