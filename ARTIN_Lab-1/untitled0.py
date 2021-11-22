# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:15:38 2021

@author: Pierro
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import random as rd


def init_centre ():
    Li_centres = []
    nb_centre=0
    while (True):
        i = rd.randint (0,len(X)-1)
        x1 = X[i,0]
        x2 = X[i,1]
        centre = (x1,x2)
        if (centre in Li_centres):
            continue
      
        Li_centres.append (centre)
        nb_centre = nb_centre +1
        if (nb_centre == K):
            break
    #Pour test
    Li_centres = []
    Li_centres.append( (2,10) )
    Li_centres.append( (5,8) )
    Li_centres.append( (1,2) )
    return Li_centres


def distance (pt1,pt2):
    # pt1 = (a1,a2) & pt2 = (b1,b2)  
    lg=len(pt1)
    somme=0
    for i in range (lg):
        dif = pt1[i] - pt2[i]
        somme = somme + np.abs(dif)
    return somme

def rallier_points (Li_centres):
     
    for x in range (len(X)):
        pt = (X[x,0] , X[x,1])
        li_distances = []
        for k in range (K): 
            di = distance (pt , Li_centres[k])
            col = "dist-{}".format(k)
            df.loc[x,col] = di
            li_distances.append(di)

        cluster = np.argmin(li_distances) 
        df.loc[x,"cluster"] = cluster
  
    
def c_gravite (cluster):
    condition = df['cluster'] == cluster
    select = df[condition]
   
    c1 = select['x1'].mean()
    c2 = select['x2'].mean()
  
    return c1,c2

def calculer_new_centres():
    nw_li_centre = []
    for k in range (K):
        nw_centre = c_gravite (k)
        nw_li_centre.append (nw_centre)
    return nw_li_centre
    

def stabilite (clust1,clust2):
    if (clust1 == clust2):
        return True
        
    else :
        return False


def KMEANS ():
    clust_pred = list(df ["cluster"])
    
    Li_centres = init_centre()
    
    while (True) : 
         
         rallier_points (Li_centres)
         clust_courant = list(df ["cluster"])
         stab = stabilite (clust_pred,clust_courant)
         if (stab == True) :
             break

         Li_centres = calculer_new_centres()
         clust_pred = clust_courant.copy()
   
    return clust_courant
resultat1 = KMEANS()

print( resultat1 )
    



