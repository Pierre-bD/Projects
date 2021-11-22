import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import os

dossier_courant = os.getcwd()  
print ( 'dossier_courant : ', dossier_courant )                          

nouveau_dossier = 'C:\\Users\\LAAAB01\\test\\test16-KNN-Kmeans'
os.chdir(nouveau_dossier)

# 1) Charger le fichier dataset dans un Dataframe df

filename = "dataset-K-Means.csv"
df = pd.read_csv( filename )
print(df)


# 2) Afficher les données sur un plan

X = df.values
labels = X[:,2]

fig = plt.figure()
x_min, x_max = 0, 11
y_min, y_max = 0, 11

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

grid_x_ticks = np.arange(x_min, x_max)
grid_y_ticks = np.arange(y_min, y_max)

plt.xticks(grid_x_ticks)
plt.yticks(grid_y_ticks)

plt.scatter(X[:,0], X[:,1], c="red", s=80)
plt.grid()
plt.show()


# 3) ajouter K colonnes 'dist-i'
K = 3
for i in range(0, K):
    #col = f"dist-{i}"
    col = "dist-{}".format(i)
    df[col] = np.nan  

print(df)    

# 4) Créer un tableau de K centres, et les tirer aléatoirement

m = len(X)    # taille dataset

Li_centres = []
compteur = 0
while True:
    i = rd.randint(0,m-1)    # tirer un centre aléatoirement
    
    a1  = df.loc[i,"x1"]
    a2  = df.loc[i,"x2"]
    centre = (a1, a2)
    #print( type (centre))
    
    if centre in Li_centres:
        continue
    
    Li_centres.append(centre)
    compteur +=1
    if (compteur == K):
        break

print(Li_centres)
    
# UNIQUEMENT POUR TEST, on force les 3 centres initiaux

Li_centres = []
Li_centres.append( (2,10) )
Li_centres.append( (5,8) )
Li_centres.append( (1,2) )
############################

    

# 4) Définir une fonction distance(pt1, pt2) qui calcule la distance Manhattan 
#    entre deux points pt1 et pt2

def distance(pt1, pt2):
    # pt1(a1, a2, ... , an)
    # pt2(b1, b2, ... , bn)
    n  = len(pt1)     

    somme = 0
    for i in range(n):
        difference = pt1[i] - pt2[i]   # ai - bi
        somme = somme + np.abs(difference)
    return somme


# 5) définir une fonction Rallier_points qui rallie chaque point au
# centre le plus proche, et qui met à jour la colonne cluster pour ce point

    
def Rallier_points(Li_centres):

    m = len(X)    # taille dataset
    
    for i in range(m):       # Pour chaque point pt
        a1  = df.loc[i,"x1"]
        a2  = df.loc[i,"x2"]
        pt = (a1,a2)
        
        Li_distances = []   # calculer la distance entre pt et chacun des centres
        for k in range(K):   
            centre = Li_centres[k]
        
            d = distance(pt, centre)
            d = round(d,2)
            col = "dist-{}".format(k)
            df.loc[i,col] = d   # stocker la distance calculée dans la colonne "dist-i"
            
            Li_distances.append(d)
            
        # Calculer la distance minimale parmi dist-0, dist-1, ...
        cluster = np.argmin(Li_distances)           
        df.loc[i, "cluster"] = cluster     # stocker le cluster dans la colonne "cluster"                 

# calculer_new_centres()

def c_gravite(cluster):
    #print("DEBUT c_gravite")
    condition = df['cluster'] == cluster        
    selection = df[ condition ] 

    c1 =selection["x1"].mean()
    c2 =selection["x2"].mean() 

    #print("c_gravite  --> c1 :", c1)
    #print("c_gravite  --> c2 :", c2)
    
    return c1,c2

def calculer_new_centres():
    #print("DEBUT calculer_new_centres")
    New_Li_centres = []

    for cluster in range(0,K):
        new_centre = c_gravite(cluster)
        New_Li_centres.append( new_centre )
        
    print("New centres : ")
    print(New_Li_centres)
    
    return New_Li_centres
        
def stabilite(new_clusters, prev_clusters):
    if ( new_clusters == prev_clusters ):
        return True
    else:
        return False

# 6) Définir la fonction MyKmeans() : boucle jusqu'à stabilisation des clusters
        
def MyKmeans():
    global Li_centres
    prev_clusters = list(df["cluster"])
    
    while (True):
       
        Rallier_points(Li_centres)
        
        print("df APRES UN PASSAGE de la boucle : ")
        print(df)
        #input("...")
        
        new_clusters = list(df["cluster"])
    
        if (stabilite(new_clusters, prev_clusters) == True):
            print("STABILITE ATTEINTE")
            break
        
        prev_clusters = new_clusters.copy()
        
        Li_centres = calculer_new_centres()

    # Retourner le cluster final obtenu 
    return new_clusters

# Lancer MyKmeans()
clusters_test_1 = MyKmeans()
print(clusters_test_1)
 
##### TESTER AVEC LA LIB SKLEARN ####################
    
# Algorithme K-Mean-Clustering

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rd
from sklearn.cluster import KMeans
import os

dossier_courant = os.getcwd()  
print ( 'dossier_courant : ', dossier_courant )                          

nouveau_dossier = 'C:\\Users\\LAAAB01\\test\\test16-KNN-Kmeans'
os.chdir(nouveau_dossier)

K = 3

filename = "dataset-K-Means.csv"
df = pd.read_csv( filename )
print(df)


# 2) Afficher les données sur un plan

X = df.values
labels = X[:,2]

plt.scatter(X[:,0], X[:, 1])                # X[:,0] = tous les x1   , X[:, 1] = tous les x2       

# Entrainer le modele de K-mean Clustering
mymodel = KMeans(n_clusters=K, random_state=3) 
# IMPORTANT random_state=3  c'est le paramètre random qui permet de générer 
# des no de clusters à partir de zéro, c.a.d  0, 1, 2, ...

mymodel.fit(X)

#Visualiser les Clusters

predictions = mymodel.predict(X) 

print(predictions)

# stocker les prédictions (no de clusters) via SKLEARN

clusters_test_2 = predictions.copy() 

print(clusters_test_2)

# Comparer avec le résultat du test précédent avec MyKmean()

for c1, c2 in zip(clusters_test_1, clusters_test_2 ):
    print(c1, c2)          


    
    




