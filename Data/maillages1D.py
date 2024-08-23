############################################################
####           Divers schemas de volumes finis        ######
####               pour pbs elliptiques 1D            ######
####                                                  ######
####        Definition et outils pour les maillages   ######
####                                                  ######
############################################################


##**************************************************************************
##    Le format des maillages
##**************************************************************************
##  Chaque maillage est une structure contenant 5 éléments :
##  *  maillage['nom']     :  le nom de la famille de maillages
##  *
##  *  maillage['nb_vol']  :  le nombre de volumes de contrôle
##  *  maillage['centres'] :  vecteur de taille nb_vol contenant la position
##                         des centres des volumes de contrôle
##  *  maillage['mes']     :  vecteur de taille nb_vol contenant la mesure
##                         des volumes de contrôle
##  *  maillage['sommets'] :  vecteur de taille nb_vol+1 contenant la position
##                         des sommets
##         maillage['sommets'][i] correspond à x_{i+1/2} pour 0<=i<=nb_vol
##  *  maillage['dist']    :  vecteur de taille nb_vol+1 contenant la distance
##                         entres les centres de deux volumes voisins
##         maillage['dist'][i] correspond à h_{i+1/2} pour 0<=i<=nb_vol
##**************************************************************************


import numpy as np
import matplotlib.pyplot as plt

import random
from sklearn.utils import shuffle

#######################################
######### Maillages uniformes #########
#######################################


def maillage_uniforme(N):

    h = 1 / N
    sommets = h * np.arange(N + 1)
    centres = h * np.arange(1, N + 1) - h / 2
    mes = h * np.ones(N)
    dist = h * np.ones(N + 1)
    dist[0] = h / 2
    dist[-1] = h / 2

    m = {
        "nom": "maillage uniforme",
        "sommets": sommets,
        "centres": centres,
        "mes": mes,
        "dist": dist,
        "nb_vol": np.shape(centres)[0],
    }

    return m


######################################
######### Maillages alternes #########
######################################


def maillage_alterne(N):
    # On s'arrange pour que le nombre de mailles soit un multiple de 4
    # ce qui assure que le point x=0.5 est un sommet du maillage
    N = (int((N - 0.1) / 4) + 1) * 4

    x = np.ones(N)
    x[1::2] = 2
    x = np.cumsum(x)
    x = x / max(x)

    sommets = np.concatenate((np.zeros(1), x))

    centres = np.zeros(N)
    centres = (sommets[1:] + sommets[:-1]) / 2

    mes = np.zeros(N)
    mes = sommets[1:] - sommets[:-1]

    dist = np.zeros(N + 1)
    dist[1:-1] = centres[1:] - centres[:-1]
    dist[0] = centres[0]
    dist[-1] = 1 - centres[-1]

    m = {
        "nom": "maillage alterne",
        "sommets": sommets,
        "centres": centres,
        "mes": mes,
        "dist": dist,
        "nb_vol": np.shape(centres)[0],
    }

    return m


####################################################
############### Maillages aleatoires ###############
##(perturbation aleatoire d'un maillage uniforme) ##
####################################################


def maillage_aleatoire(N):

    x = np.linspace(0 + (1 / (N - 1)), 1 - 1 / (N - 1), N - 1) + 0.1 * (
        2 * np.random.rand(N - 1) - 1
    ) / (N - 1)

    x = sorted(x)
    sommets = np.concatenate((np.zeros(1), x, np.ones(1)))

    # On s'arrange pour que 0.5 soit un sommet du maillage
    val = np.min(np.abs(sommets - 0.5))
    indice = np.argmin(np.abs(sommets - 0.5))
    indice = np.min(indice)
    sommets[indice] = 0.5

    centres = np.zeros(N)
    centres = (sommets[1:] + sommets[:-1]) / 2

    mes = np.zeros(N)
    mes = sommets[1:] - sommets[:-1]

    dist = np.zeros(N + 1)
    dist[1:-1] = centres[1:] - centres[:-1]
    dist[0] = centres[0]
    dist[-1] = 1 - centres[-1]

    m = {
        "nom": "maillage aleatoire",
        "sommets": sommets,
        "centres": centres,
        "mes": mes,
        "dist": dist,
        "nb_vol": np.shape(centres)[0],
    }

    return m


#################################################
######### Maillages étirés près du bord #########
#########  et raffinés près du centre   #########
#################################################


def maillage_stretch(N):

    n = np.floor(N / 2)
    n = int(n)
    sommets = np.zeros(N + 1, dtype=float)
    alp = 100 / 101  # 9/10 #3/4

    for i in range(n + 1):
        sommets[i] = (1 - alp ** (i)) / (2 * (1 - alp ** (n)))

    for i in range(n + 1, N + 1):
        sommets[i] = 1 - (1 - alp ** (N - i)) / (2 * (1 - alp ** (N - n)))

    centres = np.zeros(N)
    centres = (sommets[1:] + sommets[:-1]) / 2

    mes = np.zeros(N)
    mes = sommets[1:] - sommets[:-1]

    dist = np.zeros(N + 1)
    dist[1:-1] = centres[1:] - centres[:-1]
    dist[0] = centres[0]
    dist[-1] = 1 - centres[-1]

    m = {
        "nom": "maillage stretch",
        "sommets": sommets,
        "centres": centres,
        "mes": mes,
        "dist": dist,
        "nb_vol": np.shape(centres)[0],
    }

    return m


#################################################
######### Maillages mélangés #########
#########  et raffinés près du centre   #########
#################################################
def maillage_melange(choix_maillage, N):

    m = choix_maillage(N)  # maillage normale

    vec_centres = np.arange(m["nb_vol"])  # [1 2 3 ... N]
    random.shuffle(vec_centres)  # [8 2 5 ... N 3]
    m["centres"] = m["centres"][vec_centres]
    m["mes"] = m["mes"][vec_centres] 

    mel_centres = np.argsort(vec_centres) 
    s_KL = np.zeros((m["nb_vol"] + 1, 2), dtype=int)
    s_KL[1:, 0] = mel_centres
    s_KL[1:-1, 1] = mel_centres[1:]
    s_KL[0, 0] = mel_centres[0]
    s_KL[0, 1] = -1
    s_KL[-1, 1] = -1 

    s_KL, m["dist"], m["sommets"] = shuffle(s_KL, m["dist"], m["sommets"])

    m["s_KL"] = s_KL

    m["nom"] = m["nom"] + " melange"

    return m


##**************************************************************************
##   Fonctions d'affichage d'un maillage
##**************************************************************************
##   A n'utiliser que pour des petits nombres de volumes de contrôle
##**************************************************************************


def trace_maillage(m):
    plt.plot(m["centres"], 0 * m["centres"], "bs", markersize=5)
    plt.plot(m["sommets"], 0 * m["sommets"], "rd", markersize=5)

    plt.show()


mixm = maillage_melange(maillage_uniforme, 5)
m = maillage_uniforme(5)
print("\n\nsommets", mixm["sommets"])
print("centres", mixm["centres"])
print("distance", mixm["dist"])
print("mes", mixm["mes"])
print("s_KL", mixm["s_KL"],"\n\n")

##**************************************************************************
##   Calculs de normes discrètes
##**************************************************************************


def normeL2(m, u):
    return np.sqrt(np.sum(m["mes"] * (u**2)))


## Pour calculer le gradient, on suppose que la fonction est nulle au bord
## Ce n'est pas gênant car ces fonctions sont destinées à être appliquées
## a la difference entre la solution exacte et la solution approchee.


def normeH1(m, u):
    gu = (np.concatenate((u, np.zeros(1))) - np.concatenate((np.zeros(1), u))) / m[
        "dist"
    ]
    return np.sqrt(np.sum(m["dist"] * (gu**2)))


def normeLip(m, u):
    gu = (np.concatenate((u, np.zeros(1))) - np.concatenate((np.zeros(1), u))) / m[
        "dist"
    ]
    return np.max(np.abs(gu))
