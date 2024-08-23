############################################################
####           Divers schemas de volumes finis        ######
####               pour pbs elliptiques 1D            ######
####                                                  ######
####                 Programme principal              ######
####       permettant le calcul et le trace des sol.  ######
####                                                  ######
############################################################

##**************************************************************************
## On efface toutes les variables et on charge les fonctions utiles
##**************************************************************************
import Data
from scikits import umfpack as umf
from scipy.sparse import coo_matrix

import numpy as np
import matplotlib.pyplot as plt

##**************************************************************************
## Saisie par l'utilisateur des données du calcul
##**************************************************************************

## choix du cas test
cas_test = Data.donnees1D.cas_test

print("========================================")
print("Choix du cas test :")
for i in range(len(cas_test)):
    print(i + 1, ") ", cas_test[i]["nom"])


choix = -1
while (int(choix) <= 0) | (int(choix) > len(cas_test)):
    choix = input("Faites votre choix :")

choix = int(choix) - 1

## Chargement des donnees
donnees = cas_test[choix]

## choix du problème à résoudre
print("========================================")
print("Choix du problème :")
print("1) Laplacien Dirichlet homogène")
print("2) Laplacien Dirichlet non-homogène")
print("3) Diffusion générale Dirichlet non-homogène")
print("4) Diffusion générale Neumann non-homogène")
print("5) Diffusion générale Fourier non-homogène")
print("6) probleme générale dirichlet homogène")

choix_pb = -1
while (int(choix_pb) <= 0) | (int(choix_pb) > 6):
    choix_pb = input("Faites votre choix :")

choix_pb = int(choix_pb)
if choix_pb == 6:
    print("========================================")
    print("Choix du schéma:")
    print("1) centrée")
    print("2) upwind")
    print("3) upstream")
    choixschem = int(input("Faites votre choix :"))

## si le coeff de diffusion est variable : choix de la methode
if "methode" in donnees:
    print("========================================")
    print("Choix de la méthode de calcul du coeff de diffusion :")
    print("1) exacte")
    print("2) moyenne arithmetique")
    print("3) moyenne harmonique")

    choix = -1
    while (int(choix) <= 0) | (int(choix) > 3):
        choix = int(input("Faites votre choix :"))

    if choix == 1:
        donnees["methode"] = "exacte"
    elif choix == 2:
        donnees["methode"] = "arithmetique"
    elif choix == 3:
        donnees["methode"] = "harmonique"

## choix du type de maillage
print("========================================")
print("Choix du maillage :")
print("1) uniforme")
print("2) alterne")
print("3) aleatoire")
print("4) stretch")
choix = -1
while (int(choix) <= 0) | (int(choix) > 4):
    choix = input("Faites votre choix :")

choix = int(choix)

## choix du nombre de mailles
print("========================================")
N = input("Nombre de mailles :")
N = int(N)

## Creation du maillage
if choix == 1:
    maillage = Data.maillage_uniforme(N)
elif choix == 2:
    maillage = Data.maillage_alterne(N)
elif choix == 3:
    maillage = Data.maillage_aleatoire(N)
elif choix == 4:

    maillage = Data.maillage_stretch(N)


##**************************************************************************
## Construction du schéma et résolution du système linéaire
##**************************************************************************


## Construction de la matrice et du second membre
if choix_pb == 1:
    A, b = Data.vf_laplacien_dirh(maillage, donnees)

elif choix_pb == 2:
    A, b = Data.vf_laplacien_dirnh(maillage, donnees)
elif choix_pb == 3:
    A, b = Data.vf_diffusion_dirnh(maillage, donnees)
elif choix_pb == 4:
    A, b = Data.vf_diffusion_neunh(maillage, donnees)
elif choix_pb == 5:
    A, b = Data.vf_diffusion_fournh(maillage, donnees)
elif choix_pb == 6:
    A, b = Data.vf_laplacien_probg(maillage, donnees, choixschem)

print("Resolution")
A = A.tocsc()
sol = umf.spsolve(A, b)
if choix_pb == 4:
    moyenne = Data.InTrap(maillage["centres"], sol)
    sol = sol - moyenne
    if "moyenne" in donnees:
        sol = sol + donnees["moyenne"]
##**************************************************************************
## Affichage et calcul de l'erreur si disponible
##**************************************************************************
valeur_a_gauche = donnees["bordD"](0)
valeur_a_droite = donnees["bordD"](1)


plt.plot(
    np.concatenate((np.zeros(1), maillage["centres"], np.ones(1))),
    np.concatenate((valeur_a_gauche * np.ones(1), sol, valeur_a_droite * np.ones(1))),
    "o",
)


if "uexacte" in donnees:
    solexacte = donnees["uexacte"](maillage["centres"])

    print("========================================")
    print("Erreur en norme L2 = ", Data.normeL2(maillage, sol - solexacte))
    print("Erreur en norme Linf = ", np.max(np.abs(sol - solexacte)))
    print("Erreur en norme H1 = ", Data.normeH1(maillage, sol - solexacte))

    plt.plot(
        np.concatenate((np.zeros(1), maillage["centres"], np.ones(1))),
        np.concatenate(
            (valeur_a_gauche * np.ones(1), solexacte, valeur_a_droite * np.ones(1))
        ),
        "r",
    )

    plt.title(donnees["nom"] + ", " + maillage["nom"])
    plt.legend(["Solution Approchee", "Solution Exacte"])

else:
    plt.title("Solution Approchee")

plt.show()

print("==============  FIN  ===================")


##**************************************************************************
##                                    FIN
##**************************************************************************
