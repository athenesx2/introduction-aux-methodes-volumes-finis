############################################################
####           Divers schemas de volumes finis        ######
####               pour pbs elliptiques 1D            ######
####                                                  ######
####                 Programme principal              ######
####       permettant le trace de courbes d'erreurs   ######
####                                                  ######
############################################################

##**************************************************************************
## On efface toutes les variables et on charge les fonctions utiles
##**************************************************************************
import Data
import scikits.umfpack as umf

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
print("6) problème générale")

choix_pb = -1
while (int(choix_pb) <= 0) | (int(choix_pb) > 6):
    choix_pb = input("Faites votre choix :")

choix_pb = int(choix_pb)
if choix_pb==6:
    print("========================================")
    print("Choix du schéma:")
    print("1) centrée")
    print("2) upwind")
    print("3) upstream")
    choixschem=int(input("Faites votre choix :"))
    

## si le coeff de diffusion est variable : choix de la methode
if "methode" in donnees:
    print("========================================")
    print("Choix de la méthode de calcul du coeff de diffusion :")
    print("1) exacte")
    print("2) moyenne arithmetique")
    print("3) moyenne harmonique")

    choix = -1
    while (int(choix) <= 0) | (int(choix) > 3):
        choix = input("Faites votre choix :")
    choix=int(choix)
    if choix == 1:
        donnees["methode"] = "exacte"
    elif choix == 2:
        donnees["methode"] = "arithmetique"
    elif choix == 3:
        donnees["methode"] = "harmonique"
        print('mla')

    print(donnees)

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


## choix des différentes tailles de maillage pour le trace des courbes
print("========================================")
nb_dep = input("Nombre de mailles du maillage le plus grossier :")
nb_dep = int(nb_dep)

nb_int = input("Pas :")
nb_int = int(nb_int)

nb_fin = input("Nombre de mailles du maillage le plus fin :")
nb_fin = int(nb_fin)

NN = np.linspace(nb_dep, nb_fin, nb_int, dtype=int)
errinf = np.zeros(len(NN))
errL2 = np.zeros(len(NN))
errH1 = np.zeros(len(NN))
pas = np.zeros(len(NN))

print("========================================")

for i in range(len(NN)):  ## boucle sur les différents maillages

    ## Creation du maillage
    if choix == 1:
        maillage = Data.maillage_uniforme(NN[i])
    elif choix == 2:
        maillage = Data.maillage_alterne(NN[i])
    elif choix == 3:
        maillage = Data.maillage_aleatoire(NN[i])
    elif choix == 4:
        maillage = Data.maillage_stretch(NN[i])

    ## Calcul de la solution exacte aux centres des volumes de contrôle

    solexacte = donnees["uexacte"](maillage["centres"])

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
        A, b = Data.vf_laplacien_probg(maillage, donnees,choixschem)

    print("Resolution", NN[i])
    A = A.tocsc()
    sol = umf.spsolve(A, b)
    if choix_pb==4:
        moyenne = Data.InTrap(maillage["centres"], sol)
        print(moyenne)
        sol = sol - np.ones(NN[i]) * moyenne
    ## Calcul des erreurs et de la taille du pas du maillage
    errinf[i] = np.max(np.abs(sol - solexacte))
    errL2[i] = Data.normeL2(maillage, sol - solexacte)
    errH1[i] = Data.normeH1(maillage, sol - solexacte)
    pas[i] = np.max(maillage["mes"])


##**************************************************************************
##     Tracé des courbes d'erreur
##**************************************************************************

plt.figure()
plt.clf()

## Erreur en norme infinie
s1, i1 = np.polyfit(np.log(pas), np.log(errinf), 1)
plt.subplot(1, 3, 1)
plt.loglog(pas, errinf, "-+b")
plt.title("Norme infinie = " + str(round(s1, 2)))

## Erreur en norme L2
s2, i2 = np.polyfit(np.log(pas), np.log(errL2), 1)
plt.subplot(1, 3, 2)
plt.loglog(pas, errL2, "-+b")
plt.title("Norme L2 = " + str(round(s2, 2)))

## Erreur en norme H1
s3, i3 = np.polyfit(np.log(pas), np.log(errH1), 1)
plt.subplot(1, 3, 3)
plt.loglog(pas, errH1, "-+b")
plt.title("Norme H1 = " + str(round(s3, 2)))

plt.suptitle("Ordres de convergence - " + donnees["nom"] + " - " + maillage["nom"])

plt.savefig("test2.png")
plt.show()


print("========================================")
print("Ordre de convergence en norme infinie : ", s1)
print("Ordre de convergence en norme L2 : ", s2)
print("Ordre de convergence en norme H1 : ", s3)
print("==============  FIN  ===================")



##**************************************************************************
##           FIN
##**************************************************************************
