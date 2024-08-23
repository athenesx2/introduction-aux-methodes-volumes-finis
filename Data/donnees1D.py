############################################################
####           Divers schemas de volumes finis        ######
####               pour pbs elliptiques 1D            ######
####                                                  ######
####                 Jeux de données                  ######
####                                                  ######
############################################################


##**************************************************************************
##    Le format des jeux de données
##**************************************************************************
##  Chaque jeu de données contient (ou pas !) un certain nombre d'éléments
##  *  nom :  chaîne de caractère qui décrit le cas test
##  *  source    : fonction x->f(x) qui décrit le terme source
##  *  uexacte   : fonction x->ue(x) qui donne la solution exacte, si elle existe
##  *  coeff_k   : fonction x->k(x) qui décrit le coefficient de diffusion
##  *  bordD     : fonction x->borD(x) qui renvoie les données de Dirichlet au bord
##  *  bordN     : fonction qui renvoie la donnee de Neumann au bord
##  *  bordF     : fonction qui renvoie la donnee de Fourier au bord
##  *  coeffF    : coefficient de la condition aux limites de Fourier
##  *  dirac     : position dans l'intervalle ]0,1[ d'un terme de Dirac en second membre
##  *  moyenne   : valeur de la moyenne de la solution (utile pour Neumann)
##  *  methode   : choix de la methode de calcul du coefficient de diffusion
##                 à l'interface entre deux volumes de contrôle
############################################################################

import numpy as np


def coeff_cte(x):
    try:
        return np.ones(len(x))
    except TypeError:
        return 1


cas_test = list()

##************************************##
## un polynome de degré 2 nul au bord ##
##               k(x)=1               ##
##************************************##

fourier1 = 10


def ue1(x):
    return 0.5 * x * (1 - x)


def flux_ue1(x):
    return -0.5


def fourier_ue1(x):
    return flux_ue1(x) + fourier1 * ue1(x)


def f1(x):
    return np.ones(len(x))


cas1 = {
    "nom": "u(x)=x(1-x)/2, k(x)=1",
    "uexacte": ue1,
    "bordD": ue1,
    "source": f1,
    "moyenne": 1 / 12,
    "bordN": flux_ue1,
    "bordF": fourier_ue1,
    "coeffF": fourier1,
    "coeff_k": coeff_cte,
}

cas_test = [cas1]


##************************************##
##           u(x)=sin(pi x)           ##
##               k(x)=1               ##
##************************************##


def ue2(x):
    return np.sin(np.pi * x)


def flux_ue2(x):
    return -np.pi


def f2(x):
    return np.pi * np.pi * np.sin(np.pi * x)


cas2 = {
    "nom": "u(x)=sin(pi*x), k(x)=1",
    "uexacte": ue2,
    "bordD": ue2,
    "source": f2,
    "moyenne": 2 / np.pi,
    "bordN": flux_ue2,
    "coeff_k": coeff_cte,
}

cas_test.append(cas2)


##************************************##
##           u(x)=cos(pi x)           ##
##               k(x)=1               ##
##************************************##


def ue2_bis(x):
    return np.cos(np.pi * x)


def flux_ue2_bis(x):
    return 0


def f2_bis(x):
    return np.pi * np.pi * np.cos(np.pi * x)


cas2_bis = {
    "nom": "u(x)=cos(pi*x), k(x)=1",
    "uexacte": ue2_bis,
    "bordD": ue2_bis,
    "source": f2_bis,
    "moyenne": 0,
    "bordN": flux_ue2_bis,
    "coeff_k": coeff_cte,
}


cas_test.append(cas2_bis)

##************************************##
##       u(x)=tanh((x-0.3)/eps)       ##
##               k(x)=1               ##
##************************************##

param3 = 0.1


def ue3(x):
    return np.tanh((x - 0.3) / param3)


def flux_ue3(x):
    return (
        -(1 / param3)
        * (1 - np.tanh((x - 0.3) / param3) ** 2)
        * ((x - 0.5) / np.abs(x - 0.5))
    )


def f3(x):
    return (
        2
        / param3**2
        * np.tanh((x - 0.3) / param3)
        * (1 - (np.tanh((x - 0.3) / param3) ** 2))
    )


cas3 = {
    "nom": "u(x)=tanh((x-0.3)/eps), eps=" + str(param3) + ",k(x)=1",
    "uexacte": ue3,
    "bordD": ue3,
    "source": f3,
    "moyenne": param3
    * (np.log(np.cosh((1 - 0.3) / param3)) - np.log(np.cosh(-0.3 / param3))),
    "bordN": flux_ue3,
    "coeff_k": coeff_cte,
}

cas_test.append(cas3)

##*************************************##
##          u(x)=np.sin(pi x)          ##
##       k(x)=(1+1/2*np.cos(pi x))     ##
## par defaut le coefficient k_{i+1/2} ##
##      est évalué de façon exacte     ##
##*************************************##


def ue4(x):
    return np.sin(np.pi * x)


def k4(x):
    return 1 + 0.5 * np.cos(np.pi * x)


def f4(x):
    return np.pi * np.pi * np.sin(np.pi * x) * (1 + np.cos(np.pi * x))


def flux_ue4(x):
    return -np.pi * k4(x)


cas4 = {
    "nom": "u(x)=sin(pi*x), k(x)=1+0.5*np.cos(pi*x)",
    "uexacte": ue4,
    "bordD": ue4,
    "source": f4,
    "moyenne": 2 / np.pi,
    "bordN": flux_ue4,
    "coeff_k": k4,
    "methode": "exacte",
}

cas_test.append(cas4)


##********************************************##
##  u(x)=une fonction régulière par morceaux  ##
##         k(x)=constant par morceaux         ##
##    par defaut le coefficient k_{i+1/2}     ##
##         est évalué de façon exacte         ##
##********************************************##

k_gauche = 1
k_droit = 10


def kdis(x):
    return (x < 0.5) * (k_gauche) + (x >= 0.5) * (k_droit)


def fdis(x):
    return (x < 0.5) * (
        -2 * kdis(0) * (20 * (x - 0.5) ** 3 + 12 * (x - 0.5) ** 2 + 6 * (x - 0.5) + 2)
        - kdis(0) * (x - 0.5) * (60 * (x - 0.5) ** 2 + 24 * (x - 0.5) + 6)
    ) + (x >= 0.5) * (
        -2
        * kdis(1)
        * (kdis(0) * np.pi * np.cos(np.pi * x) - 5 * np.pi * np.sin(np.pi * x))
        + kdis(1)
        * (x - 0.5)
        * (
            kdis(0) * np.pi * np.pi * np.sin(np.pi * x)
            + 5 * np.pi * np.pi * np.cos(np.pi * x)
        )
    )


def udis(x):
    return (x < 0.5) * (
        (x - 0.5)
        * (
            5 * (x - 0.5) ** 4
            + 4 * (x - 0.5) ** 3
            + 3 * (x - 0.5) ** 2
            + 2 * (x - 0.5)
            + kdis(1)
        )
    ) + (x >= 0.5) * ((x - 0.5) * (kdis(0) * np.sin(np.pi * x) + 5 * np.cos(np.pi * x)))


casdis = {
    "nom": "u=reguliere par morceaux, k="
    + str(k_gauche)
    + " pour x<1/2 et k="
    + str(k_droit)
    + " pour x>1/2",
    "uexacte": udis,
    "bordD": udis,
    "source": fdis,
    "coeff_k": kdis,
    "methode": "exacte",
}

cas_test.append(casdis)


##*************************************##
##       u(x)=affine par morceaux      ##
##                k(x)=1               ##
##           Terme de dirac            ##
##*************************************##

x_dirac = 0.5


def ue_aff_pm(x):
    return -0.5 * np.abs(x - x_dirac)


def f_nulle(x):
    return 0 * x


casaff_pm = {
    "nom": "u affine par morceaux. Dirac en x=" + str(x_dirac) + ", k(x)=1",
    "uexacte": ue_aff_pm,
    "source": f_nulle,
    "bordD": ue_aff_pm,
    "coeff_k": coeff_cte,
    "dirac": x_dirac,
}


cas_test.append(casaff_pm)


##*************************************##
##       u(x)=abs(np.sin(2 pi x))      ##
##                k(x)=1               ##
##       Terme de dirac en x=0.5       ##
##*************************************##


def ue5(x):
    return -np.sin(2 * np.pi * (x % 0.5)) / (4 * np.pi)


def flux_ue5(x):
    return 1 / 2


def f5(x):
    return -np.pi * np.sin(2 * np.pi * (x % 0.5))


cas5 = {
    "nom": "u =|sin|, Terme source avec Dirac en 1/2, k(x)=1",
    "uexacte": ue5,
    "source": f5,
    "bordD": ue5,
    "coeff_k": coeff_cte,
    "moyenne": -1 / (2 * np.pi**2),
    "bordN": flux_ue5,
    "dirac": 0.5,
}

cas_test.append(cas5)


a = 750
b = 10


def ue2g(x):
    return -x * (1 - x) * np.exp(x)


def flux_ue2g(x):
    return -(x - 1 + x**2) * np.exp(x)


def derflux_e2g(x):
    return (3 * x + x**2) * np.exp(x)


def f2g(x):
    return -derflux_e2g(x) - a * flux_ue2g(x) + b * ue2g(x)


casg = {
    "nom": "u(x)=-x(1-x)exp(x), -u''+u'+u=f",
    "uexacte": ue2g,
    "bordD": ue2g,
    "source": f2g,
    "moyenne": 1 / 12,
    "coeffb": b,
    "coeffa": a,
}


cas_test.append(casg)


def ue1g(x):
    return np.sin(np.pi * x)


def flux_ue1g(x):
    return np.pi * np.cos(np.pi * x)


def derflux_ue1g(x):
    return -np.pi * np.pi * np.sin(np.pi * x)




def f1g(x):
    return -derflux_ue1g(x) + a * flux_ue1g(x) + b * ue1g(x)


casg = {
    "nom": "u(x)=sin(x pi), -u''+u'+u=f",
    "uexacte": ue1g,
    "bordD": ue1g,
    "source": f1g,
    "moyenne": 1 / 12,
    "coeffb": b,
    "coeffa": a,
}


cas_test.append(casg)
