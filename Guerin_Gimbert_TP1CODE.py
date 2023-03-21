# -*- coding: utf-8 -*-
#   GIMBERT Vincent 
#   GUERIN Clément
import matplotlib.pyplot as plt
import random
import math as math
import numpy as np#pour avoire des graph
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

#les tableau des temperatures
t_ete = [[3500,10], [2800,13], [1300,20], [750,25], [300,30], [900,22], [1800,18], [3100,11]] #tableau de valeur pour l'ete
t_hiver = [[3500,-15], [2800,-11], [1300,0], [750,3], [300,10], [900,2], [1800,-2], [3100,-13]] #tableau de val pour l'hiver
test = [[0,0], [1,1], [10,10], [400,400], [1000,1000], [1500,1500], [3500,3500]] #pour verifier le resultat de la fonction affine

t_ete_sklearn = ([[3500], [2800], [1300], [750], [300], [900], [1800], [3100]], 
                 [[10], [13], [20], [25], [30], [22], [18], [11]])

t_hiver_sklearn = ([[3500], [2800], [1300], [750], [300], [900], [1800], [3100]], 
                 [[-15], [-11], [0], [3], [10], [2], [-2], [-13]])

def predict_value(table, a, b):
    """
    

    Parameters
    ----------
    table : LIST
        Liste contenant des données rangés de manière similaire à t_ete_sklearn ou t_hiver_sklearn
    a : FLOAT
        Coefficient directeur de la fonction affine
    b : FLOAT
        Constante de la fonction affine
        

    Returns
    -------
    liste : LIST
        Retourne la liste des résultats de la fonction affine a*x+b

    """
    liste = list()
    for x in table:
        liste.append(fonction_affine(x[0], a, b)) #ici x[0] car on manipule une liste de liste ne contenant qu'un élément
    return liste

def liste_test_alea(xy=False):

    def temperature_fonction_hauteur(h):
        return 1*h+0

    t_val_Alea = list()
    #pour avoire x et y separer dans 2 tableau diferent
    t_val_Alea_x = list()
    t_val_Alea_y = list()
    for i in range(100):
        bruit = (random.random() * 1.5)
        altitude = random.randrange(4000)
        temp =  bruit + temperature_fonction_hauteur(altitude)
        t_val_Alea.append([altitude, temp])

        t_val_Alea_x.append([altitude])
        t_val_Alea_y.append([temp])

    if xy:#si xy true renvoyer les 2 tableau
        return t_val_Alea_x, t_val_Alea_y
    return t_val_Alea



def moindre_carre(table):
    '''
    
    Parameters
    -------
        table : LIST 
            Liste de liste contennant des données tel que le tableau d'été ou d'hiver.
        
        
    Returns
    -------
        a : FLOAT
            Coefficient directeur de la fonction affine
        b : FLOAT
            Constante de la fonction affine
    '''
    resultat_somme_xi = 0
    resultat_somme_yi = 0
    resultat_xi_yi = 0
    carre_xi = 0
    n = len(table)
    for x, y in table:
            resultat_xi_yi += x * y
            resultat_somme_xi += x
            resultat_somme_yi += y
            carre_xi += x**2

    somme_xi_plus_yi = resultat_somme_xi * resultat_somme_yi
    a = ((n*resultat_xi_yi) - somme_xi_plus_yi)/((n*carre_xi) - resultat_somme_xi**2)
    b = (resultat_somme_yi/n) - (a*(resultat_somme_xi/n))

    return a,b

def fonction_affine(x, a, b):
    return a*x+b

def moyenne_y(table, n):
    moyenne_total_y = 0

    for x, y in table:
        moyenne_total_y += y

    return moyenne_total_y/n


def maximum_vraisemblance(table, a, b):
    '''

    Parameters
    -------
        table : LIST 
            Liste de liste contennant des données tel que le tableau d'été ou d'hiver.
        a : FLOAT
            Coefficient directeur de la fonction affine
        b : FLOAT
            Constante de la fonction affine


    Returns
    -------
        max_vraisemblance : FLOAT
            Maximum de vraisemblance, la marge d'erreur. Plus ce chiffre est élevé, 
            plus la marge d'erreur est grande.
    '''
    n = len(table)
    x = 0
    y = 0
    z = 0
    for x, y in table:
        z += (y-a*x-b)**2
    max_vraisemblance = (1/n)*z
    return max_vraisemblance

def derive_partiel(table, a, b):
    '''
    
    Parameters
    -------
        table : LIST 
            Liste de liste contennant des données tel que le tableau d'été ou d'hiver.
        a : FLOAT
            Coefficient directeur de la fonction affine
        b : FLOAT
            Constante de la fonction affine

    Returns
    -------
        derive_partiel_a : FLOAT
            Dérivée partielle pour a
        derive_partiel_b : FLOAT
            Dérivée partielle pour b
    '''
    n = len(table)
    xi_fois_somme_z = 0
    somme_z = 0              

    for x, y in table:
        xi_fois_somme_z += x * (a*x+b-y)
        somme_z += (a*x+b-y)

    derive_partiel_a = (1/n) * xi_fois_somme_z
    derive_partiel_b = (1/n) * somme_z


    return derive_partiel_a, derive_partiel_b

def J(table, a, b):
    '''

    Parameters
    -------
        table : LIST 
            Liste de liste contennant des données tel que le tableau d'été ou d'hiver.
        a : FLOAT
            Coefficient directeur de la fonction affine
        b : FLOAT
            Constante de la fonction affine

    Returns
    -------
        somme_j : FLOAT
            Renvoit J pour calculer la méthode d'optimisation
    '''
    somme_j = 0

    for x, y in table:
        somme_j += (a*x+b-y)**2

    return somme_j

def methode_optimisation(table, gamma):
    '''
    
    Parameters
    -------
        table : LIST 
            Liste de liste contennant des données tel que le tableau d'été ou d'hiver.
        gamma : FLOAT
            La valeur du pas

    Returns
    -------
        jk : FLOAT
            Retourne l'estimateur méthode optimisation
        
    '''
    n = len(table)

    ak = 0
    bk = 0
    jk = J(table, ak, bk)
    listAk = []
    listBk = []
    listJk = []
    k = 0
    while True:
        listAk.append([ak, k])
        listBk.append([bk, k])
        listJk.append([jk, k])
        derive_partiel_a, derive_partiel_b = derive_partiel(table, ak, bk)

        ak = ak - (gamma * derive_partiel_a)
        bk = bk - (gamma * derive_partiel_b)

        jkplus = J(table, ak, bk)
        erreur = abs(jkplus - jk)
        jk = jkplus

        if erreur >= 10**-3 :
            break
        k+=1

    return jk


def rmse(table, a, b):
    '''
    
    Parameters
    -------
        table : LIST 
            Liste de liste contennant des données tel que le tableau d'été ou d'hiver.
        a : FLOAT
            Coefficient directeur de la fonction affine
        b : FLOAT
            Constante de la fonction affine


    Returns
    -------
        rmse : FLOAT
            Root-mean-square, l'erreur quadratique moyenne
    '''
    n = len(table)
    somme_moindre_carre_yi_y = 0

    for x, y in table:
        somme_moindre_carre_yi_y += (fonction_affine(x, a, b) - y)**2
        
    rmse = math.sqrt(somme_moindre_carre_yi_y/n)
    return rmse


def coefficient_determination(table, a, b):
    '''

    Parameters
    -------
        table : LIST 
            Liste de liste contennant des données tel que le tableau d'été ou d'hiver.
        a : FLOAT
            Coefficient directeur de la fonction affine
        b : FLOAT
            Constante de la fonction affine
            

    Returns
    -------
        coef_deter : FLOAT
            Coefficient de détermination
    '''
    n = len(table)
    moyenne_total_y = moyenne_y(table, n)
    somme_yi_y = 0
    somme_moindre_carre_yi_y = 0

    for x, y in table:
        somme_moindre_carre_yi_y += (fonction_affine(x, a, b) - moyenne_total_y)**2
        somme_yi_y += (y - moyenne_total_y)**2
        
    coef_deter = somme_moindre_carre_yi_y / somme_yi_y
    return coef_deter

def afficher_table(table):
    """

    Parameters
    ----------
    table : LIST
        Table contenant des données

    Returns
    -------
    None.

    """
    list_x = list()
    list_y = list()
    for x, y in table:
        list_x.append(x)
        list_y.append(y)

    a, b = moindre_carre(table)

    plt.plot([fonction_affine(x, a, b) for x in range(0, 3500, 1)]) #pous simuler visuelement la fonction affine

    plt.scatter(list_x,list_y)
    plt.xlabel('altitude')
    plt.ylabel('température')
    plt.title("Table d'été" if table == t_ete else "Table d'hiver")
    plt.show()


#------------------------------------------------------------------------------------------#
#                                                                                          #
#                               Manipulation de nos fonctions                              #
#                                                                                          #
#------------------------------------------------------------------------------------------#

tableChoose = liste_test_alea()
afficher_table(t_ete)
afficher_table(t_hiver)


a, b = moindre_carre(tableChoose)
print("a:", a, "b:", b)


print("max vraisemblance", maximum_vraisemblance(tableChoose, a, b))
print("methode opti", methode_optimisation(tableChoose, 0.5))
print("rmse", rmse(tableChoose, a, b))
print("coef determination", coefficient_determination(tableChoose, a, b))

print("-----------------\n")
print("question")
#question 2

#-----------------------------------Pour Table été-----------------------------------------#
a_E, b_E = moindre_carre(t_ete)
a_H, b_H = moindre_carre(t_hiver)
print("point carre été = ", a_E," x +",b_E)
print("max vraisemblance", maximum_vraisemblance(t_ete, a_E, b_E))
print("methode opti", methode_optimisation(t_ete, 0.5))
print("rmse", rmse(t_ete, a_E, b_E))
print("coef determination", coefficient_determination(t_ete, a_E, b_E))
print("-----------------\n")


#-----------------------------------Pour Table hiver---------------------------------------#
print("point carre hiver = ", a_H," x +",b_H)
print("max vraisemblance", maximum_vraisemblance(t_hiver, a_H, b_H))
print("methode opti", methode_optimisation(t_hiver, 0.5))
print("rmse", rmse(t_hiver, a_H, b_H))
print("coef determination", coefficient_determination(t_hiver, a_H, b_H))
print("-----------------\n")

print("REPONSE AUX 3 QUESTIONS DU TP")
print("temperature a 1000m en été :", fonction_affine(1000, a_E, b_E))
print("temperature a 1000m en hiver :", fonction_affine(1000, a_H, b_H))

#question 3
a_M = (a_E + a_H)/2#moyenne en fonction des deux saisons
b_M = - (a_M * 300 - 15)#temperature a l'altitude 0 pour trouver il faut resoudre -0.006548553557466771 * 300 +  b_M = 15
print("temperature a 1000m quand il fait 15°C à 300m :", fonction_affine(1000, a_M, b_M))





#------------------------------------------------------------------------------------------#
#                                                                                          #
#                         Manipulation de la bibliothèque sklearn                          #
#                                                                                          #
#------------------------------------------------------------------------------------------#

#-----------------------------------Pour Table été-----------------------------------------#
reg = linear_model.LinearRegression().fit(t_ete_sklearn[0], t_ete_sklearn[1])

a = reg.coef_[0][0]     #on est obligé de spécifier [0][0] car coef_ retourne un tableau 
                        #à 2 dimensions avec pour unique valeur le coefficient

b = reg.intercept_[0]   #Même chose ici, mais c'est un tableau 1D.
print("------------------------\n")
print("BIBLIOTHEQUE SKLEARN")
print("\n")
print("Table été :\n a = ", a,"\n b = ", b)
print(" max vraisemblance = ", maximum_vraisemblance(t_ete, a, b))
print(" methode opti = ", methode_optimisation(t_ete, 0.5))
print(" coefficient de détermination = ", reg.score(t_ete_sklearn[0], t_ete_sklearn[1]))
print(" RMSE = ", mean_squared_error(t_ete_sklearn[1], predict_value(t_ete_sklearn[0], a, b)))
print("\n")

#---------------------------------Pour Table Hiver-----------------------------------------#
reg = linear_model.LinearRegression().fit(t_hiver_sklearn[0], t_hiver_sklearn[1])

a = reg.coef_[0][0]
b = reg.intercept_[0]

print("Table hiver :\n a = ", a,"\n b = ", b)
print(" max vraisemblance = ", maximum_vraisemblance(t_hiver, a, b))
print(" methode opti = ", methode_optimisation(t_hiver, 0.5))
print(" coefficient de détermination = ", reg.score(t_hiver_sklearn[0], t_hiver_sklearn[1]))
print(" RMSE = ", mean_squared_error(t_hiver_sklearn[1], predict_value(t_hiver_sklearn[0], a, b)))
print("\n")