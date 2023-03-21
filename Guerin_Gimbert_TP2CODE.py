import random
import math
import matplotlib.pyplot as plt
import numpy as nd


def loi_poisson(k, lambda_):
    '''
    
    Parameters
    ----------
    k : INT
        Un entier naturel.
    lambda_ : FLOAT
        Le paramètre de la loi de Poisson.

    Returns
    -------
    FLOAT
        Probabilité que k événements se produisent pour un 
        temps lambda donné.

    '''

    division = (lambda_**k) / math.factorial(k)
    return math.exp(-lambda_) * division

def loi_binomial(k, n, p):
    '''
    
    Parameters
    ----------
    k : INT
        Un entier naturel.
    n : INT
        Nombre d'essais à réaliser.
    p : FLOAT
        Probabilité de succès.

    Returns
    -------
    FLOAT
        Probabilité de k succès pour n expérience.

    '''

    k_parmi_n = math.factorial(n) / (math.factorial(k) * math.factorial(n-k))
    return k_parmi_n * (p**k) * ((1-p)**(n-k))


def loi_normal(x, mu, sigma):
    '''

    Parameters
    ----------
    x : FLOAT
        Réel quelconque.
    mu : FLOAT
        Espérance.
    sigma : FLOAT
        Variance.

    Returns
    -------
    FLOAT
        Probabilité que l'événement x se réalise en suivant la loi normale.

    '''

    puissance = -(1/2) * ((x-mu)/sigma)**2
    val = 1/(sigma*math.sqrt(2*math.pi))
    return val * math.e**puissance

def loi_exponential(x, lambda_): 
    '''

    Parameters
    ----------
    x : FLOAT
        Un réel quelconque.
    lambda_ : FLOAT
        Paramètre de la loi exponentielle.

    Returns
    -------
    FLOAT
        Probabilité que x se réalise en suivant la loi exponentielle.

    '''
    return math.e**(-lambda_*x)

def bon_coef_student(taille_echantillon, pourcent99):
    '''

    Parameters
    ----------
    taille_echantillon : INT
        Taille de l'échantillon mesuré.
    pourcent99 : BOOLEAN
        Si vrai, défini l'intervalle de confiance à 99%.

    Returns
    -------
    r : FLOAT
        Le coefficient correspondant aux paramètres voulu.

    '''
    if(pourcent99):
        if(taille_echantillon == 20):
            r = 2.8453
        else:
            r = 2.5759
    else:
        if(taille_echantillon == 20):
            r = 2.086
        else:
            r = 1.96
    return r

def interval_de_confiance(table, taille_echantillon, variance=None, moyenne_emp=None, variance_emp=None, pourcent99=True):
    ''' 

    Parameters
    ----------
    table : LIST
        DESCRIPTION.
    taille_echantillon : INT
        Taille de l'échantillon.
    variance : FLOAT, optional
        Variance. The default is None.
    moyenne_emp : FLOAT, optional
        Moyenne empirique. The default is None.
    variance_emp : FLOAT, optional
        Variance empirique. The default is None.
    pourcent99 : BOOLEAN, optional
        Si vrai, défini l'intervalle de confiance à 99%. The default is True.

    Returns
    -------
    interval_confiance : TUPLE
        Retourne un tuple contenant deux flottants, la borne inférieure
        et supérieure de l'intervalle de confiance.

    '''
    if(moyenne_emp == None):
        moyenne_emp = moeyenne_empirique(table)
    n = taille_echantillon

    if(variance == None):
        if(variance_emp == None):
            variance_emp = variance_empirique(table, moyenne_emp)

        s = math.sqrt(variance_emp)
    else :
        s = math.sqrt(variance)

    ta = bon_coef_student(n, pourcent99)

    borne_inf = moyenne_emp - ta * (s/math.sqrt(n))
    borne_sup = moyenne_emp + ta * (s/math.sqrt(n))
    interval_confiance = (borne_inf, borne_sup)
    return interval_confiance

def moeyenne_empirique(list_element):
    '''

    Parameters
    ----------
    list_element : LIST
        Liste de nombres réels.

    Returns
    -------
    FLOAT
        Retourne la moyenne empirique.
    
    '''
    moyenne_empirique = 0
    for e in list_element:
        moyenne_empirique += e
    return moyenne_empirique/len(list_element)

def variance_empirique(list_element, moyenne_emp):
    '''

    Parameters
    ----------
    list_element : LIST
        Liste de nombres réels.
    moyenne_emp : FLOAT
        Moyenne empirique.

    Returns
    -------
    FLOAT
        Retorune la variance empirique.

    '''
    sommme_xi_x_emp = 0
    for x in list_element:
        sommme_xi_x_emp += (x - moyenne_emp)**2
    return sommme_xi_x_emp/len(list_element)#s**2



def view_hist(Simule, Theorique, name_loi, str_otption, fonction_repartition=None):
    '''

    Parameters
    ----------
    Simule : (LIST, LIST)
        Liste contenant une liste de valeur et une liste de probabilité.
    Theorique : LIST
        Liste de valeurs théorique.
    name_loi : STRING
        Nom de la loi à afficher.
    str_otption : STRING
        Affiche dynamiquement des valeurs
    fonction_repartition : LIST, optional
        Si non-vide, affiche la fonction de répartition. The default is None.

    Returns
    -------
    None.

    '''
    if(fonction_repartition != None):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(name_loi + " " + str_otption)

        ax1.set_title("la fonction densité (Simulée et théorique) ")
        ax1.bar(Simule[0], Simule[1])

        ax1.plot(Theorique[0], Theorique[1], 'ro')
        ax1.plot(Theorique[0], Theorique[1], 'r+')

        ax2.set_title("la fonction de répartition")
        ax2.plot(fonction_repartition[0], fonction_repartition[1])

    else:
        plt.title(name_loi + " " + str_otption)
        plt.bar(Simule[0], Simule[1])


        plt.plot(Theorique[0], Theorique[1], 'ro')
        plt.plot(Theorique[0], Theorique[1], 'r+')

    plt.show()


    #print("Simule", name_loi, " ", str_otption, " :", interval_de_confiance(Simule))

def fonction_densite(liste_simule, precision=False):
    '''
    céer deux list avec chaque val trouver et leur proba
    
    Parameters
        liste_simule (list): liste de valeur simulée
    Returns
        val, proba (list, list): 
            val sont les valeurs trouvées et proba est la probabilité
            que la valeur se réalise.
    '''
    
    valeur_ocurence = dict()
    for e in liste_simule:
        if(precision):
            e = e*10
            e = int(e)
            e = e/10
        else:
            e = int(e)#pour arrondire sinon on a des graph peux visuel sur la fonction loi normale
        if e in valeur_ocurence:
            valeur_ocurence[e] += 1
        else:
            valeur_ocurence[e] = 1

    val = list()
    proba = list()
    for key, value in valeur_ocurence.items():
        val.append(key)
        proba.append(value/len(liste_simule))

    return [val, proba]

def fonction_repartition_loi_poison(lambda_, borne_sup):
    '''
    
    Parameters
    ----------
    lambda_ : FLOAT
        Le paramètre de la loi de Poisson.
    borne_sup : FLOAT
        Une borne supérieur.

    Returns
    -------
    FLOAT
        Valeur nécessaire à la fonction de répartition.

    '''
    Sn = 0
    for n in range(0, borne_sup):
        Sn = Sn + ((lambda_**n)/(math.factorial(n)))

    return (math.e**(-lambda_))* Sn

def fonction_repartition_loi_normal(x, mu, sigma):
    '''
    
    Parameters
    ----------
    x : FLOAT
        Réel quelconque.
    mu : FLOAT
        Espérance.
    sigma : FLOAT
        Variance.

    Returns
    -------
    FLOAT
        Valeur nécessaire à la fonction de répartition.

    '''
    erf = (x-mu)/(sigma*math.sqrt(2))
    return (1/2) * (1 + math.erf(erf))

def fonction_repartition_loi_exponentielle(x, lambda_):
    '''
   
    Parameters
    ----------
    x : FLOAT
        Un réel quelconque.
    lambda_ : FLOAT
        Paramètre de la loi exponentielle.

    Returns
    -------
    FLOAT
        Valeur nécessaire à la fonction de répartition.

    '''
    exp = (-lambda_ * x)
    return 1 - math.exp(exp)


def loi_poisson_theorique_simule(nb_essais, list_lambda):
    '''
    créé un tableau pour chaque option avec liste théorique et 
    simulée pour la loi de poisson et 
    l'affiche avec view_hist
    
    Parameters
    -------
        nb_essais (int): nombre d'essais
        list_lambda (list): list de tous les lambda
        
    Returns
    -------
        None
        
    '''
    for lambda_ in list_lambda: #pour chaque lambda
        Simule=[] #on utilise numpy

        proba_theorique = []
        val_theorique = []

        repartition = []

        for i in range(nb_essais):#pour chaque tirage
            Simule.append(nd.random.poisson(lambda_))
        densite_simule = fonction_densite(Simule)

        for val in range(int(min(densite_simule[0])), int(max(densite_simule[0])+1)):
            proba_theorique.append(loi_poisson(val, lambda_))
            val_theorique.append(val)
            repartition.append((val, fonction_repartition_loi_poison(lambda_, val)))

        #Afficher
        str_otption = "λ = " + str(lambda_)
        view_hist(densite_simule, (val_theorique, proba_theorique), "Loi Poisson", str_otption, repartition)

def loi_binomial_theorique_simule(nb_essais, list_n_p):
    '''
    Créé un tableau pour chaque option avec la liste théorique et 
    simulée pour la loi binomiale et l'affiche avec view_hist
    
    Parameters
    -------
        nb_essais (int): nombre d'essais
        list_n_p (list de tuple): list de tuple avec (n, p)
        
    Returns
    -------
        None
        
    '''
    for n, p in list_n_p: # pour chaque tuple n, p
        Simule=[] #on utilise numpy

        proba_theorique = []
        val_theorique = []

        for i in range(nb_essais):#pour chaque tirage
            Simule.append(nd.random.binomial(n, p))
        densite_simule = fonction_densite(Simule)

        for val in range(int(min(densite_simule[0])), int(max(densite_simule[0])+1)):
            proba_theorique.append(loi_binomial(val, n, p))
            val_theorique.append(val)

        #Afficher
        str_otption = "n = " + str(n) + ", p = " + str(p)
        view_hist(densite_simule, (val_theorique, proba_theorique), "Loi Binomiale", str_otption)

def loi_normal_theorique_simule(nb_essais, list_mu_sigma):
    '''
    Créé un tableau pour chaque option avec la liste théorique 
    et simulée pour la loi normale et l'affiche avec view_hist
    
    Parameters
    -------
        nb_essais (int): nombre d'essais
        list_mu_sigma (list de tuple): list de tuple avec (mu, sigma)
        
    Returns
    -------
        None
    '''
    for mu, sigma in list_mu_sigma:# pour chaque tuple mu, sigma
        Simule=[] #on utilise numpy

        proba_theorique = []
        val_theorique = []

        repartition = []

        for i in range(nb_essais):#pour chaque tirage
            Simule.append(nd.random.normal(mu, sigma))

        densite_simule = fonction_densite(Simule)

        for val in range(int(min(densite_simule[0])), int(max(densite_simule[0])+1)):
            proba_theorique.append(loi_normal(val, mu, sigma))
            val_theorique.append(val)
            repartition.append((val,fonction_repartition_loi_normal(val, mu, sigma)))

        #Afficher
        str_otption = "μ = " + str(mu) + ", σ = " + str(sigma)
        view_hist(densite_simule, (val_theorique, proba_theorique), "Loi Normale", str_otption, repartition)

def loi_exponential_theorique_simule(nb_essais, list_lambda):
    '''
    Créé un tableau pour chaque option avec la liste théorique 
    et simulée pour la loi exponentielle et l'affiche avec view_hist
    
    Parameters
    -------
        nb_essais (int): nombre d'essais
        list_lambda (list): list de tout les lambda
        
    Returns
    -------
        None
        
    '''
    for lambda_ in list_lambda: #pour chaque lambda
        Simule=[] #on utilise numpy

        proba_theorique = []
        val_theorique = []
        repartition = []
        
        for i in range(nb_essais):#pour chaque tirage
            Simule.append(nd.random.exponential(lambda_))

        densite_simule = fonction_densite(Simule)

        for val in range(int(min(densite_simule[0])), int(max(densite_simule[0])+1)):
            proba_theorique.append(loi_exponential(val, lambda_))
            val_theorique.append(val)
            repartition.append((val, fonction_repartition_loi_exponentielle(val, lambda_)))

        #Afficher
        str_otption = "λ = " + str(lambda_)
        view_hist(densite_simule, (val_theorique, proba_theorique), "Loi exponentielle", str_otption, repartition)

# Lois discrete
nb_essais = 1000
loi_poisson_theorique_simule(nb_essais, [1, 10, 30])
loi_binomial_theorique_simule(nb_essais, [(50, 0.5),(50, 0.7), (50, 0.2)])

loi_normal_theorique_simule(nb_essais, [(25, math.sqrt(50*0.25))])
loi_exponential_theorique_simule(nb_essais, [0.5, 2])


#Temp de temp_reaction
list_theorique = [0.98, 1.4, 0.84, 0.88, 0.54, 0.68, 1.35, 0.76, 0.72, 0.99, 0.88, 0.75, 0.49, 1.09, 0.68, 0.60, 1.13, 1.35, 1.13, 0.91]

mu = sum(list_theorique)/len(list_theorique)
sigma = math.sqrt(0.2)

proba_theorique = []
val_theorique = []

densite_simule = fonction_densite(list_theorique, True)


for val in range(0, int(max(densite_simule[0])+1)*10):
    proba_theorique.append(loi_normal(val/10, mu, sigma))
    val_theorique.append(val/10)

#Afficher
str_otption = "μ = " + str(mu) + ", σ = " + str(sigma)
view_hist(densite_simule, (val_theorique, proba_theorique), "normal temps reaction", str_otption)

m_emp_temp = moeyenne_empirique(list_theorique)
v_emp_temp = variance_empirique(list_theorique, m_emp_temp)

print("la moyenne empirique", m_emp_temp)
print("la variance empirique", v_emp_temp)
print("\nEn supposant que variance empirique² = 0.2\n---------------------------------------\n")
print("intervalle de confiance `a 1 − α = 95%", interval_de_confiance(list_theorique, len(list_theorique), sigma, moyenne_emp=m_emp_temp, variance_emp=None, pourcent99=False))
print("intervalle de confiance `a 1 − α = 99%", interval_de_confiance(list_theorique, len(list_theorique), sigma, moyenne_emp=m_emp_temp, variance_emp=None, pourcent99=True))
print("\nEn supposant que variance empirique = Sn\n---------------------------------------\n")
print("intervalle de confiance `a 1 − α = 95%", interval_de_confiance(list_theorique, taille_echantillon=len(list_theorique), variance=None, moyenne_emp=m_emp_temp, variance_emp=v_emp_temp, pourcent99=False))
print("intervalle de confiance `a 1 − α = 99%", interval_de_confiance(list_theorique, taille_echantillon=len(list_theorique), variance=None, moyenne_emp=m_emp_temp, variance_emp=v_emp_temp, pourcent99=True))

#Estimation d'une Proportion
nb_etu = 1000
suivi_cours = 637
pourcentage_etu_suivant_cours = suivi_cours/nb_etu

print("\nIntervalle de confiance des élèves ayant algorithmie :")
print("Moyenne = ", pourcentage_etu_suivant_cours)
print("intervalle de confiance `a 1 − α = 95%", interval_de_confiance(table=None, taille_echantillon=nb_etu, variance=None, moyenne_emp=pourcentage_etu_suivant_cours, variance_emp=v_emp_temp, pourcent99=False))


#verifier le théorème centrale limite
nb_essais = 1000
nb_tirage = 12
mu = nb_tirage/2
sigma = 0.5

proba_theorique = []
val_theorique = []

valeur_ocurence = dict()
for t in  range(nb_essais):
    somme = 0
    for i in range(nb_tirage):#pour chaque tirage
        somme += nd.random.randint(2)

    if somme in valeur_ocurence:
        valeur_ocurence[somme] += 1
    else:
        valeur_ocurence[somme] = 1

val_Simuler = list()
proba_Simuler = list()
for key, value in valeur_ocurence.items():
    val_Simuler.append(key)
    proba_Simuler.append(value/(nb_essais))

for val in range(int(min(val_Simuler)), int(max(val_Simuler)+1)):
    proba_theorique.append(loi_normal(val, mu, sigma))
    val_theorique.append(val)

#Afficher
str_otption = "μ = " + str(mu) + ", σ = " + str(sigma)
view_hist((val_Simuler, proba_Simuler), (val_theorique, proba_theorique), "normal", str_otption)