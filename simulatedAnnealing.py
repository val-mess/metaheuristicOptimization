import numpy as np
import matplotlib.pyplot as plt
import time

# Liste des valeurs des paramètres
temps_vals = [100, 10, 10]
tempgel_vals = [10, 0.01, 0.01]
alpha_vals = [0.99, 0.75, 0.99]
kequil_vals = [20, 20, 2]

# Efface les données
np.random.seed(10)

# Réglages recuit simulé
Temp = 100 # Température initiale puis courante (décroît au cours de l'algorithme)
TempGel = 0.01  # Température de gel (fixe); constitue un critère d'arrêt
alpha = 0.9  # Taux de décroissance
 # Nombre d'itérations sur un palier de température

# Problème Voyageur de commerce; DEFINITION DES VILLES et paramétrages
# Nbr de villes
N = 50
kEquil = 12*N 
np.random.seed(10)

def plotGraf(cycle, xpos, ypos):
    plt.figure()
    graf = np.append(cycle, cycle[0])
    plt.plot(xpos[graf - 1], ypos[graf - 1])
    plt.plot(xpos[graf - 1], ypos[graf - 1], 'o')
    plt.title('Chemin le meilleur obtenu - $T_0$ = 100'.format(Temp))
    #plt.savefig('temp_ini_100.pdf', format='pdf')
    plt.show()

# choix typologie des villes
Cercle = False  # disposées sur un cercle si Cercle=True sinon aléatoire sur un carré Cercle=False
if Cercle:
    rad = np.random.rand(N) * 2 * np.pi
    xpos = np.cos(rad)
    ypos = np.sin(rad)
else:
    xpos = 2 * np.random.rand(N) - 1
    ypos = 2 * np.random.rand(N) - 1

dx2 = np.square(np.subtract.outer(xpos, xpos))
dy2 = np.square(np.subtract.outer(ypos, ypos))
distance = np.sqrt(dx2 + dy2)

# Cycle aléatoire de DEPART (villes classées de 1 à N)
Cycle = np.arange(1, N + 1)
np.random.shuffle(Cycle)

# Evaluation de la performance initiale
CoutCourant = sum(distance[Cycle - 1, np.roll(Cycle, -1) - 1])
CoutMeilleur = CoutCourant
CycleMeilleur = Cycle
#plotGraf(CycleMeilleur, xpos, ypos)


# fonction COUT
def Eval(d, c, n):
    Z = 0
    for kc in range(n - 1):
        Z += d[c[kc] - 1, c[kc + 1] - 1]
    Z += d[c[n - 1] - 1, c[0] - 1]
    return Z


# fonction PERMUTATION cycle voisin
def Voisin(c, n):
    Ip = np.floor(1 + n * np.random.rand())
    Jp = np.floor(1 + n * np.random.rand())
    mx = int(max(Ip, Jp))
    mn = int(min(Ip, Jp))
    testV = np.random.rand()
    if testV < 0:
        Jp = Ip + 1
    cc = c.copy()
    cc[mn:mx] = np.flipud(cc[mn:mx])
    cycleV = cc
    return cycleV


# fonction d'ACCEPTATION
def Prendre(cout, coutcourant, T):
    DeltaCout = cout - coutcourant
    p = np.random.rand()
    V = False
    if DeltaCout > 0:
        if p < np.exp(-DeltaCout / T):
            V = True
    if DeltaCout < 0:
        V = True
    return V


# fonction DECROISSANCE DE LA TEMPERATURE
def palier(Tc, aT):
    T = Tc * aT  # Loi géométrique simple
    return T


# Initialisation du chronomètre
start_time = time.time()

# ALGORITHME DU RECUIT SIMULE proprement dit
ci = 1
cp = 1
Tabcout=[]
Tabaccep= []
Tabtemp=[]

while Temp > TempGel:
    drap = 0

    for g in range(1, kEquil + 1):  # Boucle sur l'équilibre
        Candidat = Voisin(Cycle, N)
        CoutCandidat = Eval(distance, Candidat, N)

        if Prendre(CoutCandidat, CoutCourant, Temp):
            Cycle = Candidat
            CoutCourant = CoutCandidat
            drap += 1

        if CoutCourant < CoutMeilleur:
            CoutMeilleur = CoutCourant
            CycleMeilleur = Cycle

        Tabcout.append(CoutCourant)
        Tabtemp.append(Temp)
        ci += 1

    Tabaccep.append(drap / kEquil)
    cp += 1
    Temp = palier(Temp, alpha)
    
plotGraf(CycleMeilleur, xpos, ypos)

# Mesure du temps de calcul
temps_calcul = time.time() - start_time


# Mesure du temps de calcul
temps_calcul = time.time() - start_time

# FIN RECUIT SIMULE

# TRACE de l'évolution de la fonction coût en fonction des itérations
plt.figure(2)
plt.plot(Tabtemp[:ci-1])
plt.title('Évolution de la température')
plt.xlabel('Itérations')
plt.ylabel('Température')
plt.savefig('tempGraphe.pdf', format='pdf')

# TRACE de l'évolution de la décroissance de température
plt.figure(3)
plt.plot(Tabcout[:ci-1])
plt.title('Évolution de la fonction coût')
plt.xlabel('Itérations')
plt.ylabel('Coût')
plt.savefig('coutGraphe.pdf', format='pdf')

# TRACE de l'évolution de l'acceptation
plt.figure(1)
plt.plot(Tabaccep[:cp-1])
plt.title("Évolution du taux d'acceptation")
plt.xlabel('Paliers')
plt.ylabel('Taux')

plt.show()

print("Temps de Calcul = ", temps_calcul)
print("Distance optimisée la meilleure obtenue = ", CoutMeilleur)
