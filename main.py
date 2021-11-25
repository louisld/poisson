import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from memory_profiler import profile
from mpl_toolkits.mplot3d import Axes3D


class Maillage(object):
    """
    Objet permettant de résoudre l'équation de Poisson avec un maillage donné.
    """

    def __init__(self, I, J):
        """
        I, J : Taille du maillage
        """
        self.I = I
        self.J = J
        self.A = sp.lil_matrix((I*J, I*J))
        self.X = np.zeros(I*J)
        self.S = np.zeros(I*J)

    def discretisation_laplacien(self, condition="Dirichlet"):
        """
        Initialisation des coefficients des matrices du laplacien discret soit
        avec les conditions de Dirichlet soit avec celles de Newman.
        """
        epsx = self.I**2
        epsy = self.J**2

        self.A.setdiag(-2*(epsx + epsy))
        self.A.setdiag(epsy, 1)
        self.A.setdiag(epsy, -1)
        self.A.setdiag(epsx, self.J)
        self.A.setdiag(epsx, -self.J)

    def source_horizontale(self, x, y, longueur, valeur):
        """
        Création d'une source de charge suivant une ligne horizontale.
        """
        i0 = int(x*self.I)
        j0 = int(y*self.J)
        l0 = int(longueur*self.J)
        for i in range(i0, i0+l0):
            self.S[i*self.J + j0] = valeur
            self.X[i*self.J + j0] = valeur

    def condition_dirichlet(self, Vmin, Vmax):
        """
        Applique les conditions de dirichlet : les sources sont nuls au bord
        """
        V = (Vmax + Vmin)/2
        for j in range(self.J):
            # Pour i = 0
            self.X[self.J * j] = V
            # Pour i = self.I
            self.X[self.I + (self.J-1) * j] = V
        for i in range(self.I):
            # Pour j = 0
            self.X[i] = V
            # Pour j = self.J
            self.X[i + self.J*(self.J-1)] = V

    def gauss_seidel(self):
        """
        Itération de Gauss-Seidel avec sur-relaxation de paramètre omega
        """
        X_old = self.X
        self.X = self.X - self.LD_inv * (self.A * self.X - self.S)
        return np.linalg.norm(X_old - self.X)/np.linalg.norm(self.X)

    def sor(self):
        """
        Itération de Gauss-Seidel avec sur-relaxation de paramètre omega.
        Méthode de calcul trop lente.
        """
        X_old = self.X
        for i in range(self.I*self.J):
            s1 = 0
            s2 = 0
            for j in range(self.J*self.J):
                if j < i:
                    s1 += self.A[i, j] * self.X[j]
                if j > i:
                    s2 += self.A[i, j] * X_old[j]
            self.X[i] = (1 - self.omega) * self.X[i] + self.omega / \
                self.A[i, i] * (self.S[i] - s1 - s2)
        return np.linalg.norm(X_old - self.X)/np.linalg.norm(self.X)

    @profile
    def calculer(self, seuil):
        """
        Boucle de calcul
        """
        self.omega = 2/(1 + np.sin(np.pi/(self.J-1)))
        self.L = sp.tril(self.A, format="csc")
        self.D = sp.diags(self.A.diagonal(), 0, format="csc")
        self.U = sp.triu(self.A, format="csc")
        self.LD_inv = sp.linalg.inv(self.D + self.L)
        eps = np.inf
        while eps > seuil:
            print(eps)
            eps = self.gauss_seidel()


if __name__ == '__main__':
    """
    Création d'un maillage 65x65 avec deux sources linéaire de charges 1 et -1
    qui correspondent aux deux armatures du condensateur. On calcule ensuite la
    solution grâce à la méthode de Gauss-Siedel.
    """
    m = Maillage(65, 65)
    m.discretisation_laplacien()
    m.source_horizontale(0.25, 0.4, 0.5, 1)
    m.source_horizontale(0.25, 0.6, 0.5, -1)
    m.condition_dirichlet(-1, 1)
    m.calculer(1e-3)
    exit()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(range(65), range(65))
    ax.plot_surface(X, Y, m.X.reshape(65, 65), cmap=cm.coolwarm)
    plt.show()
