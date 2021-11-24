import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from memory_profiler import profile


class Maillage(object):
    """
    Objet permettant de résoudre l'équation de Poisson avec un maillage donné.
    """
    @profile
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

    def gauss_seidel(self):
        """
        Itération de Gauss-Seidel avec sur-relaxation de paramètre w
        """
        X_old = self.X
        self.X = self.LD_inv * (self.omega * self.S
                                + (1 - self.omega) * self.D
                                - self.omega * self.U * self.X)

        return np.linalg.norm(X_old - self.X)/np.linalg.norm(self.X)

    @profile
    def calculer(self):
        """
        Boucle de calcul
        """
        self.omega = 2/(1 + np.sin(np.pi/(self.J-1)))
        self.L = sp.tril(self.A, format="csc")
        self.D = sp.diags(self.A.diagonal(), 0, format="csc")
        self.U = sp.triu(self.A, format="csc")
        self.LD_inv = sp.linalg.inv(self.D + self.omega * self.L)
        eps = np.inf
        for i in range(100):
            print(i)
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
    m.calculer()

    plt.contour(m.X.reshape(65, 65), 20)
    plt.show()
