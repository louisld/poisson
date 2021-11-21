import matplotlib.pyplot as plt
import numpy as np


class Maillage(object):
    """
    Objet permettant de résoudre l'équation de Poisson avec un maillage donné.
    """

    def __init__(self, I, J):
        """
        N : Taille du maillage
        """
        self.I = I
        self.J = J
        self.A = np.zeros((I*J, I*J))
        self.U = np.zeros(I*J)
        self.S = np.zeros(I*J)

    def discretisation_laplacien(self, condition="Dirichlet"):
        """
        Initialisation des coefficients des matrices du laplacien discret.
        """
        epsx = self.I**2
        epsy = self.J**2

        if condition == "Dirichlet":
            self.A = np.diag([-2*(epsx+epsy)]*(self.I*self.J))
            self.A += np.diag([epsy]*(self.I*self.J-1), k=1)
            self.A += np.diag([epsy]*(self.I*self.J-1), k=-1)
            self.A += np.diag([epsx]*(self.I*self.J-self.J), k=self.J)
            self.A += np.diag([epsx]*(self.I*self.J-self.J), k=-self.J)
        if condition == "Newman":
            self.A = np.diag([-2*epsy-epsx]*self.J + [-2*(epsx+epsy)]
                             * ((self.I-2)*self.J) + [-2*(epsy)-epsx]*self.J,
                             k=0)
            self.A += np.diag([epsy]*(self.I*self.J-1), k=1)
            self.A += np.diag([epsy]*(self.I*self.J-1), k=-1)
            self.A += np.diag([epsx]*(self.I*self.J-self.J), k=self.J)
            self.A += np.diag([epsx]*(self.I*self.J-self.J), k=-self.J)

    def source_horizontale(self, x, y, longueur, valeur):
        i0 = int(x*self.I)
        j0 = int(y*self.J)
        l0 = int(longueur*self.J)
        for i in range(i0, i0+l0):
            self.S[i*self.J + j0] = valeur
            self.U[i*self.J + j0] = valeur

    def gauss_seidel(self):
        """
        Itération de Gauss-Seidel avec sur-relaxation de paramètre w
        """
        self.U = ((1 - self.omega) * self.U
                  + self.omega * (self.A_inv @ (self.A @ self.U - self.S)))

    def calculer(self, iterations):
        self.omega = 2/(1 + np.sin(np.pi/(self.J-1)))
        self.A_inv = np.linalg.inv(np.tril(self.A))
        for k in range(iterations):
            self.gauss_seidel()


m = Maillage(65, 65)
m.discretisation_laplacien()
m.source_horizontale(0.25, 0.4, 0.5, 1)
m.source_horizontale(0.25, 0.6, 0.5, -1)
m.calculer(100)

plt.contour(m.U, 50)
plt.show()
