import matplotlib.pyplot as plt
import numpy as np


class Maillage(object):
    def __init__(self, N):
        """
        Matrice des coefficients :
        A : à gauche
        B : en haut
        C : à droite
        D : en bas
        E : au centre
        S : source
        """
        self.N = N
        self.M = np.zeros((self.N+2, self.N+2))
        self.A = np.zeros((self.N+2, self.N+2))
        self.B = np.zeros((self.N+2, self.N+2))
        self.C = np.zeros((self.N+2, self.N+2))
        self.D = np.zeros((self.N+2, self.N+2))
        self.E = np.zeros((self.N+2, self.N+2))
        self.S = np.zeros((self.N+2, self.N+2))
        self.U = np.zeros((self.N+2, self.N+2))

    def discretisation_laplacien(self):
        for i in range(1, self.N+1):
            for j in range(1, self.N+1):
                self.A[i, j] = 1
                self.B[i, j] = 1
                self.C[i, j] = 1
                self.D[i, j] = 1
                self.E[i, j] = -4
        for i in range(1, self.N+1):
            self.A[i, 1] = 0
            self.A[i, self.N] = 0
            self.B[i, 1] = 0
            self.B[i, self.N] = 0
            self.C[i, 1] = 0
            self.C[i, self.N] = 0
            self.D[i, 1] = 0
            self.D[i, self.N] = 0
            self.E[i, 1] = 1
            self.E[i, self.N] = 1
        for j in range(1, self.N+1):
            self.A[1, j] = 0
            self.A[self.N, j] = 0
            self.B[1, j] = 0
            self.B[self.N, j] = 0
            self.C[1, j] = 0
            self.C[self.N, j] = 0
            self.D[1, j] = 0
            self.D[self.N, j] = 0
            self.E[1, j] = 1
            self.E[self.N, j] = 1

    def condition_dirichlet(self, valeur):
        for i in range(1, self.N+1):
            self.S[i, 1] = valeur
            self.S[i, self.N] = valeur
            self.U[i, 1] = valeur
            self.U[i, self.N] = valeur
        for j in range(1, self.N+1):
            self.S[1, j] = valeur
            self.S[self.N, j] = valeur
            self.U[1, j] = valeur
            self.U[self.N, j] = valeur

    def source_horizontale(self, x, y, longueur, valeur):
        i0 = int(x*self.N)
        j0 = int(y*self.N)
        l0 = int(longueur*self.N)
        for i in range(i0, i0+l0):
            self.A[i, j0] = 0
            self.B[i, j0] = 0
            self.C[i, j0] = 0
            self.D[i, j0] = 0
            self.E[i, j0] = 1
            self.S[i, j0] = valeur
            self.U[i, j0] = valeur

    def gauss_seidel(self, w):
        """
        Itération de Gauss-Seidel avec sur-relaxation de paramètre w
        """
        for j in range(1, self.N+1):
            for i in range(1, self.N+1):
                self.U[i, j] = ((1-w)*self.U[i, j]
                                + w*(self.S[i, j]
                                     - self.A[i, j]*self.U[i, j-1]
                                     - self.C[i, j]*self.U[i, j+1]
                                     - self.D[i, j]*self.U[i-1, j]
                                     - self.B[i, j]*self.U[i+1, j]
                                     )/self.E[i, j]
                                )

    def calculer(self, iterations):
        omega = 2/(1 + np.sin(np.pi/(self.N-1)))
        for k in range(iterations):
            self.gauss_seidel(omega)


m = Maillage(65)
m.discretisation_laplacien()
m.condition_dirichlet(0.0)
m.source_horizontale(0.25, 0.4, 0.5, 1)
m.source_horizontale(0.25, 0.6, 0.5, -1)
print(m.S)
print(m.U)
m.calculer(100)
print(m.U)

plt.contour(m.U, 50)
plt.show()

# fig, ax = plt.subplots(1, 1)
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
#
# plt.plot([0.25, 0.75], [0.4, 0.4], color="red")
# plt.plot([0.25, 0.75], [0.6, 0.6], color="blue")
# plt.show()
