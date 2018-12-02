import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class HomeworkClasses:
    """Clase para la ejecución del metodo de regresion RANSAC"""
    def __init__(self, datasheet):
        self.datasheet = datasheet
        self.data = None
        self.x_inliers = []
        self.y_inliers = []
        self.x_outliers = []
        self.y_outliers = []
        self.data_test = None
        self.read_data()

    def read_data(self):
        """Leer datos de un csv"""
        self.data = pd.read_csv(self.datasheet, sep='\t', header=None).T
        self.data.columns = ['x_data', 'y_data']  # Definir etiquetas de las columnas

    @staticmethod
    def model(data, nro_puntos):
        """Hallar modelo lineal para dos puntos dados"""
        data_sample = data.sample(nro_puntos)  # Puntos aleatorios para reconstruir el modelo
        indices = data_sample.index.tolist()  # Indices de elementos aleatorios
        matriz_svd = [data_sample.loc[indices, 'x_data'].tolist(),
                      data_sample.loc[indices, 'y_data'].tolist(),
                      [1]*len(data_sample.loc[indices, 'x_data'].tolist())]
        matriz_svd = np.array(matriz_svd).T  # Matriz que contiene la relación entre puntos aleatorios
        matriz_u, matriz_s, matriz_vh = np.linalg.svd(matriz_svd)  # Descomposicion de la matriz
        val_mod = [matriz_vh[0, -1], matriz_vh[1, -1], matriz_vh[2, -1]]  # Obtencion de los parámetros a, b y c
        slop = -val_mod[0]/val_mod[1]  # Cálculo de la pendiente del modelo
        interc = -val_mod[2]/val_mod[1]  # Cálculo de la intersección del modelo
        # Distancia de todos los puntos al modelo
        distances = abs(val_mod[0]*np.array(data['x_data'].tolist())+val_mod[1] *
                        np.array(data['y_data'].tolist())+val_mod[2])/math.pow(math.pow(val_mod[0], 2) +
                                                                               math.pow(val_mod[1], 2), 0.5)
        prom_distances = sum(distances)/len(distances)  # Promedio de distancias

        return slop, interc, indices, prom_distances, val_mod

    def ransac_iterations(self, num_iter=10, num_samples=40, nro_puntos=6, ratio=0, ransac_ratio=0.8):
        """Ejecutar iteraciones de RANSAC"""
        ideal_interc = 0   # Intercepto ideal
        ideal_slope = 0  # Pendiente Ideal
        for itera in range(num_iter):
            slope, interc, indices, prom_distances, val_mod = self.model(self.data, nro_puntos)
            self.x_inliers = []
            self.y_inliers = []
            self.x_outliers = []
            self.y_outliers = []
            num_inliers = 0  # Numero de inliers dentro del modelo
            for index, row in self.data.iterrows():  # Iterar cada uno de los datos
                x_point, y_point = row['x_data'], row['y_data']
                # Calcular distancia del punto actual al modelo
                own_dist = abs(val_mod[0]*x_point+val_mod[1]*y_point+val_mod[2])/math.pow(math.pow(val_mod[0], 2) +
                                                                                          math.pow(val_mod[1], 2), 0.5)
                if own_dist <= prom_distances:  # Si la distancia del actual es menor que el promedio
                    self.x_inliers.append(x_point)  # Almacenar inliers
                    self.y_inliers.append(y_point)
                    num_inliers += 1
                else:
                    self.x_outliers.append(x_point)
                    self.y_outliers.append(y_point)  # Almacenar outliers

            if num_inliers/num_samples > ratio:  # Hallar mejor modelo que el anterior
                ratio = num_inliers/num_samples
                ideal_interc = interc
                ideal_slope = slope

            if num_inliers > num_samples*ransac_ratio:
                print('Se encontro el modelo! Ieración: ', str(itera))
                break
        print(ideal_interc)
        print(ideal_slope)
        return ideal_slope, ideal_interc

    def plot_data(self, slope, interc):
        """Graficas de los datos, modelo y outliers"""
        plt.plot(self.data.x_data, self.data.y_data, marker='o', label='Datos Entrada: '+str(len(self.data.x_data)),
                 color='#00cc00', linestyle='None', alpha=0.4)
        plt.plot(self.data.x_data, slope*self.data.x_data+abs(interc), label='Modelo')
        plt.scatter(self.x_outliers, self.y_outliers,  color='#00cc00',
                    label='Outliers: ' + str(len(self.x_outliers)), edgecolors='r', linewidths=1.3, alpha=0.7)
        plt.grid(which='major', color='0.75', linestyle='dashed')
        plt.title('Modelo')
        plt.legend()
        plt.show()
