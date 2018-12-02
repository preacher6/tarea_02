#!/usr/bin/env python

__author__ = "Juan Grajales"
__license__ = "GPL"

__email__ = "juafelgrajales@utp.edu.co"

import os
from homework_classes import HomeworkClasses


def main():
    n_iter = 100  # Número de iteraciones a realizar
    n_samples = 41  # Porcentaje aceptado de inliers para detener el proceso
    nro_puntos = 6  # Numero de puntos para la obtencion del modelo
    file = os.path.join("data", "tarea2data.csv")  # Ruta del archivo
    homeclass = HomeworkClasses(file)  # Invocar la clase principal para el metodo de RANSAC
    slope, interc = homeclass.ransac_iterations(n_iter, n_samples, nro_puntos)  # Obtener modelo e inliers
    homeclass.plot_data(slope, interc)  # Graficar datos obtenidos


if __name__.endswith("__main__"):
    main()
