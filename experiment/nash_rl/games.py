import numpy as np
import nashpy as nash

U_1 = np.array([[4, 1], [6, 2]])
U_2 = np.array([[4, 6], [1, 2]])
ipd_game = nash.Game(U_1, U_2)
