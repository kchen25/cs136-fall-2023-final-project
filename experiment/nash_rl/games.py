import numpy as np
import nashpy as nash

U_1 = np.array([[3, 0], [5, 1]])
U_2 = np.array([[3, 5], [0, 1]])
ipd_game = nash.Game(U_1, U_2)

U_3 = np.array([[0, 0], [2, -4]])
U_4 = np.array([[0, 2], [0, -4]])
chicken_game = nash.Game(U_3, U_4)