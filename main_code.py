import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import matplotlib.gridspec as gridspec


# Dictionaries of areas

areas_dict = {
  "leg_1": [(40,0), (40, 30), (50, 30), (50, 0)],
  "leg_2": [(60,10), (60, 30), (70, 30), (70, 10)]
}



result_matrix = np.zeros((100, 200))

for key in areas_dict.keys():
    print(areas_dict[key])