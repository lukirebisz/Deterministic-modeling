from _pytest.junitxml import record_xml_attribute
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import matplotlib.gridspec as gridspec


# Dictionaries of areas

areas_dict = {
  "leg_1": [(40,0), (40, 30), (50, 30), (50, 0)],
  "leg_2": [(60,10), (60, 30), (70, 30), (70, 10)],
  "leg_3": [(110, 10), (110, 30), (120, 30), (120, 10)],
  "leg_4": [(130, 0), (130, 30), (140, 30), (140, 0)],
  "body": [(40, 30), (40, 70), (140, 70), (140, 30)],
  "tail": [(40, 70),(40, 100), (50, 100), (50, 70)],
  "neck": [(140, 60), (140, 70), (150, 70), (150, 60)],
  "head": [(150, 50), (150, 80), (180, 80), (180, 50)]
}



#for key in areas_dict.keys():
 #   print(areas_dict[key])



class Cat:
    def __init__(self, resolution, areas_dict:dict):
        self.resolution = resolution
        self.areas_dict = areas_dict
    
    def get_zero_matrices(self):
        matrices = []
        for key in self.areas_dict.keys():
            x_0 = min(tup[0] for tup in areas_dict[key])
            x_1 = max(tup[0] for tup in areas_dict[key])
            y_0 = min(tup[1] for tup in areas_dict[key])
            y_1 = max(tup[1] for tup in areas_dict[key])
            matrices.append(np.zeros((x_1 - x_0, y_1 - y_0)))


            #print((x_1 - x_0, y_1 - y_0))
        return matrices



cat1 = Cat([100, 200], areas_dict)
matrices = cat1.get_zero_matrices()
#print(matrices)