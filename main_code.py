import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import matplotlib.gridspec as gridspec



class Cat:
    def __init__(self, resolution, areas_dict:dict, parameters:dict):
        self.resolution = resolution
        self.areas_dict = areas_dict
        self.parameters = parameters

        self.matrices = {}
        self.total_matrices = []

        self.variance = 0
        
        self.GM_Neumann_2D()
        self.create_total_matrices()


    def create_total_matrices(self):
        x_max = self.resolution[0]
        y_max = self.resolution[1]
        vec_t = np.arange(0, self.parameters["T_max"], self.parameters["h_t"])

        total_matrix = np.zeros((len(vec_t), y_max, x_max))
        for key in self.areas_dict.keys():
                x_0 = min(tup[0] for tup in self.areas_dict[key])
                x_1 = max(tup[0] for tup in self.areas_dict[key])
                y_0 = min(tup[1] for tup in self.areas_dict[key])
                y_1 = max(tup[1] for tup in self.areas_dict[key])
                total_matrix[:, y_0:y_1, x_0:x_1] = self.matrices[key][:, :, :]

        self.total_matrices = total_matrix
        

    def show_i_th_step(self, i: int):
        x_max = self.resolution[0]
        y_max = self.resolution[1]

        total_matrix = np.zeros((y_max, x_max))
        for key in self.areas_dict.keys():
                x_0 = min(tup[0] for tup in self.areas_dict[key])
                x_1 = max(tup[0] for tup in self.areas_dict[key])
                y_0 = min(tup[1] for tup in self.areas_dict[key])
                y_1 = max(tup[1] for tup in self.areas_dict[key])
                total_matrix[(y_max-y_1):(y_max-y_0), (x_0):(x_1)] = self.matrices[key][i, :, :]

        plt.imshow(total_matrix, cmap = "YlOrBr")


    def get_variance(self):
         sum_variance = 0
         sum_areas = 0

         for key in self.areas_dict.keys():
            x_0 = min(tup[0] for tup in self.areas_dict[key])
            x_1 = max(tup[0] for tup in self.areas_dict[key])
            y_0 = min(tup[1] for tup in self.areas_dict[key])
            y_1 = max(tup[1] for tup in self.areas_dict[key])
            area = (x_1 - x_0)*(y_1 - y_0)
            
            solution = self.matrices[key][-1, :, :]
            vector = np.reshape(solution, -1)
            variance = np.var(vector)

            sum_areas = sum_areas + area
            sum_variance = sum_variance + variance*area

            self.variance = sum_variance/sum_areas

         return  self.variance 


    def GM_Neumann_2D(self):
        h_x = self.parameters["h_x"]
        h_y = self.parameters["h_y"]
        h_t = self.parameters["h_t"]
        rho = self.parameters["rho"]
        mu_a = self.parameters["mu_a"]
        mu_h = self.parameters["mu_h"]
        D_a = self.parameters["D_a"]
        D_h = self.parameters[ "D_h"]
        rho_a = self.parameters["rho_a"]
        rho_h = self.parameters["rho_h"]
        T_max = self.parameters["T_max"]
        a_0_func = lambda x, y: np.abs(np.random.random(x.shape))
        h_0_func = lambda x, y: np.abs(np.random.random(x.shape))
        g = lambda t: 0
        
        for key in self.areas_dict.keys():
            x_0 = min(tup[0] for tup in self.areas_dict[key])
            x_1 = max(tup[0] for tup in self.areas_dict[key])
            y_0 = min(tup[1] for tup in self.areas_dict[key])
            y_1 = max(tup[1] for tup in self.areas_dict[key])


            x = np.arange(x_0, x_1, h_x)
            y = np.arange(y_0, y_1, h_y)

            vec_t = np.arange(0, T_max, h_t)
            vec_x = np.meshgrid(x, y)[0]
            vec_y = np.meshgrid(x, y)[1]

            result_matrix_a_0 = a_0_func(vec_x, vec_y)
            result_matrix_a = np.zeros((len(vec_t), vec_y.shape[0], vec_x.shape[1]))
            result_matrix_a[0, 1:-1, 1:-1] = result_matrix_a_0[1:-1, 1:-1]
            result_matrix_a[0, 0, :] = result_matrix_a[0, 1, :] + h_x*g(0)
            result_matrix_a[0, -1, :] = result_matrix_a[0, -2, :] + h_x*g(0)
            result_matrix_a[0, :, 0] =  result_matrix_a[0, :, 1] + h_y*g(0)
            result_matrix_a[0, :, -1] = result_matrix_a[0, :, -2] + h_y*g(0)

            result_matrix_h_0 = h_0_func(vec_x, vec_y)
            result_matrix_h = np.zeros((len(vec_t), vec_y.shape[0], vec_x.shape[1]))
            result_matrix_h[0, 1:-1, 1:-1] = result_matrix_h_0[1:-1, 1:-1]
            result_matrix_h[0, 0, :] = result_matrix_h[0, 1, :] + h_x*g(0)
            result_matrix_h[0, -1, :] = result_matrix_h[0, -2, :] + h_x*g(0)
            result_matrix_h[0, :, 0] =  result_matrix_h[0, :, 1] + h_y*g(0)
            result_matrix_h[0, :, -1] = result_matrix_h[0, :, -2] + h_y*g(0)


            for i in range(len(vec_t) - 1):
                t = vec_t[i]
                result_matrix_a[i+1, 0, :] = result_matrix_a[i+1, 1, :] + h_x*g(t)
                result_matrix_a[i+1, -1, :] = result_matrix_a[i+1, -2, :] + h_x*g(t)
                result_matrix_a[i+1, :, 0] =  result_matrix_a[i+1, :, 1] + h_y*g(t)
                result_matrix_a[i+1, :, -1] = result_matrix_a[i+1, :, -2] + h_y*g(t)

                result_matrix_a[i+1, 1:-1, 1:-1] = result_matrix_a[i, 1:-1, 1:-1] + h_t*(D_a*((result_matrix_a[i, 2:, 1:-1] 
                                + result_matrix_a[i, :-2, 1:-1] - 2*result_matrix_a[i, 1:-1, 1:-1])/h_x +
                                + (result_matrix_a[i, 1:-1, 2:] + result_matrix_a[i, 1:-1, :-2] - 
                                   2*result_matrix_a[i, 1:-1, 1:-1])/h_y)
                                + rho*(result_matrix_a[i, 1:-1, 1:-1]**2)/result_matrix_h[i, 1:-1, 1:-1] -  
                                mu_a*result_matrix_a[i, 1:-1, 1:-1] +  rho_a)

                result_matrix_h[i+1, 0, :] = result_matrix_h[i+1, 1, :] + h_x*g(t)
                result_matrix_h[i+1, -1, :] = result_matrix_h[i+1, -2, :] + h_x*g(t)
                result_matrix_h[i+1, :, 0] =  result_matrix_h[i+1, :, 1] + h_y*g(t)
                result_matrix_h[i+1, :, -1] = result_matrix_h[i+1, :, -2] + h_y*g(t)

                result_matrix_h[i+1, 1:-1, 1:-1] = result_matrix_h[i, 1:-1, 1:-1] + h_t*(D_h*((result_matrix_h[i, 2:, 1:-1] 
                                + result_matrix_h[i, :-2, 1:-1] - 2*result_matrix_h[i, 1:-1, 1:-1])/h_x +
                                + (result_matrix_h[i, 1:-1, 2:] + result_matrix_h[i, 1:-1, :-2] - 
                                   2*result_matrix_h[i, 1:-1, 1:-1])/h_y)
                                + rho*(result_matrix_a[i, 1:-1, 1:-1]**2) - mu_h*result_matrix_h[i, 1:-1, 1:-1] + rho_h)

            self.matrices[key] = result_matrix_a[:, :, :]



    
    def animation(self):
        x_max = self.resolution[0]
        y_max = self.resolution[1]
        x = np.arange(0, x_max, self.parameters["h_x"])
        y = np.arange(0, y_max, self.parameters["h_y"])
        
        total_matrices = self.total_matrices

        solution_a_cp = total_matrices[0::100, :, :]
        solution_a_cp = solution_a_cp.transpose()
        

        fig = plt.figure(figsize=(12, 5), facecolor='white')
        fig.suptitle("Gierer-Meinhardt model")

        gs = gridspec.GridSpec(1, 1)

        ax1 = plt.subplot(gs[0, 0])

        ax1.set_xlim((np.min(x), np.max(x)))
        ax1.set_ylim((np.min(y), np.max(y)))
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$y$")
        ax1.set_title("$a(x,y,t)$")


        pcolormesh_d = ax1.pcolormesh(x, y, solution_a_cp[:,:,0].T, shading='gouraud', cmap = "YlOrBr")

        cb1 = fig.colorbar(pcolormesh_d, ax=ax1)

        def animation(j):
            pcolormesh_d.set_array((solution_a_cp[:, :, j].T).ravel())
            return pcolormesh_d

        anim = anm.FuncAnimation(fig,
                                func = animation,
                                frames = solution_a_cp.shape[2],
                                interval = 100,
                                blit = False
                                )
        plt.close()
        plt.rc('animation', html='jshtml')
        return anim
            


areas_dict = {
  "leg_1": [(40,0), (40, 30), (50, 30), (50, 0)],
  "leg_2": [(60,0), (60, 30), (70, 30), (70, 0)],
  "leg_3": [(110, 0), (110, 30), (120, 30), (120, 0)],
  "leg_4": [(130, 0), (130, 30), (140, 30), (140, 0)],
  "body": [(40, 30), (40, 70), (140, 70), (140, 30)],
  "tail": [(40, 70),(40, 100), (50, 100), (50, 70)],
  "head": [(140, 50), (140, 80), (170, 80), (170, 50)]
}



parameters = {
  "h_x": 1,
  "h_y": 1,
  "h_t": 0.0001,
  "rho": 1,
  "mu_a": 1,
  "mu_h": 1,
  "D_a": 1.5,
  "D_h": 0.5,
  "rho_a": 1,
  "rho_h": 0.8,
  "T_max": 0.75
}


cat1 = Cat([200, 100], areas_dict, parameters)


cat1.show_i_th_step(-1)