import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import matplotlib.gridspec as gridspec



class Cat:
    """A class used to represent a Cat, patterns of its skin which are calculated by the Gierer-Meinhardt model.
    """
    

    def __init__(self, resolution:list, areas_dict:dict, parameters:dict):
        """ The init function which sets values of variables:
        
        Args:
            resolution (list): the resolution of full image,
            areas_dict (dict): dictionary with names and coordinates of body parts,
            parameters (dict): parameters of Gierer-Meinhardt model.


        Moreover, the init function automatically calculates the solution of Gierer-Meinhardt model
        and saves it as the variables matrices and total_matrices.
        """
        self.resolution = resolution
        self.areas_dict = areas_dict
        self.parameters = parameters

        self.matrices = {}
        self.total_matrices = []

        self.variance = 0
        
        self.GM_Neumann_2D()
        self.create_total_matrices()


    def create_total_matrices(self):
        """ Function creates the total matrix from matrices of solutions for each body parts
        and saves it as the variable total_matrices."""
        
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
        """ Function creates the plot of i-th step of the solution of Gierer-Meinhardt model.

        Args:
            i (int): the index of solution
        """
        x_max = self.resolution[0]
        y_max = self.resolution[1]

        total_matrix = np.zeros((y_max, x_max))
        for key in self.areas_dict.keys():
                x_0 = min(tup[0] for tup in self.areas_dict[key])
                x_1 = max(tup[0] for tup in self.areas_dict[key])
                y_0 = min(tup[1] for tup in self.areas_dict[key])
                y_1 = max(tup[1] for tup in self.areas_dict[key])
                total_matrix[(y_max-y_1):(y_max-y_0), (x_0):(x_1)] = self.matrices[key][i, ::-1, :]

        plt.imshow(total_matrix, cmap = "YlOrBr")

    def get_result_matrix(self):
        """Function returns the last step of the solution of Gierer-Meinhardt model
        in the plot coordinates (point (0,0) in the top left corner).

        Returns:
            list: Matrix of last step of the solution of Gierer-Meinhardt model.
        """
        x_max = self.resolution[0]
        y_max = self.resolution[1]

        total_matrix = np.zeros((y_max, x_max))
        for key in self.areas_dict.keys():
                x_0 = min(tup[0] for tup in self.areas_dict[key])
                x_1 = max(tup[0] for tup in self.areas_dict[key])
                y_0 = min(tup[1] for tup in self.areas_dict[key])
                y_1 = max(tup[1] for tup in self.areas_dict[key])
                total_matrix[(y_max-y_1):(y_max-y_0), (x_0):(x_1)] = self.matrices[key][-1, ::-1, :]

        return total_matrix
    
    def get_all_result_matrices(self):
        """Function returns all steps of the solution of Gierer-Meinhardt model.

        Returns:
            list: Matrix of solutions.
        """
        return self.total_matrices


    def get_variance(self):
         """Function calculates the weighted arithmetic mean of variances with wieghts 
         equal area of the appropriate body part (result matrix)

         Returns:
             float: weighted mean variance
         """
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
            """Function calculates the solution of Gierer-Meinhardt model using 
            finite difference numerical scheme with homogeneous Neumann conditions
            for each of the rectangle body parts separately.
            """
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
            c = self.parameters["c"]
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
                                    + rho*(result_matrix_a[i, 1:-1, 1:-1]**2)/(result_matrix_h[i, 1:-1, 1:-1] *(1 + c*result_matrix_a[i, 1:-1, 1:-1]**2)) -  
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
        """Function creates the animation of the solutions of Gierer-Meinhardt model
        in time.

        Returns:
            FuncAnimation: animation.
        """
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
        ax1.set_title("$a(x,y,t)$: intensity of the pigment")


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
            
            

