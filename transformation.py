'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
YOUR NAME HERE
CS 252 Data Analysis Visualization, Spring 2022
'''
import numpy as np
from numpy import sin, cos, radians
import matplotlib.pyplot as plt
import palettable
from sympy import rad
import analysis
import data


class Transformation(analysis.Analysis):

    def __init__(self, orig_dataset, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        orig_dataset: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables,
            `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`
        '''
        
        super().__init__(data=data)
        self.original_dataset = orig_dataset

    def project(self, headers):
        '''Project the original dataset onto the list of data variables specified by `headers`,
        i.e. select a subset of the variables from the original dataset.
        In other words, your goal is to populate the instance variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list matches the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables is optional.

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables). Determine and fill in 'valid' values for all the `Data` constructor
        keyword arguments (except you dont need `filepath` because it is not relevant here).
        '''

        # Get the projected data
        projected_data = self.original_dataset.select_data(headers)
        # Get a new header to column dictionary
        new_header2col = {headers[index]:index for index in range(len(headers))}

        self.data = data.Data(headers=headers, data=projected_data,
                            header2col=new_header2col,header_dict=new_header2col)


    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''
        
        output_array = np.hstack((self.data.get_all_data(), np.ones((self.data.get_num_samples(),1))))

        return output_array

    def translation_matrix(self, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The transformation matrix.

        '''

        # Initilize the output as the identity matrix first
        output = np.eye(self.data.get_num_dims() + 1)
        # Set the translations column as a copy of the magnitudes
        translations = [vals for vals in magnitudes]
        # Add a 1 to the translations column for the homogenous coord
        translations.append(1)
        # Create the transformation matrix
        output[:,-1] = translations

        return output

        

    def scale_matrix(self, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''

        # Initilize the output as the identity matrix first
        output = np.eye(self.data.get_num_dims() + 1)
        # Set the scales column as a copy of the magnitudes
        scales = [vals for vals in magnitudes]
        # Add a 1 to the scales column for the homogenous coord
        scales.append(1)

        output[np.arange(output.shape[0]),np.arange(output.shape[0])] = scales

        return output
        

    def translate(self, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''

        # Get the homogenous data array
        dataH = self.get_data_homogeneous()
        # Get the translation matrix
        T = self.translation_matrix(magnitudes)

        # Calculate the translated data
        output = (T @ dataH.T).T
        # Remove the homogenous column
        output = np.delete(output, -1, axis=1)

        # updating the data field with a new one
        self.data = data.Data(headers=self.data.get_headers(), data=output,
                            header2col=self.data.get_mappings(),header_dict=self.data.get_mappings())

        return output

    def scale(self, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        
        # Get the homogenous data
        dataH = self.get_data_homogeneous()
        # Get the scaling matrix
        S = self.scale_matrix(magnitudes)

        # Calculate the scaled data matrix
        output = (S @ dataH.T).T
        output = np.delete(output, -1, axis=1)

        # Updating the data field with the new data
        self.data = data.Data(headers=self.data.get_headers(), data=output,
                            header2col=self.data.get_mappings(),header_dict=self.data.get_mappings())

        return output

    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The projected dataset after it has been transformed by `C`
        '''

        # Get homogeneous data
        dataH = self.get_data_homogeneous()
        # Computation
        output = (C @ dataH.T).T
        # Remove homogeneous column
        output = np.delete(output, -1, axis=1)

        # Updating the data field with the new data
        self.data = data.Data(headers=self.data.get_headers(), data=output,
                            header2col=self.data.get_mappings(),header_dict=self.data.get_mappings())
        
        return output
        

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''

        
        # Get global min from data array
        global_min = np.amin(self.data.get_all_data())
        # Get global max from data array
        global_max = np.amax(self.data.get_all_data())
        # Calculating range
        data_range = global_max - global_min

        dataH = self.get_data_homogeneous()
        T = self.translation_matrix([-global_min for i in range(self.data.get_num_dims())])
        S = self.scale_matrix([(1/data_range) for i in range(self.data.get_num_dims())])
        # Normalize the data
        output = (S @ T @ dataH.T).T
        # Remove homogeneous column
        output = np.delete(output, -1, axis=1)

        # Updating the data field with the new data
        self.data = data.Data(headers=self.data.get_headers(), data=output,
                            header2col=self.data.get_mappings(),header_dict=self.data.get_mappings())

        return output
        

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''

        # Get mins and maxs from all variables
        mins,maxs = self.range(self.data.get_headers())

        # Create homogeneous data matrix 
        dataH = self.get_data_homogeneous()
        T = self.translation_matrix(-mins)
        S = self.scale_matrix(1/(maxs-mins))
        output = (S @ T @ dataH.T).T
        # Remove homogeneous column
        output = np.delete(output, -1, axis=1)

        # Updating the data field with the new data
        self.data = data.Data(headers=self.data.get_headers(), data=output,
                            header2col=self.data.get_mappings(),header_dict=self.data.get_mappings())
        
        return output

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''

        variable_axis = self.data.get_mappings()[header]

        if variable_axis == 0:
            rotation_matrix = np.asarray([[1, 0, 0, 0],
                                          [0, cos(radians(degrees)), -sin(radians(degrees)), 0],
                                          [0, sin(radians(degrees)), cos(radians(degrees)), 0],
                                          [0, 0, 0, 1]])
        elif variable_axis == 1:
            rotation_matrix = np.asarray([[cos(radians(degrees)), 0, sin(radians(degrees)), 0],
                                          [0, 1, 0, 0],
                                          [-sin(radians(degrees)), 0, cos(radians(degrees)), 0],
                                          [0, 0, 0, 1]])
        else:
            rotation_matrix = np.asarray([[cos(radians(degrees)), -sin(radians(degrees)), 0, 0],
                                          [sin(radians(degrees)), cos(radians(degrees)), 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]])

        return rotation_matrix

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!
        '''
        
        dataH = self.get_data_homogeneous()
        R = self.rotation_matrix_3d(header, degrees)
        output = (R @ dataH.T).T
        output = np.delete(output, -1, axis=1)

        # Updating the data field with the new data
        self.data = data.Data(headers=self.data.get_headers(), data=output,
                            header2col=self.data.get_mappings(),header_dict=self.data.get_mappings())

        return output

    def scatter3d(self, xlim, ylim, zlim, better_view=False):
        '''Creates a 3D scatter plot to visualize data the x, y, and z axes are drawn, but not ticks

        Axis labels are placed next to the POSITIVE direction of each axis.

        Parameters:
        -----------
        xlim: List or tuple indicating the x axis limits. Format: (low, high)
        ylim: List or tuple indicating the y axis limits. Format: (low, high)
        zlim: List or tuple indicating the z axis limits. Format: (low, high)
        better_view: boolean. Change the view so that the Z axis is coming "out"
        '''
        if len(self.data.get_headers()) != 3:
            print("need 3 headers to make a 3d scatter plot")
            return

        headers = self.data.get_headers()
        xyz = self.data.get_all_data()

        if better_view:
            # by default, matplot lib puts the 3rd axis heading up
            # and the second axis heading back.
            # rotate it so that the second axis is up and the third is forward
            R = np.eye(3)
            R[1, 1] = np.cos(np.pi/2)
            R[1, 2] = -np.sin(np.pi/2)
            R[2, 1] = np.sin(np.pi/2)
            R[2, 2] = np.cos(np.pi/2)
            xyz = (R @ xyz.T).T

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # Scatter plot of data in 3D
        ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        ax.plot(xlim, [0, 0], [0, 0], 'k')
        ax.plot([0, 0], ylim, [0, 0], 'k')
        ax.plot([0, 0], [0, 0], zlim, 'k')
        ax.text(xlim[1], 0, 0, headers[0])

        if better_view:
            ax.text( 0, zlim[0], 0, headers[2])
            ax.text( 0, 0, ylim[1], headers[1])
        else:
            ax.text(0, ylim[1], 0, headers[1])
            ax.text(0, 0, zlim[1], headers[2])

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        plt.show()

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''

        # Getting plot data
        plot_data = self.original_dataset.select_data([ind_var, dep_var, c_var])

        # Separating to X,Y,Z axis
        X = plot_data[:,0]
        Y = plot_data[:,1]
        Z = plot_data[:,2]

        # Setting color to grayscale
        color_map = palettable.colorbrewer.sequential.Greys_9

        fig = plt.figure(figsize=(5,5))

        a = plt.scatter(X, Y, c=Z, s=75, cmap=color_map.mpl_colormap, edgecolor='black')
        cbar = plt.colorbar(a)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.title(title)
        cbar.set_label(c_var)
        
        

    def heatmap(self, headers=None, title=None, cmap="gray"):
        '''Generates a heatmap of the specified variables (defaults to all). Each variable is normalized
        separately and represented as its own row. Each individual is represented as its own column.
        Normalizing each variable separately means that one color axis can be used to represent all
        variables, 0.0 to 1.0.

        Parameters:
        -----------
        headers: Python list of str (or None). (Optional) The variables to include in the heatmap.
            Defaults to all variables if no list provided.
        title: str. (Optional) The figure title. Defaults to an empty string (no title will be displayed).
        cmap: str. The colormap string to apply to the heatmap. Defaults to grayscale
            -- black (0.0) to white (1.0)

        Returns:
        -----------
        fig, ax: references to the figure and axes on which the heatmap has been plotted
        '''

        # Create a doppelganger of this Transformation object so that self.data
        # remains unmodified when heatmap is done
        data_clone = data.Data(headers=self.data.get_headers(),
                               data=self.data.get_all_data(),
                               header2col=self.data.get_mappings())
        dopp = Transformation(self.data, data_clone)
        dopp.normalize_separately()

        fig, ax = plt.subplots()
        if title is not None:
            ax.set_title(title)
        ax.set(xlabel="Individuals")

        # Select features to plot
        if headers is None:
            headers = dopp.data.headers
        m = dopp.data.select_data(headers)

        # Generate heatmap
        hmap = ax.imshow(m.T, aspect="auto", cmap=cmap, interpolation='None')

        # Label the features (rows) along the Y axis
        y_lbl_coords = np.arange(m.shape[1]+1) - 0.5
        ax.set_yticks(y_lbl_coords, minor=True)
        y_lbls = [""] + headers
        ax.set_yticklabels(y_lbls)
        ax.grid(linestyle='none')

        # Create and label the colorbar
        cbar = fig.colorbar(hmap)
        cbar.ax.set_ylabel("Normalized Features")

        return fig, ax
