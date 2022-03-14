'''data.py
Reads CSV files, stores data, access/filter data by variable name
Henry Pacheco Cachon
CS 251 Data Analysis and Visualization
Spring 2022
'''

import numpy as np

class Data:
    def __init__(self, filepath=None, headers=None, data=None, header2col=None, header_dict=None):

        '''
        Data object constructor

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file
        headers: Python list of strings or None. List of strings that explain the name of each
            column of data.
        data: ndarray or None. shape=(N, M).
            N is the number of data samples (rows) in the dataset and M is the number of variables
            (cols) in the dataset.
            2D numpy array of the datasets values, all formatted as floats.
        header2col: Python dictionary or None.
                Maps header (var str name) to column index (int).
                Example: "sepal_length" -> 0
        header_dict: Python dictionary or None.
                Maps the header to column index in raw data set
        '''

        # Setting headers field
        self.headers = headers
        # Setting data field
        self.data = data
        # Setting header2col field
        self.header2col = header2col
        # Setting header_dictionary field 
        self.header_dictionary = header_dict
        # Saving filepath
        self.filepath = filepath
        # Setting number of columns field
        self.num_cols = 0
        # Setting number of rows field
        self.num_rows = 0

        # If the filename is not none...
        if (filepath):

            # Set the filename field
            self.filepath = filepath
            # Call on the read method
            self.read(filepath)
        else:
            self.num_cols = self.data.shape[1]
            self.num_rows = self.data.shape[0]


    def parse_headers(self, lines):

        '''
        Parses the first two lines of the csv

        Parameter:
        ----------
        line: String. Line read from a csv file

        Returns:
        None.
        '''

        # Get the first line and split it
        first_line = lines[0].replace('\n','').replace(' ','').split(',')

        # Get the second line and split it 
        second_line = lines[1].replace('\n','').replace(' ', '').split(',')

        if "numeric" in second_line:

            # Setting up a dictionary index:datatype
            type_dictionary = {index:second_line[index] for index in range(len(second_line))}
            # Removing any datatype that isn't a numeric type from the type_dictionary
            type_dictionary = {index:second_line[index] for index in type_dictionary if second_line[index]=='numeric'}

            # Setting up the header dictionary
            self.header_dictionary = {first_line[index]:index for index in type_dictionary}
            # Setting up the header list
            self.headers = [key for key in self.header_dictionary]
            # Setting up the header2col dictionary
            self.header2col = {self.headers[index]:index for index in range(len(self.headers))}
        else:

            print("No numeric columns detected")
            pass
        

    def parse_data(self, fh):
        
        '''
        Parses the data part of the csv

        Parameters:
        -----------
        fh: An open object type

        Output:
        -----------
        None.
        '''

        # Reading the first line containing data
        line = fh.readline()

        # Data list
        currentData = []

        # Looping through all the lines
        while(line):

            # Ignoring new line characters
            if line != "\n":
                
                # Split line by commas
                elements = line.replace('\n','').replace(' ', '').split(',')
                # Getting all the numerical elements
                numeric_elements = [elements[i] for i in list(self.header_dictionary.values())]
            
                # Appending row into currentData list
                currentData.append(numeric_elements)

            # Reading the next line
            line = fh.readline()

        # Saving data as a float array and putting it in the data field
        self.data = np.asarray(currentData, dtype=np.float)
        self.num_cols = self.data.shape[1]
        self.num_rows = self.data.shape[0]
        

    def read(self, filepath):

        '''
        Read in the .csv file `filepath` in 2D tabular format. Convert to numpy ndarray called
        `self.data` at the end (think of this as 2D array or table).

        Format of `self.data`:
            Rows should correspond to i-th data sample.
            Cols should correspond to j-th variable / feature.

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file

        Returns:
        -----------
        None. (No return value).
        '''
        
        # Reading file in filepath
        with open(filepath, 'r') as fh:

            # Reading the first line containing headers
            first_line = fh.readline()
            # Reading the second line containing data types
            second_line = fh.readline()
            
            # Only parsing headers if they were not set already
            if self.headers == None:
                # Parsing the first two lines
                self.parse_headers([first_line,second_line])

            # Parsing data lines
            self.parse_data(fh)
            

    def __str__(self):
        '''toString method

        (For those who don't know, __str__ works like toString in Java...In this case, it's what's
        called to determine what gets shown when a `Data` object is printed.)

        Returns:
        -----------
        str. A nicely formatted string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.
        '''
        
        out_string = ""

        # Put headers in the string
        for i in range(len(self.headers)):
            
            if i == len(self.headers) - 1:
                out_string += self.headers[i] + "\n"
            else:
                out_string += self.headers[i] + "      "

        # Put data in string
        for i,j in np.ndindex(self.data.shape):

            if i != 0 and i % 5 == 0:
                break

            if j != 0 and j % (self.num_cols - 1) == 0:
                out_string += str(self.data[i][j]) + "\n"
            else:
                out_string += str(self.data[i][j]) + "      "
        
        return out_string
            
            
        

    def get_headers(self):
        '''Get method for headers

        Returns:
        -----------
        Python list of str.
        '''
        
        return self.headers

    def get_mappings(self):
        '''Get method for mapping between variable name and column index

        Returns:
        -----------
        Python dictionary. str -> int
        '''
        return self.header2col

    def get_num_dims(self):
        '''Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        '''
        
        return self.data.shape[1]

    def get_num_samples(self):
        '''Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        '''

        return self.data.shape[0]

    def get_sample(self, rowInd):
        '''Gets the data sample at index `rowInd` (the `rowInd`-th sample)

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        '''

        return self.data[rowInd]

    def get_header_indices(self, headers):
        '''Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers`
            list.
        '''

        return [self.header2col[header] for header in headers]

    def get_all_data(self):
        '''Gets a copy of the entire dataset

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_data_samps, num_vars). A copy of the entire dataset.
            NOTE: This should be a COPY, not the data stored here itself.
            This can be accomplished with numpy's copy function.
        '''
        return np.copy(self.data)

    def head(self):
        '''Return the 1st five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). 1st five data samples.
        '''
        return self.data[:5:]

    def tail(self):
        '''Return the last five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). Last five data samples.
        '''
        return self.data[-5::]

    def limit_samples(self, start_row, end_row):
        '''Update the data so that this `Data` object only stores samples in the contiguous range:
            `start_row` (inclusive), end_row (exclusive)
        Samples outside the specified range are no longer stored.

        (Week 2)

        '''
        self.data = self.data[start_row : end_row :]

    def select_data(self, headers, rows=[]):
        '''Return data samples corresponding to the variable names in `headers`.
        If `rows` is empty, return all samples, otherwise return samples at the indices specified
        by the `rows` list.

        (Week 2)

        For example, if self.headers = ['a', 'b', 'c'] and we pass in header = 'b', we return
        column #2 of self.data. If rows is not [] (say =[0, 2, 5]), then we do the same thing,
        but only return rows 0, 2, and 5 of column #2.

        Parameters:
        -----------
            headers: Python list of str. Header names to take from self.data
            rows: Python list of int. Indices of subset of data samples to select.
                Empty list [] means take all rows

        Returns:
        -----------
        ndarray. shape=(num_data_samps, len(headers)) if rows=[]
                 shape=(len(rows), len(headers)) otherwise
            Subset of data from the variables `headers` that have row indices `rows`.

        Hint: For selecting a subset of rows from the data ndarray, check out np.ix_
        '''

        # Getting a list of the columns associated with the input headers
        columns = [self.header2col[header] for header in headers]

        if rows != []:

            # Getting the indices of the subarray from the inputs
            subarray_index = np.ix_(rows, columns)

            return self.data[subarray_index]

            
        else:

            # Creating a list of all the row indices in self.data
            all_rows = [i for i in range(self.data.shape[0])]
            # Getting indices of the subarray
            subarray_index = np.ix_(all_rows, columns)

            return self.data[subarray_index]

