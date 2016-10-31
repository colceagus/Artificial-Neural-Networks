import numpy as np
# Vertical Stack (row wise) 
# Adds dimension row wise 
# If we have the data in the columns, it adds another example to our data
# If we consider columns data being on the horizontal, and rows stacked 
# on the vertical, we add data (examples) to the rows, add more rows etc.
mat = np.ones((10000, 724))
print("Vertical Stack (row wise): Adds a Dimension row wise. If we consider \n" +
	  "columns data being on the horizontal, and rows stacked on the vertical, \n" +
	  "we add data (examples) to the rows, add more rows etc.")
print mat.shape,
ones = np.zeros((1, 724)) 
mat = np.vstack((mat, ones))
print " -> (%d, %d)" % (mat.shape) # (10001, 724)

# Horizontal Stack (column wise)
# Adds dimension column wise
# If we have the data in the columns, it adds info to our data
# Having 724 feature points stored in column form, it adds one more 
# feature point, resulting 725 feature points
mat = np.ones((10000, 724))
print("Horizontal Stack (column wise): Adds dimension column wise. If we have \n" +
	  "the data in the columns, it adds info to our data. Having 724 feature \n" +
	  "points stored in column form, it adds one more feature point, resulting " + 
	  "in 725 feature points")
print mat.shape,
# we add a one (bias) to all our examples
ones = np.zeros((10000, 1)) 
mat = np.hstack((mat, ones))
print " -> (%d, %d)" % (mat.shape) # (10000, 725)
