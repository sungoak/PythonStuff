import scipy as sp	# import scipy functions as class sp

a = sp.array([1,2,3])		# create 1-d array
b = sp.array([[1,2,3],[4,5,6]])	# create 2-d array
c = sp.arange(4)			# creates 1-d array, 0,1,2,3
d = sp.linspace(0,9,5)	# creates array from 0,9 with 5 elements

a1 = sp.r_[0:10]	# short-hand linspace() for 0:10, array = 					0,1,2,3,4,5,6,7,8,9

a2 = sp.ones([4,3])	# matrix of ones, 4 rows by 3 columns
a3 = sp.identity(4)	# creates 4x4 identity matrix of ones

# use of mgrid and ogrid to create multidimensional grids or arrays/matrices
x,y = sp.mgrid[0:3,4:7]	# creates x = 0,1,2 repeated row-wise 3x, same for y
x1,y1 = sp.ogrid[0:3,4:7] 	# creates x = 0,1,2, y = 4,5,6 (linear arrays 						only)

# Creating Matrices
A = sp.mat('1,2;3,4')	# 1,2 is row vector separated by ; column vector sep
print(A)	# output matrix A
2*A	# scalar multiplication of matrix A
A.T	# matrix A transpose
A*A	# matrix multiplication
A**2	# Matrix 2 to the power of 2 (same as A*A)
B = 5*sp.diag([1.,3,5])
sp.mat(B)	# converts B to matrix type
sp.mat(B).I	# computes matrix inverse of sp.mat(B)

# isnan, isfinite, isinf are newly added functions to NumPy
C = B	# copy matrix B into C
C[0,1] = sp.nan	# insert NaN value into 1st row, 2nd column element
sp.isnan(C) # yields all 'False' elements except one element as True

# looking at complex numbers
a4 = sp.array([1+1j,2,5j])	# create complex array 
sp.iscomplexobj(a4)		# determine whether a4 is complex (TRUE)
sp.isreal(a4)	# determines element by element which are REAL or COMPLEX
sp.iscomplex(a4)	# determination of COMPLEX elements
type(a4)	# outputs type and functional dependencies

# concatenating matrices and arrays:
a5 = sp.array([1,2,3])
a6 = sp.array([4,5,6])
sp.vstack((a5,a6))	# vertical array concatenation
sp.hstack((a5,a6))	# horizontal array concat
dstack((a5,a6))		# vertical array concat, transposed

# view all variables that have been created thus far:
sp.who()	# python command, similar to MATLAB

# importing and using matplotlib (plotting library from MATLAB)
t = sp.linspace(0,100)	# create time array 't'
y = sp.sin(2*sp.pi*t)	# create sinusoidal series y
import pylab as pl	# import pylab (MATPLOTLIB) as 'pl' class


