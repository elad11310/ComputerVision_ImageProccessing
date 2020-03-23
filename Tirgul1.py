import numpy as np
## numpy array is a matrix object. openCV uses numpy arrays to store images.
## numpy very fast and suggests different tools to deal with matrix(multipie,divide...)
import cv2
## the package for the openCV in python

import matplotlib.pyplot as plt
## another pacage for openCV


## some actions on np arrays.
a = np.array([1, 2, 3, 4])
print(a[:2])  ##[1,2] slice -->:<-- if empty from start to end.
print(a[-1])  ## [4] prints from last
print(a[-2])  ## [3]

print('-------')

A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [11, 22, 33, 44]])
B = np.array([[1, 2, 3], [5, 6, 7], [11, 22, 33], [8, 6, 3]])
print(A[1, 2])  ## 7
print(A[0, 1:3])  ## 2,3 slice from index 1 to 3 not included
print(A.shape)  ## 3,4 because its matrix with 3 rows and 4 cols
print(A[:3])  ## all the matrix should be printed.

print('-------')

print(A[0 :])  ## prints first row [1,2,3,4]
print('-------')
print(A[0::2])  ## prints from first row and skip 2 rows [1,2,3,4],[11,22,33,44]
print('-------')

##index accessing:
ind = (1, -1)  ## tuple
print(A[ind, :])  ## will print the second row and the last row

print(np.where(A < 6))  ## y Coordinate :[0,0,0,0,1], X coordinate: [0,1,2,3,0] prints all the indexes(x,y) where A[x,y]<6
print('-------')
##A[A<6]=99 ## all the coordinates where A[x,y]<6 will replaced in 99
print(A)
print('-------')
print(A.reshape(2, 6))  ## changing the shape of the matrix from 3,4 to 2,6 - returns a new mat
print('-------')
print(np.sum(A))  ## 626 , prints the sum of all the variables in the mat.
print('-------')
print(A.sum(axis=0))  ## summing only the cols
print('-------')
print(A.sum(axis=1))  ## summing only the rows.
print('-------')
print(A + A)  ## summing mat A itsself
print('-------')
print(A.dot(B))  ## multiple A By B

## untill now it was an introduction to numpy and some functions
## now we'll see some things about openCV

img = cv2.imread('sample_image.jpg')  ## reading the image
##cv2.imshow('Corona_Beer',img) ## representing the image in GUI (title, image)
##cv2.waitKey(0)
##cv2.imsave('sample_image.jpg',img) ## saving the img
cv2.imshow('sample_image.jpg',img[:,:,0]) ## in cv2 the colors are B,G,R instead of R,G,B , there is an option to convert.
cv2.waitKey(0)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) ## for converting from B,G,R to R,G,B


## another way , even better way - Matplotlib
plt.imshow(img)
plt.show()

## some plot functions
plt.plot(155,22,'*') # for a point
plt.show()
x = np.random.randint(0,500,(10,2))
plt.plot(x[:,0],x[:,1],'*') ## for alot of points
plt.show()
plt.plot(x[:,0],x[:,1],'-*') # use ‘-’ for lines

plt.matshow(A) ## disply the image in a matrix.
plt.colorbar()
plt.show()

