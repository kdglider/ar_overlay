import cv2
import numpy as np
import cv2

#im = cv2.imread('lena.jpg')
im = cv2.imread("cola.jpg")
#im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
background = cv2.imread('stadium.jpg')
#background = cv2.cvtColor(background,cv2.COLOR_BGR2RGB)

row,col,ch=im.shape
print(row)
print(col)
print(ch)


rows_p,cols_p,ch_p = background.shape
print(rows_p)
print(cols_p)
print(ch_p)

pts1 = np.float32([[0,0],[974,0],[0,974],[974,974]]) # cola coords 
pts2 = np.float32([[560,83],[1140, 516],[95,527],[98,733]]) # stadium tile coords

M = cv2.getPerspectiveTransform(pts1,pts2)
print(M)
bunchX=[]; bunchY=[]

tt = np.array([[1],[1],[1]])
tmp = np.dot(M,tt)
bunchX.append(tmp[0]/tmp[2])
bunchY.append(tmp[1]/tmp[2])
   
tt = np.array([[im.shape[1]],[1],[1]])
tmp = np.dot(M,tt)
bunchX.append(tmp[0]/tmp[2])
bunchY.append(tmp[1]/tmp[2])
   
tt = np.array([[1],[im.shape[0]],[1]])
tmp = np.dot(M,tt)
bunchX.append(tmp[0]/tmp[2])
bunchY.append(tmp[1]/tmp[2])
   
tt = np.array([[im.shape[1]],[im.shape[0]],[1]])
tmp = np.dot(M,tt)
bunchX.append(tmp[0]/tmp[2])
bunchY.append(tmp[1]/tmp[2])
   
refX1 = int(np.min(bunchX))
refX2 = int(np.max(bunchX))
refY1 = int(np.min(bunchY))
refY2 = int(np.max(bunchY))
   
# Final image whose size is defined by the offsets previously calculated
final = np.zeros((int(refY2-refY1),int(refX2-refX1),3))

Store_Matrix=np.empty([3,1])
for i in range(row):
    for j in range(col):
        Store_Matrix=np.dot(M,im[i,j])
        x1=int(tmp[0]/tmp[2])-refX1
        y1=int(tmp[1]/tmp[2])-refY1
        if x1>0 and y1>0 and y1<refY2-refY1 and x1<refX2-refX1:
            final[y1,x1,:]=im[i,j,:]

cv2.imwrite("_tmp_final.png",final)
# Simple Interpolation
# Interpolate empty pixels from the original image, ignoring pixels outside (extrapolating)
Mi = np.linalg.inv(M)
for i in range(final.shape[0]):
    for j in range(final.shape[1]):
        if sum(final[i,j,:])==0:
            tt = np.array([[j+refX1],[i+refY1],[1]])
            tmp = np.dot(Mi,tt)
            x1=int(tmp[0]/tmp[2])
            y1=int(tmp[1]/tmp[2])

            if x1>0 and y1>0 and x1<im.shape[1] and y1<im.shape[0]:
                final[i,j,:] = im[y1,x1,:]

cv2.imwrite("Warped Image.jpg",final)
image=cv2.imread('Warped Image.jpg')
cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
cv2.imshow('Warped Image',image)
print(np.shape(image))



cv2.waitKey(0)
cv2.destroyAllWindows()
