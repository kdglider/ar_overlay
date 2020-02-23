import numpy as np
from numpy import array
from scipy.linalg import svd
np.set_printoptions(suppress=True)

"""x= reference corner = x
xp=img coner =xp"""

"""def Get_Homography(x, xp):
    A=np.array([[-x[0,0],-x[0,1],-1,0,0,0,x[0,0]*xp[0,0],x[0,1]*xp[0,0],xp[0,0]],
                [0,0,0,-x[0,0],-x[0,1],-1,x[0,0]*xp[0,1],x[0,1]*xp[0,1],xp[0,1]],
                [-x[1,0],-x[1,1],-1,0,0,0,x[1,0]*xp[1,0],x[1,1]*xp[1,0],xp[1,0]],
                [0,0,0,-x[1,0],-x[1,1],-1,x[1,0]*xp[1,1],x[1,1]*xp[1,1],xp[1,1]],
                [-x[2,0],-x[2,1],-1,0,0,0,x[2,0]*xp[2,0],x[2,1]*xp[2,0],xp[2,0]],
                [0,0,0,-x[2,0],-x[2,1],-1,x[2,0]*xp[2,1],x[2,1]*xp[2,1],xp[2,1]],
                [-x[3,0],-x[3,1],-1,0,0,0,x[3,0]*xp[3,0],x[3,1]*xp[3,0],xp[3,0]],
                [0,0,0,-x[3,0],-x[3,1],-1,x[3,0]*xp[3,1],x[3,1]*xp[3,1],xp[3,1]]])

    U, s, VT = svd(A)
    V=np.transpose(VT)
    global HM
    HM=V[:,-1]
    a=0
    b=0
    
    #for loop is for verification that |H|=1 

    for i in range(len(HM)):
        b=HM[i]*HM[i]
        a=a+b
    print("|H|=",a)
    HM=np.reshape(HM,(3,3))
    print("Homography Matrix=",HM)
    print("Type",type(HM))"""

def Rot_Trans_Mat(Homo_OK,insintric_par):
    inv_insintric_par=np.linalg.inv(insintric_par)
    dot1=inv_insintric_par.dot(Homo_OK[:, 0])
    dot2=inv_insintric_par.dot(Homo_OK[:, 1])
    Lamda=((np.linalg.norm(dot1)+np.linalg.norm(dot2))/2)**-1
    print(Lamda)
    B_Tilda=Lamda*inv_insintric_par.dot(Homo_OK)
    if np.linalg.det(B_Tilda)<0:
        B=Lamda*B_Tilda*-1
    else:
        B=Lamda*B_Tilda
    r1=np.array([B[:,0]])
    r2=np.array([B[:,1]])
    r3=np.cross(r1,r2)
    r1=Lamda*r1
    r2=Lamda*r2
    t=np.array([B[:,2]])
    t=Lamda*t
    print(r1)
    print(r2)
    print(r3)
    print(t)
    q=np.vstack((r1,r2))
    q=np.vstack((q,r3))
    q=np.vstack((q,t))
    Q=np.transpose(q)
    print("Rot_Trans_Mat:",Q)
    print("Projection Matrix:",insintric_par.dot(Q))


#i=np.array([[1,2],[2,3],[5,6],[6,5]]) # points passed for homograpgy 
#j=np.array([[16,23],[25,31],[35,6],[66,58]])
camera_parameters = np.array([[ 21, 51, 7 ],[14, 37, 9 ],[ 43, 7, 9 ]]) 

Get_Homography(i,j)
Rot_Trans_Mat(HM,camera_parameters) #remove """ """ and # to run the code since it passes HM as argument 
 







