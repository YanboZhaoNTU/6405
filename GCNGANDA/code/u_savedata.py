import numpy as np
import os

def saveResult(dataName='', algName='', result=[], time1=0, time2=0, time3=0, filename='result.txt'):
    f = open(filename, 'a')
    print(dataName, end='\t', file=f)
    print(algName, end='\t', file=f)
    for i in range(len(result)):
        print(result[i], end='\t', file=f)
    if(time1>0):
        print(time1, end='\t', file=f)
    if(time2>0):
        print(time2, end='\t', file=f)
    if(time3>0):
        print(time3, end='\t', file=f)
    print('', file=f)
    f.close()

def saveMat(mat,dim=2,filename='rst'):
    f = open(filename,'a')
    if(dim==1):
        for i in range(len(mat)):
            print(mat[i],end='\t',file=f)
        print('',file=f)
        print('==============================',file=f)
    elif(dim==2):
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                print(mat[i][j],end='\t',file=f)
            print('',file=f)
        print('==============================',file=f)
    elif(dim==3):
        for i in range(len(mat)):
            print('----------'+str(i+1)+'----------',file=f)
            for j in range(len(mat[i])):
                for k in range(len(mat[i][j])):
                    print(mat[i][j][k],end='\t',file=f)
                print('',file=f)
        print('==============================',file=f)
    else:
        return None
    f.close()

def savearray(mat,filename='rst'):
    dims = np.shape(mat)
    dim = len(dims)
    f = open(filename,'a')
    if(dim==1):
        for i in range(len(mat)):
            print(mat[i],end='\t',file=f)
        print('',file=f)
    elif(dim==2):
        for i in range(dims[0]):
            for j in range(dims[1]):
                print(mat[i][j],end='\t',file=f)
            print('',file=f)
    f.close()
    