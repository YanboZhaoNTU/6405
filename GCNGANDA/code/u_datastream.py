from u_mReadData import *
import math

class stream():
    def __init__(self, dataX, dataY, num_trunk=6):
        self.dataX = dataX
        self.dataY = dataY
        n = len(dataY)
        batch_size = int(math.ceil(n/num_trunk))
        id1 = 0
        self.idx = []
        for i in range(num_trunk):
            id2 = min(id1+batch_size,n)
            self.idx.append(np.arange(id1,id2))
            id1 = id2
        self.num_trunk = num_trunk

    def datastream(self,trunkId):
        if(trunkId>self.num_trunk):
            print('stream error')
            return None
        return self.dataX[self.idx[trunkId]]
    
    def datastream_label(self,trunkId):
        if(trunkId>self.num_trunk):
            print('stream error')
            return None
        return self.dataY[self.idx[trunkId]]

if __name__ == '__main__':
    numdata = 14
    datasnames = ["Birds","CHD_49","Corel5k","Emotions","GpositiveGO","Image","Langlog","Scene","Chemistry","Philosophy","Tmc2007_500","Water_quality","Yeast","Yelp"]
    rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
    # rd = ReadData(datas=datasnames,genpath='data/')

    num_trunk = 6
    '''k-fold with 1 result'''
    r_unlabeled = [4]
    for dataIdx in range(numdata):
        print(dataIdx)
        X,Y = rd.readDataall(dataIdx)
        mstream = stream(X,Y, num_trunk=num_trunk)
        X0, Y0 = mstream.datastream(0),mstream.datastream_label(0)

        for trunk in range(num_trunk-1):
            Xtr = mstream.datastream(trunk+1)
            Xte = mstream.datastream_label(trunk+1)

    
