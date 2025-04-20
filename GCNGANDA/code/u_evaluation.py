import numpy as np

def evaluate(predict, target, threshold=0.5):
    predict = np.array(predict)
    target = np.array(target)
    if(np.shape(predict) != np.shape(target)):
        print('matrix error')
        return
    dimData,numLabel = np.shape(predict)

    oneerror,coverage,rankingloss,avg_precision = 0,0,0,0
    acc,precision,recall,f1,hamming,hitrate,subsetAcc = 0,0,0,0,0,0,0
    '''classification based'''
    for i in range(dimData):
        a,b,c,d = 0,0,0,0
        for j in range(numLabel):
            if(predict[i][j]>=threshold and target[i][j]==1):
                a +=1
            if(predict[i][j]>=threshold and target[i][j]==0):
                c +=1
            if(predict[i][j]<threshold and target[i][j]==1):
                b +=1
            if(predict[i][j]<threshold and target[i][j]==0):
                d +=1
        if(a+b+c==0):
            acc += 1
            precision += 1
            recall += 1
            f1 += 1
            hitrate += 1
        else:
            acc += a/(a+b+c)
            if(a+c !=0 ):
                precision += a/(a+c)
            if(a+b !=0 ):
                recall += a/(a+b)
            f1 += 2*a/(2*a+b+c)
            if(a>0):
                hitrate += 1
        hamming += (b+c)/(a+b+c+d)
        if(b==0 and c==0):
            subsetAcc += 1
    
    '''ranking based'''
    seq,rank = getSeq(predict)
    dim_rank = dimData
    for i in range(dimData):
        dim_ti = np.sum(target[i])
        if(dim_ti==0 or dim_ti==numLabel):
            dim_rank -= 1
            continue

        if(round(target[i][int(seq[i][0])]) != 1):
            oneerror += 1	#error, when the most confident one is incorrectly estimated
        
        cnt_cov = dim_ti
        r=0
        while(r<numLabel and cnt_cov!=0):
            if( target[i][int(seq[i][r])]==1 ):
                cnt_cov -= 1
            r += 1
        coverage += r
        
        cnt_rank = 0
        for j in range(numLabel):
            if(target[i][j]==0):
                continue
            for k in range(numLabel):
                if(target[i][k]==1):
                    continue
                if(rank[i][j] > rank[i][k]):
                    cnt_rank += 1
        rankingloss += cnt_rank/(dim_ti*(numLabel-dim_ti))

        cnt_pre = 0
        for j in range(numLabel):
            if(target[i][j]==0):
                continue
            tmp = 0
            for k in range(numLabel):
                if(target[i][k]==0):
                    continue
                if(rank[i][j] >= rank[i][k]):
                    tmp += 1
            cnt_pre += tmp/rank[i][j]
        avg_precision += cnt_pre/dim_ti

    coverage /= numLabel
    # print(oneerror,coverage,rankingloss,avg_precision)
    if(dim_rank==0):
        output2 = np.array([0,0,0,1])
    else:
        output2 = np.array([oneerror,coverage,rankingloss,avg_precision])/dim_rank
    output1 = np.array([acc,precision,recall,f1,hitrate,subsetAcc,hamming])/dimData
    return np.append(output1,output2)

def getSeq(y):
    seq = []
    rank = []
    for yi in y:
        # print(np.argsort(yi))
        # print(np.argsort(yi)[::-1])
        # print(np.argsort(np.argsort(yi)[::-1]))
        # print(np.argsort(np.argsort(yi)[::-1])+1)
        tmp = np.argsort(yi)[::-1]
        seq.append(tmp)
        rank.append(np.argsort(tmp)+1)
    return np.array(seq),np.array(rank)

if __name__ == '__main__':
    tst = [[0.45, 0.66, 0.73, 0.29, 0.80]]
    print(getSeq(tst))

    labels = [[1,0,1,0,0],
        [1,0,1,0,0],
        [1,0,1,0,0],
        [1,0,1,0,0],
        [1,0,1,0,0],
        [1,0,1,0,0],
        [1,0,1,0,0],
        [1,0,1,0,0],
        [1,0,1,0,0],
        [1,0,1,0,0],
        [1,0,1,0,0],
        [1,0,1,0,0],
        [0,0,0,0,0],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [0,0,0,0,0]]
    output = [[0.8,0.3,0.7,0.2,0.1],
        [0.8,0.9,0.7,0.2,0.1],
        [0.8,0.75,0.7,0.2,0.1],
        [0.8,0.6,0.7,0.2,0.1],
        [0.4,0.3,0.7,0.2,0.1],
        [0.4,0.3,0.45,0.2,0.1],
        [0.3,0.4,0.45,0.2,0.1],
        [0.3,0.45,0.4,0.2,0.1],
        [0.8,0.7,0.4,0.3,0.2],
        [0.8,0.9,0.4,0.3,0.2],
        [0.8,0.9,0.3,0.4,0.2],
        [0.8,0.9,0.2,0.3,0.4],
        [0.4,0.3,0.2,0.1,0],
        [0.9,0.8,0.7,0.6,0.55],
        [0.9,0.8,0.7,0.6,0.4],
        [0.6,0.3,0.2,0.1,0]]
    print(evaluate(output,labels))
    # from mEvaluation import evaluate as et
    # from u_savedata import saveResult
    # for i in range(len(labels)):
    #     saveResult(result=np.array(evaluate([output[i]],[labels[i]],0.5)))
    #     saveResult(result=np.array(et([output[i]],[labels[i]])))
        