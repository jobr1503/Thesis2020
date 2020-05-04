# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:08:23 2020

@author: jbgba
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler   
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import display,Image
import pydotplus
import time
from datetime import datetime
import math
#import csv
import sys
import random


#MLP extension of DT
def neural_network_MLPClassifier(features,target):
    features = features.astype(float)
    target = target.astype(float)   
    from sklearn.neural_network import MLPClassifier   
    clf = MLPClassifier(activation = 'tanh', learning_rate_init = 1e-3, momentum = 0.9  , tol = 1e-5, max_iter = 10000, n_iter_no_change = 10)
    clf.fit(features,target)
    predicted = clf.predict(features)
    return predicted
    #expected = y_test
    #print("--------------------------------------\nNeural Network MLPClassifier\n")
    #print(model)
    #print(confusion_matrix(expected, predicted))
    #print(classification_report(expected, predicted))
    #print('Total Accuracy: ',accuracy_score(expected,predicted))
    #return accuracy_score(y_test, predicted)


#Building DT    
def splittingNode(dataset,q,returnThis,binVector):
    #dataset=dataset.astype(np.int64) ##ONLY WHEN NOT NORMALIZED
    dataset=dataset.astype(float)
   # print('NEW ITERATION\n',q)
        
    #Calculates no of "pos","neg" in dataset sent
    count1=1 #NOTERA DETTA FÖR ATT SLIPPA DIVISION MED 0
    count0=1
    for i in range(len(dataset)):
        if dataset[i,0]==1:
            count1=count1+1
        if dataset[i,0]==0:
            count0=count0+1
    #print(count1,count0)        
    vote=0 #Sätter ett startvärde som inte förekommer någon annanstans
    if count1>=count0:
        vote=1
    if count0>count1:
        vote=-1
    #if count1==count0:
    #    vote=-1
    #Calculate entropy of class in the dataset sent
    EY=-(((count1/(count1+count0))*math.log2(count1/(count1+count0)))+((count0/(count1+count0))*math.log2(count0/(count1+count0))))
    #print(EY)
    
    #Calculates all means between adjancent elements
    means=np.zeros([len(dataset)-1,len(np.matrix.transpose(dataset))-1])
    for j in range(len(np.matrix.transpose(dataset))-1): #Target attribute should not be considered
        if j>0:  #Ignores target attribute
            dataset=dataset[dataset[:,j].argsort()] #Sorts dataset in numeric order based on the attribute j
            for i in range(len(dataset)-1):
                means[i,j-1]=(float(dataset[i,j])+float(dataset[i+1,j]))/float(2)
                #means[i,j-1]=math.floor((dataset[i,j]+dataset[i+1,j])/2)

    #print('DATASET: ',pd.DataFrame(dataset))
    #print('MEANS: ',means)

    splitNode=np.zeros(len(np.matrix.transpose(dataset))-1)
    splitPoint=np.zeros(len(np.matrix.transpose(dataset))-1)
    IYAi=np.zeros(len(np.matrix.transpose(dataset))-1)
    EYAij=np.zeros(len(np.matrix.transpose(dataset))-1)
    #Calculations of a DT
    for j in range(len(np.matrix.transpose(dataset))):
        minEYAij=0
        if j>0:
            a=dataset[:,j]
            ##MLP EXTENSION
            for y in range(len(binVector)):
                for t in range(len(np.transpose(binVector))):
                    if j==binVector[y,t]:
                        a=neural_network_MLPClassifier(dataset[:,binVector[y,:]],dataset[:,0])
            ##MLP EXTENSION
            a=a[a.argsort()] #Sorted version of attribute j
            a=a.astype(float)
            for i in range(len(dataset)-1):
                    #Counts classes pos,neg based on splitpoint
                    count1Less=1
                    count0Less=1
                    count1More=1
                    count0More=1
                    dataset=dataset[dataset[:,j].argsort()]
                    for k in range(len(dataset)):
                        if k<means[i,j-1]:
                            if dataset[k,0]==1:
                                count1Less=count1Less+1
                            if dataset[k,0]==0:
                                count0Less=count0Less+1
                        if k>=means[i,j-1]:
                            if dataset[k,0]==1:
                                count1More=count1More+1
                            if dataset[k,0]==0:
                                count0More=count0More+1
                    #Calculates entrophy based on splitpoint j
                    EYAij[j-1]=(((len(np.array(np.where(a < means[i,j-1])[0]))/len(a)))*(-(((count1Less/(count1Less+count0Less))*math.log2(count1Less/(count1Less+count0Less)))+((count0Less/(count1Less+count0Less))*math.log2(count0Less/(count1Less+count0Less))))))+(((len(np.array(np.where(a >= means[i,j-1])[0]))/len(a)))*(-(((count1More/(count1More+count0More))*math.log2(count1More/(count1More+count0More)))+((count0More/(count1More+count0More))*math.log2(count0More/(count1More+count0More)))))) #ADD THE REST!
                    #Look up the smallest entrophy in attribute j
                    if i==0: 
                        minEYAij=EYAij[j-1]
                        splitNode[j-1]=j
                        splitPoint[j-1]=means[i,j-1]
                    else:
                        if  EYAij[j-1]<minEYAij:
                            minEYAij=EYAij[j-1]
                            splitNode[j-1]=j
                            splitPoint[j-1]=means[i,j-1]
        #Calculates information gained
        IYAi[j-1]=EY-minEYAij
        #if IYAi[j-1]<=0:
        #    IYAi[j-1]=-1

    #Calculates values to return
    #print(IYAi)
    
    #print('IYAi',pd.DataFrame(IYAi))
    #IYAi=np.array(pd.DataFrame(IYAi))
    #setting='internalNode'
    #print(IYAi)
    setting='internalNode'
    posInfoControl=0
    maxInfo=0
    maxInfoIndex=0
    datasetLower=[]
    datasetHigher=[]
    splitPointInSplitAttr=0
    for i in range(len(IYAi)):
       if IYAi[i]>maxInfo:
            maxInfo=IYAi[i]
            maxInfoIndex=i
            posInfoControl=1
            #print('MAX INFO:',maxInfo)
            #print('MAX INFO INDEX:',maxInfoIndex)
               
            splitPointInSplitAttr=splitPoint[maxInfoIndex] #The splitpoint of that attribute
            
            #for i in range(len(splitPoint)):
            #    if splitPoint[i]==splitPointInSplitAttr:
            #        maxInfoIndex=i
            
            #Splits dataset into 2 datasets
            dataset=dataset[dataset[:,maxInfoIndex].argsort()] #Sort dataset based on attribute that yeilded highest accuracy
            #rowsLower=np.array(np.where(dataset[:,maxInfoIndex] < splitPointInSplitAttr)[0]) #How many rows in dataset are below the splitpoint
            #rowsHigher=np.array(np.where(dataset[:,maxInfoIndex] >= splitPointInSplitAttr)[0])
            rowsLower=np.where(dataset[:,maxInfoIndex].astype(float) < float(splitPointInSplitAttr))[0] #How many rows in dataset are below the splitpoint
            rowsHigher=np.where(dataset[:,maxInfoIndex].astype(float) >= float(splitPointInSplitAttr))[0]
            datasetLower=np.array(dataset[rowsLower,:],dtype='<U32') #I NEED TO STORE THESE IN AN 3-dim MATRIX (OTHERWISE THEY WILL BE OVERWRITTEN EACH TIME). ALSO GIVES GOOD WAY TO SENT FUNCTION CALLS SYSTEMATICALLY
            datasetHigher=np.array(dataset[rowsHigher,:],dtype='<U32') #1:len(dataset),:
            
            #if IYAi[i]<=maxInfo:
            #    maxInfo=IYAi[i]
            #    maxInfoIndex=i
            #    print('MAX INFO:',maxInfo)
            #    print('MAX INFO INDEX:',maxInfoIndex)
            #posInfoControl=1
    if posInfoControl==0:
        setting='Leaf'
    #    maxInfo=0
    #    maxInfoIndex=0
    #maxInfo=M #max info gain from all attributes
    #splitAttr=maxInfoIndex #The index of attribute with max info gain from split


    #Vote class
    oneLower=0
    zeroLower=0
    for w in range(len(datasetLower)):
        if datasetLower[w,0]==1:
            oneLower=oneLower+1
        if datasetLower[w,0]==0:
            zeroLower=zeroLower+1
    if oneLower>=zeroLower:
        voteclassLower=1
    if zeroLower>oneLower:
        voteclassLower=-1
    #if oneLower==zeroLower:
    #    voteclassLower=-1
    oneHigher=0
    zeroHigher=0
    for w in range(len(datasetHigher)):
        if datasetHigher[w,0]==1:
            oneHigher=oneHigher+1
        if datasetHigher[w,0]==0:
            zeroHigher=zeroHigher+1     
    if oneHigher>=zeroHigher:
        voteclassHigher=1
    if zeroHigher>oneHigher:
        voteclassHigher=-1
    #if oneHigher==zeroHigher:
    #    voteclassHigher=-1
    
    
    #for i in range(q*2):        
    #    villkorsMatrix[q,i]=splitPointInSplitAttr
    #list.insert(index, elem)
    #EVENTUELLT NÅGOT KNASIGT HÄR. BORDE KUNNA ANVÄNDA MIG AV vstack OM JAG VET VILKA index SOM HÖR TILL VARJE PARTITION.
    returnThis=[]
    #if q==0: #DELA UPP NEDAN LITE, JAG BEHÖVER INTE VETA INFO FÖR NÄSTA NOD ÄNNU!!!!!!
    returnThis=[setting,q,vote,maxInfoIndex,splitPointInSplitAttr,maxInfo,'higherThanSplitPoint',voteclassHigher,datasetHigher,'lowerThanSplitPoint',voteclassLower,datasetLower]
    
    #for i in range(2):
    #d=len...
    ###If q>0 = Internal Node. #If maxInfo<=0 eller len<5 = Leaf #Max 3 lager #Vill bara se info för en nod, inte för nästa noder (fixa detta genom att lägga över information från returnThis till en annan matris (utan dataseten)) 
    
    #if q>0:
    #    returnThisThis=['LowerLeaf',q,vote,splitAttr,splitPointInSplitAttr,maxInfo,'higherThanSplitPoint',voteclassHigher,datasetHigher,'lowerThanSplitPoint',voteclassLower,datasetLower]
    #    returnThisThis=np.array(returnThisThis,dtype='object')
        #returnThis=np.vstack([returnThis,returnThisThis])
    #    for i in range(len(returnThisThis)):
    #        returnThis.insert(len(returnThisThis)+i,returnThisThis[i])
    
    #KAN LOOPA ÖVER ALLA ELEMENT OCH SEDAN KANSKE TRANSFORMERA .reshape[9,-1]?
        #np.insert(returnThis,0,returnThisThis,axis=0)
        #returnThis=np.column_stack([returnThis,returnThisThis])
    #D=np.vstack([D,[datasetLower,datasetHigher]])
    #returnThis=returnThis.reshape([9,-1])
    #returnThis = np.reshape(returnThis, (9, -1))
    #np_array=np.asarray(returnThis)
    # reshape array into 4 rows x 2 columns, and transpose the result
    #returnThis = np_array.reshape(9, -1) 
    #h=features[:,binVector[i]]
    #H=np.column_stack([H,h])
    
    #print("")
    #print(returnThis)
    #print(len(returnThis))
    
    #Stores so I can redo all splits when evavulating
    #returnThis[q]=[maxInfo,splitAttr,splitPointInSplitAttr] #Info gain, Attribute that should be splitted, Value of means (splitpoint) that yeilded highest info gain
    
    #print("")
    #print(returnThis)
    q=q+1
    return returnThis,q


#PROGRAM START
#TRAINING: Construction of training data and building model  
iterations=1
accuracy=np.zeros(iterations)
for n in range(iterations):    
    startTime = datetime.now() 
    #length=50
    length=1000
    
    #randNr=random.randrange(0,100)
    #print(randNr)
    
    histogramVariable1=np.zeros([length,10])
    histogramVariable2=np.zeros([length,10])
    
    
    for i in range(len(histogramVariable1)):
        for j in range(10):
            histogramVariable1[i,j]=random.randrange(0,100)
            histogramVariable2[i,j]=random.randrange(0,100)
    
    #print(pd.DataFrame(histogramVariable1))
    
    norm_histogramVariable1=np.zeros([length,10])
    norm_histogramVariable2=np.zeros([length,10])
    
    for i in range(len(histogramVariable1)):
        summaHV1=sum(histogramVariable1[i])
        summaHV2=sum(histogramVariable2[i])
        for j in range(10):
            norm_histogramVariable1[i,j]=histogramVariable1[i,j]/summaHV1
            norm_histogramVariable2[i,j]=histogramVariable2[i,j]/summaHV2
            
    #print(pd.DataFrame(norm_histogramVariable1))
    
    ##CLASSES:
    target=np.zeros([length,1])
    for k in range(len(histogramVariable1)):
        controlHV1=0
        controlHV2=0
        if (norm_histogramVariable1[k,0]+norm_histogramVariable1[k,1]+norm_histogramVariable1[k,2])<0.35:
            controlHV1=1
            #print('HV1: ',k,' ',controlHV1)
        if (norm_histogramVariable2[k,0]+norm_histogramVariable2[k,1]+norm_histogramVariable2[k,2])<0.35:
            controlHV2=1
            #print('HV2: ',k,' ',controlHV2)
        if controlHV1==1 and controlHV2==1:
                target[k]=1
                #print('Target')
    
    #print(pd.DataFrame(target))
    
    syntesData=np.column_stack((histogramVariable1,histogramVariable2))
    syntesData=np.column_stack((target,syntesData))
    
    norm_syntesData=np.column_stack((norm_histogramVariable1,norm_histogramVariable2))
    norm_syntesData=np.column_stack((target,norm_syntesData))
    #dataset=norm_syntesData
    dataset=syntesData
    #print(pd.DataFrame(norm_syntesData))
    
    #q=0
    #returnThis=[]
    bin1=[1,2,3,4,5,6,7,8,9,10]
    bin2=[11,12,13,14,15,16,17,18,19,20]
    binVector=[bin1,bin2]
    binVector=np.array(binVector)
    
    #returnThis1=splittingNode(dataset,q,returnThis,binVector)
    
    
    ##BUILDING TREE AND EVAVULATING
    lowerDataset00=[]
    lowerDataset10=[]
    higherDataset10=[]
    higherDataset00=[]
    lowerDataset11=[]
    higherDataset11=[]
    
    returnThis=[]
    classMatrix=np.zeros([5,16])
    villkorsMatrix=np.ones([5,16])
    attrSplitMatrix=np.ones([5,16],dtype=int)
    infoGainMatrix=np.ones([5,16])
    #villkors=[]
    #D=[]
    q=0 ##HA EN MED MLP OCH EN UTAN MLP!!!!!!!!!!!!!!!!!!!!
    ####SÖKA PÅ lista i lista Python OCH SE VART DET LEDER MIG!!!!
    ##ROOT NODE 0.0
    if len(dataset)>0: ##Stop critera 1 #STOP CRITERIA 2 är att info gain = 0 (ta bort maxspärr i splitnoden)!!!!
        returnThis00,q=splittingNode(dataset,q,returnThis,binVector)
        #villkors=returnThis[4]
        villkorsMatrix[0,0]=returnThis00[4]
        classMatrix[0,0]=returnThis00[2]
        attrSplitMatrix[0,0]=returnThis00[3]
        infoGainMatrix[0,0]=returnThis00[5]
        higherDataset00=returnThis00[8]
        lowerDataset00=returnThis00[11]
    
    ##Gren lower 1.0
    if len(lowerDataset00)>0:
        returnThis10,q=splittingNode(lowerDataset00,q,returnThis00,binVector) #Ändra namn på returnThis? NEJ, ÄNDRA TILLBAKA
        #lower=returnThis[4]
        try:
            villkorsMatrix[1,0]=returnThis10[4]
            classMatrix[1,0]=returnThis10[2]
            attrSplitMatrix[1,0]=returnThis10[3]
            infoGainMatrix[1,0]=returnThis10[5]
            higherDataset10=returnThis10[8]
            lowerDataset10=returnThis10[11]
        except ValueError:
            villkorsMatrix[1,0]=-1
            higherDataset10=-1
            lowerDataset10=-1
    
    ##2.0
    if len(lowerDataset10)>0:
        returnThis20,q=splittingNode(lowerDataset10,q,returnThis10,binVector)
        #higher=returnThis[4]
        try:
            villkorsMatrix[2,0]=returnThis20[4]
            classMatrix[2,0]=returnThis20[2]
            attrSplitMatrix[2,0]=returnThis20[3]
            infoGainMatrix[2,0]=returnThis20[5]
            higherDataset20=returnThis20[8]
            lowerDataset20=returnThis20[11]
        except ValueError:
            villkorsMatrix[2,0]=-1
            higherDataset20=-1
            lowerDataset20=-1
    
    ##2.1
    if len(higherDataset10)>0:
        returnThis21,q=splittingNode(higherDataset10,q,returnThis10,binVector)
        #higher=returnThis[4]
        try:
            villkorsMatrix[2,1]=returnThis21[4]
            classMatrix[2,1]=returnThis21[2]
            attrSplitMatrix[2,1]=returnThis21[3]
            infoGainMatrix[2,1]=returnThis21[5]
            higherDataset21=returnThis21[8]
            lowerDataset21=returnThis21[11]
        except ValueError:
            villkorsMatrix[2,1]=-1
            higherDataset21=-1
            lowerDataset21=-1
    
    ##Gren higher #1.1
    if len(higherDataset00)>0:
        returnThis11,q=splittingNode(higherDataset00,q,returnThis00,binVector) ##FEL HÄR!!!
        #higher=returnThis[4]
        try:
            villkorsMatrix[1,1]=returnThis11[4]
            classMatrix[1,1]=returnThis11[2]
            attrSplitMatrix[1,1]=returnThis11[3]
            infoGainMatrix[1,1]=returnThis11[5]
            higherDataset11=returnThis11[8]
            lowerDataset11=returnThis11[11]
        except ValueError:
            villkorsMatrix[1,1]=-1
            higherDataset11=-1
            lowerDataset11=-1
    #2.2
    if len(lowerDataset11)>0:
        returnThis22,q=splittingNode(lowerDataset11,q,returnThis11,binVector)
        #higher=returnThis[4]
        try:
            villkorsMatrix[2,2]=returnThis22[4]
            classMatrix[2,2]=returnThis22[2]
            attrSplitMatrix[2,2]=returnThis22[3]
            infoGainMatrix[2,2]=returnThis22[5]
            higherDataset22=returnThis22[8]
            lowerDataset22=returnThis22[11]   
        except ValueError:
            villkorsMatrix[2,2]=-1
            higherDataset22=-1
            lowerDataset22=-1  
    
    #2.3 ##JAG HAR JU NÄSTKOMMANDE KLASS OCKSÅ, UTNYTTJA DET OM INTE infoGain<=0
    if len(higherDataset11)>0:
        returnThis23,q=splittingNode(higherDataset11,q,returnThis11,binVector)
        #higher=returnThis[4]
        try:
            villkorsMatrix[2,3]=returnThis23[4]
            classMatrix[2,3]=returnThis23[2]
            attrSplitMatrix[2,3]=returnThis23[3]
            infoGainMatrix[2,3]=returnThis23[5]
            higherDataset23=returnThis23[8]
            lowerDataset23=returnThis23[11]
        except ValueError:
            villkorsMatrix[2,3]=-1
            higherDataset23=-1
            lowerDataset23=-1
    
    
    print('VILLKORSMATRIX \n')
    print(pd.DataFrame(villkorsMatrix))
    print("")
    print('SPLIT ATTR \n')
    print(pd.DataFrame(attrSplitMatrix))
    print("")
    print('CLASS MATRIX \n')
    print(pd.DataFrame(classMatrix))
    print("")
    print('INFO GAIN MATRIX \n')
    print(pd.DataFrame(infoGainMatrix))
    print("")
    
    ####---->>>>>EVAVULATION
    #for i in range(len(features)):
    #        for j in range(len(np.matrix.transpose(features))):
    #            features[i,j]=toInt(features[i,j])
    #features=features.astype(int)
    
    
    ##confusionMatrix1=pd.crosstab(target.astype(int),classes)
    ##confusionMatrix = pd.DataFrame(confusionMatrix1)
    ##plt.figure(figsize=(3,3))
    ##ax = plt.axes()
    ##ax.set_title('confusion matrix')
    ##sns.heatmap(confusionMatrix,ax=ax,annot=True,fmt='n',cmap="YlGnBu",yticklabels=True,xticklabels=True)
        
    #villkors.insert(q,[higher,lower])
    
    #if q==1:
    #    tree=[]
    #    for i in range(6):
    #        tree.insert(i,returnThis[i])
    
    #if len(returnThis[8])>4:
    #    returnThis,q=splittingNode(returnThis[8],q,returnThis)
    #if len(returnThis[11])>4:
    #    returnThis,q=splittingNode(returnThis[11],q,returnThis)
        
    #treeCol=[]
    #if q>1:
    #    for i in range(6):
    #        treeCol.insert(i,returnThis[(q-1)*12+i])
    #    tree=np.column_stack((tree,treeCol))
    
    #stackedGeneralization(dataset)
    
    
    #SKICKA IN DATASETEN IGEN SYSTEMATISKT?
        #Lägre, högre
        #Lägre, högre, Lägre, högre,
        #Lägre, högre, lägre, högre, lägre, högre
        #TILL DESS ATT NÅGOT DATASET <len(5) eller info gain<=0
        #NOTERA DETTA I OUTPUTEN, 1, 2.0,2.1,2.0.3.0,2.0.3.1,2.1.3.0,2.1.3.1, OSV? #FRÅGA TONY OM DETTA!!!!
    
    #for i in range(q):
        #print('hej')
        #returnThis,q=upper(returnThis[8+(i*12)],q,returnThis)
    
    #for i in range(q):
        #print('hej')
    #returnThis,q=higherDataset(returnThis[8],q,returnThis)
    
    #returnThis,q=splittingNode(returnThis[8],q,returnThis)
    #returnThis,q=splittingNode(returnThis[20],q,returnThis)
    
    
     
    #np.savetxt('testfile.txt',dataset,delimiter=',',fmt='%s') 
    
    
    #DT(dataset,featureNames)
    #for i in range(len(D)):
    #    if len(D[i])>4:
    #        D,returnThis,q=splittingNode(D[i],q,D,returnThis)
    #dataset=list(dataset)
    #DT(dataset,featureNames)
    
    #DT(dataset,featureNames)
    
    #print(len(featureNames))
    #print(len(np.transpose(dataset)))
    
    
    



#C=[-1,1,1,1,-1,1,-1,1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,1,1,1,1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,-1,-1,1,1,]
#C_pred=[-1,1,1,1,-1,1,-1,1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,1,1,1,1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,1,1,]
#print(confusion_matrix(C,C_pred))
#print(classification_report(C,C_pred))
#print('Total Accuracy: ',accuracy_score(C,C_pred))   


#TESTING: Construction of testing data and evaluating model 
    #length=50
    length=1000
    
    #randNr=random.randrange(0,100)
    #print(randNr)
    
    histogramVariable1=np.zeros([length,10])
    histogramVariable2=np.zeros([length,10])
    
    
    for i in range(len(histogramVariable1)):
        for j in range(10):
            histogramVariable1[i,j]=random.randrange(0,100)
            histogramVariable2[i,j]=random.randrange(0,100)
    
    #print(pd.DataFrame(histogramVariable1))
    
    norm_histogramVariable1=np.zeros([length,10])
    norm_histogramVariable2=np.zeros([length,10])
    
    for i in range(len(histogramVariable1)):
        summaHV1=sum(histogramVariable1[i])
        summaHV2=sum(histogramVariable2[i])
        for j in range(10):
            norm_histogramVariable1[i,j]=histogramVariable1[i,j]/summaHV1
            norm_histogramVariable2[i,j]=histogramVariable2[i,j]/summaHV2
            
    #print(pd.DataFrame(norm_histogramVariable1))
    
    ##CLASSES:
    target=np.zeros([length,1])
    for k in range(len(histogramVariable1)):
        controlHV1=0
        controlHV2=0
        if (norm_histogramVariable1[k,0]+norm_histogramVariable1[k,1]+norm_histogramVariable1[k,2])<0.35:
            controlHV1=1
            #print('HV1: ',k,' ',controlHV1)
        if (norm_histogramVariable2[k,0]+norm_histogramVariable2[k,1]+norm_histogramVariable2[k,2])<0.35:
            controlHV2=1
            #print('HV2: ',k,' ',controlHV2)
        if controlHV1==1 and controlHV2==1:
                target[k]=1
                #print('Target')
    
    #print(pd.DataFrame(target))
    
    syntesData=np.column_stack((histogramVariable1,histogramVariable2))
    syntesData=np.column_stack((target,syntesData))
    
    norm_syntesData=np.column_stack((norm_histogramVariable1,norm_histogramVariable2))
    norm_syntesData=np.column_stack((target,norm_syntesData))
    #dataset=norm_syntesData
    dataset=syntesData
    
    classes=np.zeros([len(dataset)]) ##LÄGGA IN INFO GAIN I EKV. SÅ ATT LEAFs INTE BLIR VALDA!!!!
    for i in range(len(dataset)):
    #    if attrSplitMatrix[0,0]!=1:
            if infoGainMatrix[0,0]!=1:
                if infoGainMatrix[0,0]>0:
                    classes[i]=classMatrix[0,0]
                    ##Root node
                    if int(dataset[i,attrSplitMatrix[0,0]])<villkorsMatrix[0,0]:
                        if infoGainMatrix[1,0]!=1:
                            if infoGainMatrix[1,0]>0:
                                classes[i]=classMatrix[1,0]
                                if int(dataset[i,attrSplitMatrix[1,0]])<villkorsMatrix[1,0]:
                                    if infoGainMatrix[2,0]!=1:
                                        if infoGainMatrix[2,0]>0:
                                            classes[i]=classMatrix[2,0]
                                else:
                                    if infoGainMatrix[2,1]!=1:
                                        if infoGainMatrix[2,1]>0:
                                            classes[i]=classMatrix[2,1]
                    else:
                        if infoGainMatrix[1,1]!=1:
                            if infoGainMatrix[1,1]>0:
                                classes[i]=classMatrix[1,1]
                                if int(dataset[i,attrSplitMatrix[1,1]])<villkorsMatrix[1,1]:
                                    if infoGainMatrix[2,2]!=1:
                                        if infoGainMatrix[2,2]>0:
                                            classes[i]=classMatrix[2,2]
                                else:
                                    if infoGainMatrix[2,3]!=1:
                                        if infoGainMatrix[2,3]>0:
                                            classes[i]=classMatrix[2,3]
    #            else:
    #                classes[i]=
                    
    #print('CLASSES: ',classes)
    
    for i in range(len(classes)):
        if classes[i]==-1:
            classes[i]=0
    
    ##CONFUSION MATRIX (FIX WITH BETTER VISUALIZATION!!!)
    print(confusion_matrix(target.astype(int),classes))
    accuracy[n]=accuracy_score(target.astype(int),classes)
    print(classification_report(target.astype(int),classes))
    print('Total Accuracy: ',accuracy_score(target.astype(int),classes))                
    #for i in range(len(villkorsMatrix)):
    #    for j in range(len(np.transpose(villkorsMatrix))):
    #        if villkorsMatrix[i,j]!=1:
    #            if features[attrSplitMatrix[i,j]]:
    #                print('')
    
    print('\nRun-time: ', datetime.now() - startTime)

####FÅ DET ATT DUNGERA UTAN NORMALISERING FÖRST!!!!


runs=np.zeros([iterations])
for i in range(len(runs)):
    runs[i]=i+1

plt.plot(runs,accuracy,color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=12)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy Graph')
plt.grid
plt.show()

mean=0
for i in range(len(accuracy)):
    mean=mean+(accuracy[i]/len(accuracy))
print(mean)
print(max(accuracy))
    
    