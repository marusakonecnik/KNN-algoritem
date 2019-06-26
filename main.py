import pandas as pd
import numpy as nm
import math
from collections import defaultdict
from array import *
import operator
import matplotlib
from sklearn import preprocessing
import sys,getopt

def calculateEuclideanDistance(instance1, instance2, length):
	distance = 0
	for i in range(length):
		distance += pow((instance1[i] - instance2[i]), 2)

	return math.sqrt(distance)

def getClosestNeighbours(csvTrainingFile,testInstance,k):
    distances = {}
    row=[]
    neighbors=[]
    for i in range(len(csvTrainingFile)):
        for columnName in csvTrainingFile:
            row.append(csvTrainingFile[columnName][i])
        distance = calculateEuclideanDistance(row, testInstance, len(row)-1)
        distances[i] = distance
        row = []

    sortedByValues = sorted(distances.items(), key=operator.itemgetter(1))
    for i in range(k): neighbors.append(sortedByValues[i])

    return neighbors

def classify(neighbors,csvTrainingData):
    neighborsClasses = []
    for i in range(len(neighbors)):
        neighborIndex = neighbors[i][0];
        neighborsClasses.append(csvTrainingData[csvTrainingData.columns[-1]][neighborIndex]);
    prediction =max(set(neighborsClasses),key=neighborsClasses.count)

    return prediction

#metrices
def accuracy(confusionMatrix):
    correct = 0
    correctlyClassifiedElements =pd.DataFrame(nm.diag(confusionMatrix), index=[confusionMatrix.index, confusionMatrix.columns]).values
    sumOfDiagonalElements = 0

    for value in correctlyClassifiedElements:
        sumOfDiagonalElements+=value

    sumOFAllElements = matrix.values.sum()

    return sumOfDiagonalElements/sumOFAllElements

def precicion(Class,matrix):

    truePositive =matrix.at[Class,Class]
    all = matrix[Class].sum()

    return truePositive/all

def recall(Class,matrix):
    truePositive =matrix.at[Class,Class]
    sum=0
    for columnName in matrix:
        sum+=matrix.at[Class,columnName]

    return truePositive/sum

def FScore(Class,matrix):
    return 2*(precicion(Class,matrix)*recall(Class,matrix))/(precicion(Class,matrix)+recall(Class,matrix))

def confusionMatrix(predictions,testSet,classes):

    matrix = pd.DataFrame(0,index=classes, columns=classes)

    for i in range(len(predictions)):
        correctClass= testSet[testSet.columns[-1]][i]
        if predictions[i]==correctClass: matrix.at[correctClass,correctClass]+=1

        else:    matrix.at[correctClass,predictions[i]]+=1
    return matrix

def main(argv):
    trainingFile = ''
    testFile = ''
    k = ''
    i=False
    try:
        opts,args = getopt.getopt(argv,"ht:T:k:i",["trainFile=,tFile=,kn=,i=="])
    except getopt.GetoptError:
        print('test.py -t <inputfile> -T <outputfile> -k <k> -optional<i>')
    for opt, arg in opts:
        if opt == '-h':
            print
            'test.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-t", "--trainFile"):
            trainingFile = arg
        elif opt in ("-T", "--tFile"):
            testFile = arg
        elif opt in ("-k", "--kn"):
            k = arg
        elif opt in ("-i,-optional"):
            i = True
    return testFile,trainingFile,k,i
     
#main
if __name__ == "__main__":
   test, train,kStr,show =main(sys.argv[1:])
k = (int(kStr))
csvTestFile = pd.read_csv(test)
csvTrainingFile = pd.read_csv(train)

csvTrainingFile.iloc[:,0:-1] = preprocessing.normalize(csvTrainingFile.iloc[:,0:-1])
csvTestFile.iloc[:,0:-1] = preprocessing.normalize(csvTestFile.iloc[:,0:-1])

classes = csvTrainingFile[csvTrainingFile.columns[-1]].unique()
#generate predictions
predictions=[]
row= []
for i in range(len(csvTestFile)):
    for columnName in csvTestFile:
        row.append(csvTestFile[columnName][i])
    neighbors = getClosestNeighbours(csvTrainingFile,row, k)
    predictions.append(classify(neighbors,csvTrainingFile))
    row = []

matrix = confusionMatrix(predictions,csvTestFile,classes)
print(predictions)
print(f"Confusion matrix: \n {matrix}")

if(show==True):
    metrices = pd.DataFrame(0.0,index=["Precision","Recall","FScore"],columns=classes)

    for cl in classes:
        metrices.at['Precision',cl] =precicion(cl,matrix)
        metrices.at['Recall', cl] = recall(cl, matrix)
        metrices.at['FScore', cl] = FScore(cl, matrix)

    metr = ["Precision","Recall","FScore"]

    sums = []
    suma = 0
    for m in metr:
        for columnName in metrices:
            suma += metrices.at[m, columnName]*(matrix[columnName].sum()/len(predictions))
        sums.append(suma)
        suma=0

    print(f"Metrices: \n{metrices} \n")
    print(f"Model accuracy:  {accuracy(matrix)}")
    print("Precision: "+str(sums[0]))
    print("Recall: "+str(sums[1]))
    print("FScore: "+str(sums[2]))
