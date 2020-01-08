from scipy.signal import argrelextrema
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import operator
import os
import random
import cv2
import collections
import seaborn as sn
import matplotlib.pyplot as plt

# plot the confusion matrices
def plotConfusionMatrix(confusion, title, labels=[]):

    # plot the matrix as a heatmap
    fig = plt.axes()
    fig.set_title(title)
    sn.heatmap(confusion, ax=fig, fmt ='g', cmap='BuPu', annot=True, \
               xticklabels=labels, yticklabels=labels)
    plt.savefig(title)

    plt.close()

    return

# calculate the mahalanobis distance
def getMDistance(x, y_mean, y_var):

    covInverse = np.linalg.inv(y_var)
    subMean = x - y_mean
    det = np.linalg.det(y_var)
    temp1 = -0.5*np.log(det)
    temp2 = 0.5*(np.dot(subMean.T,np.dot(covInverse,subMean)))
    mDistance = temp1-temp2

    return mDistance

# get the max mahalanobis distance
def getProb(vector, classes, means, vars):

    # initialize probability of class
    probClass = dict()

    # get the probability of the feature
    for c in classes:
        probClass[c] = getMDistance(vector, means[c], vars[c])

    mostLikely = max(probClass.items(), key=operator.itemgetter(1))[0]

    return mostLikely

# find the probability distributions for edges, corners, and flat regions
def findDistributions(r, v):

    # find all the local maximum
    maxInd = argrelextrema(r, np.greater)
    maxExtrema = r[maxInd]

    # find all the local minimum
    minInd = argrelextrema(r, np.less)
    minExtrema = r[minInd]

    # initialize and apply threshold
    cornerThreshold = np.mean(maxExtrema, axis=0) * 4
    edgeThreshold = np.mean(minExtrema, axis=0) * 2.1

    # initialize lists for our corners, edges, and flat regions
    corners = []
    flats = []
    edges = []
    y_true = []
    i = 0

    # find corners, edges, and flats
    for x in range(r.shape[0]):
        for y in range(r.shape[1]):

            if r[x,y] >= cornerThreshold:
                cl = 'corners'
                corners.append(v[i])
            elif r[x,y] <= edgeThreshold:
                cl = 'edges'
                edges.append(v[i])
            else:
                cl = 'flats'
                flats.append(v[i])

            y_true.append(cl)
            i += 1

    # convert to numpy array for easier processing
    corners = np.array(corners)
    flats = np.array(flats)
    edges = np.array(edges)

    # get class means
    cornerMean = corners.mean(axis=0)
    flatMean = flats.mean(axis=0)
    edgeMean = edges.mean(axis=0)

    # get class covariance matrices
    cornerVar = np.cov(corners.T)
    edgeVar = np.cov(edges.T)
    flatVar = np.cov(flats.T)

    # put everything into a dictionary
    means = dict()
    vars = dict()

    means['corners'] = cornerMean
    means['flats'] = flatMean
    means['edges'] = edgeMean

    vars['corners'] = cornerVar
    vars['flats'] = flatVar
    vars['edges'] = edgeVar

    return means, vars, y_true

# predict the classes given the distribution and calculate error rates
def findErrorRates(r, v):

    # initialize list of our classes
    classes = ['corners', 'flats', 'edges']

    # get the probablity distribution of each class
    means, vars, y_true = findDistributions(r, v)

    # list of predicted classes
    y_pred = []

    # find the class for each points eigen values
    for key, value in v.items():
        cl = getProb(value, classes, means, vars)
        y_pred.append(cl)

    # get total accuracy and other metrics
    report = classification_report(y_true, y_pred, labels=classes)
    accuracy = accuracy_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred, labels=classes)

    # plot confusion matrix
    plotConfusionMatrix(confusion, title='HCD1 Confusion Matrix', labels=classes)

    return accuracy, report

# get the max 10 and min 10 values of r for one image
def getFeatures(key, image):

    file = 'DigitDataset/' + key + "/" + image
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.float32(img)
    r = cv2.cornerHarris(img, 3, 3, 0.05)
    r = r.flatten()
    r = np.sort(r)
    low = list(r[0:10])
    up = list(r[-10:])
    features = low + up
    features = np.asarray(features)

    return features

# get the max 10 and min 10 values of r for an entire set of images
def getFeaturesSet(set):

    features = dict()

    for key, value in set.items():
        features[key] = []
        for v in value:
            f = np.atleast_2d(getFeatures(key, v))
            features[key].append(f.T)
        features[key] = np.concatenate(features[key], axis=1)

    return features

# get the weights and biases from training data
def trainLDFs(means, covariance):

    hs = dict()
    bs = dict()

    # invert the covariance matrix
    inv = np.linalg.inv(covariance)

    # calculate the H and B of the LDF for each class
    for key, value in means.items():
        value = np.atleast_2d(value).T
        hs[key] = np.matmul(inv, value)
        inside = np.matmul(inv, value)
        bs[key] = (-0.5 * np.matmul(value.T, inside)).item()

    return hs, bs

# calculate g(x) for every class given a feature vector
def gOfx(matrix, hs, bs):

    # initialize dict for g(x)
    gX = dict()

    matrix = np.atleast_2d(matrix).T

    # calculate g(x) of each class
    for key, value in hs.items():
        gX[key] = (np.matmul(matrix.T, value) + bs[key]).item()

    # get highest g(x) value
    prediction = max(gX.items(), key=operator.itemgetter(1))[0]

    return prediction

# test given data with provided weights and biases
def testLDF(split, hs, bs, label=''):

    # initialize list for the true & predicted classes
    predicted = []
    true = []

    # calculate the prediction given the weights and biases
    for key, value in split.items():
        for image in value:
            featureVector = getFeatures(key, image)
            predicted.append(gOfx(featureVector, hs, bs))
            true.append(key)

    # get some post processing metrics
    accuracy = accuracy_score(true, predicted)
    confusion = confusion_matrix(true, predicted)
    report = classification_report(true, predicted, labels=list(split.keys()))

    # plot confusion matrix, saved to same directory
    plotConfusionMatrix(confusion, title= label + " Data Confusion Matrix", labels=split.keys())

    return accuracy, report

# perform LDF
def bayesianLDF(splitTrain, splitTest):

    # initialize mean dict and features numpy array
    means = dict()
    allFeatures = []

    # get features dict
    features = getFeaturesSet(splitTrain)
    features = collections.OrderedDict(sorted(features.items()))

    # get means and make one big matrix
    for key, value in features.items():
        means[key] = value.mean(axis=1)
        allFeatures.append(value)

    # make one big matrix
    allFeatures = np.concatenate(allFeatures, axis=1)

    # compute covariance of our matrix
    covariance = np.cov(allFeatures)

    # get weights and biases of training data
    hs, bs = trainLDFs(means, covariance)

    # test our training data
    trainAccuracy, trainReport = testLDF(splitTrain, hs, bs, label="Train")

    # test our test data
    testAccuracy, testReport = testLDF(splitTest, hs, bs, label="Test")

    return trainAccuracy, testAccuracy, trainReport, testReport

def main():

    # Problem 1

    # open image
    hcd1 = "input_hcd1.jpg"
    img = cv2.imread(hcd1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.float32(img)
    r = cv2.cornerHarris(img, 3, 3, 0.05)
    v = cv2.cornerEigenValsAndVecs(img, 2, 3)
    v = v[:,:,0:2].reshape(v.shape[0] * v.shape[1], 2)
    dictV = dict()

    # convert our vector into a dict
    for i in range(0, len(v)):
        dictV[i] = v[i]

    # predict corners, edges, and flats
    accuracy, report = findErrorRates(r, dictV)
    print("HCD1 Accuracy: ", accuracy)
    print("HCD1 Report: \n", report)
    print("-" * 300)

    # Problem 2

    # set random seed
    random.seed(258)

    # get all folders in the digit dataset
    folders = [folder for folder in os.listdir('DigitDataset') if not folder.endswith('.csv')]

    # initialize dictionaries
    splitTrain = dict()
    splitTest = dict()

    # split test and training sets
    for f in sorted(folders):
        # get the images of the class
        newDir = 'DigitDataset/' + f + "/"
        all = sorted(os.listdir(newDir))

        # take the first 50 images of the class
        train = all[0:50]

        # shuffle the rest of the images in the class
        rest = all[50:]
        random.shuffle(rest)

        # take the next 50 images in the shuffled class
        test = rest[0:50]
        splitTrain[f] = train
        splitTest[f] = test

    # perform bayesian LDF
    trainAccuracy, testAccuracy, trainReport, testReport \
        = bayesianLDF(splitTrain, splitTest)

    print("LDF Train Accuracy: ", trainAccuracy)
    print("LDF Train Report: \n", trainReport)
    print("LDF Test Accuracy: ", testAccuracy)
    print("LDF Test Report: \n", testReport)
    print("-" * 300)
    print("Done!")

if __name__ == "__main__":
    main()


