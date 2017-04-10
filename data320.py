import os
from sklearn import linear_model


def filelist(fs="../FMAT"):
    """
    Starting from the home folder, specify direction of FMAT folder.
    eg. "../FMAT/",
    :input: directory location
    :return: list of files
    """
    files = []
    for folder in [x for x in os.listdir(fs) if
                   os.path.isdir(os.path.abspath(os.path.join(fs, x)))]:
        lpath = os.path.abspath(os.path.join(fs, folder))
        # print(os.path.isdir(os.path.abspath(lpath)))
        # print(os.path.abspath(lpath))
        if lpath.endswith("NE"):
            continue
        x = os.listdir(lpath)
        # print(x)
        for folder in [x for x in os.listdir(lpath) if
                       os.path.isdir(os.path.abspath(os.path.join(lpath, x)))]:
            full = os.path.join(lpath, folder)
            vals = os.listdir(full)
            filtvals = [x for x in vals if x.endswith('M.csv')]
            # print(folder)
            for x in filtvals:
                final = os.path.join(full, x)
                files.append(final)

    return files


flist = filelist()

# print(files)


import pandas as pd
excelpath = os.path.abspath("../FMAT/FMAT Data Collection.xlsx")
df1 = pd.read_excel(excelpath)
nSubjects = 5
# Groundtruth mapping from subject name to weight values over 5 experiments
groundTruth = {}
for i in range(nSubjects): # Assuming 5 subjects
    print(i)
    weights = df1.iloc[i][5:10]
    subject = df1['Subject'][i]
    groundTruth[subject] = weights.values
    # groundTruth.append(np.concatenate(([subject], weights.values)))

# groundTruth = np.asarray(groundTruth)

# connections = mapping subject to filename
connections = {'O': 'LM', 'RA': 'AR', 'PJ': 'JP', 'SN': 'NSiddiqi'}

from PressureMat_PreProc import mainProcessor

# flist = ['../FMAT/MLeo/Task1/O02_M.csv','../FMAT/MLeo/Task2/O06_M.csv','../FMAT/MLeo/Task3/O10_M.csv']
dlist = []
for i in flist:
    dlist.append((os.path.basename(os.path.normpath(i)), (mainProcessor(i, True))))
    # append the output of MainProcessor and the filename


    # df, calib_weight, calib_pressure, hdf = dlist[0]

import numpy as np

# dataVector = np.empty((len(dlist),3))
dataVector = []
# d2 = []
i = 0
for j in range(len(dlist)):
    data = dlist[j]
    fname = data[0]
    df = data[1][0]
    vectors = df.iloc[:, 1:2289].transpose()  # (2288, # frames)
    # print(vectors.shape)
    # df.iloc[:,1] = groundVal
    frames = np.reshape(vectors.values, (44, 52, vectors.shape[1]))  # (44, 52, # frames)
    averageFrame = np.mean(vectors, axis=1)
    frameVal = averageFrame.sum()  # reduce the mean onenorm down to one val
    # print(connections)
    for beg in connections:
        if beg in fname:
            key = connections[beg]
            groundVal = groundTruth[key][i % 5]
            i += 1
    dataVector.append([fname, float(frameVal), float(groundVal)])

idx = 0

dataVector = pd.DataFrame(np.asarray(dataVector))

dataVector.astype(str)


fdata = dataVector[[1,2]].apply(pd.to_numeric).values
X = dataVector[1].apply(pd.to_numeric).values
y = np.transpose(dataVector[2].apply(pd.to_numeric).values.reshape(1, -1))

X = X.reshape((20, 1))
y = y.reshape(1, -1)
print(X.shape)
print(y.shape)
print(fdata.shape)
np.random.shuffle(fdata)
f_train = fdata[:15]
X_train = fdata[:15,0].reshape(15, 1)
Y_train = fdata[:15,1].reshape(15, 1)
X_test = fdata[15:, 0].reshape(5, 1)
Y_test = fdata[15:, 1].reshape(5, 1)
reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(X_train, Y_train)
reg.coef_
fdata.shape
f_train.shape
print(X[0])
print(X_train[0])
print(reg.intercept_)


from matplotlib import pyplot as plt
from sklearn import linear_model

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# reg = linear_model.LinearRegression()
# reg.fit(X,y)
# B = (1.0/(np.dot(X, X) * np.transpose(X)*y))
plt.scatter(X.reshape(-1,1), y.reshape(-1,1), 400, color='yellow')
plt.scatter(X_train.reshape(-1,1), Y_train.reshape(-1,1), 200, color='black')
plt.scatter(X_test.reshape(-1,1), Y_test.reshape(-1,1), 200, color='orange')
pred = np.transpose(reg.predict(X_test))
print(pred.shape)
plt.scatter(X_test, pred.transpose(), 140)
print("PREDICTION:", reg.predict(X))
plt.plot(X, reg.predict(X), linewidth=3)

# plt.plot(df.iloc[:, 0].ravel(), linewidth=1.2)
print('Coefficients: \n', reg.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((pred - Y_test) ** 2))
print('Variance score: %.2f' % reg.score(X_test, Y_test))
ax.legend()
plt.show()