import os
from PressureMat_PreProc import mainProcessor
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
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

excelpath = os.path.abspath("../FMAT/FMAT Data Collection.xlsx")
df1 = pd.read_excel(excelpath)
nSubjects = 5
# Groundtruth mapping from subject name to weight values over 5 experiments
groundTruth = {}
for i in range(nSubjects):  # Assuming 5 subjects
    print(i)
    weights = df1.iloc[i][5:10]
    subject = df1['Subject'][i]
    groundTruth[subject] = weights.values

connections = {'O': 'LM', 'RA': 'AR', 'PJ': 'JP', 'SN': 'NSiddiqi'}

dlist = []
for i in flist:
    dlist.append((os.path.basename(os.path.normpath(i)), (mainProcessor(i, True))))
    # append the output of MainProcessor and the filename


dataVector = []
# d2 = []
i = 0
for data in dlist:
    i = 0
    fname = data[0]
    print(fname)
    df = data[1][0]
    vectors = df.iloc[:, 1:2289].transpose()  # (2288, # frames)
    frames = np.reshape(vectors.values, (44, 52, vectors.shape[1]))  # (44, 52, # frames)
    averageFrame = np.mean(vectors, axis=1)
    frameVal = averageFrame.sum()  # reduce the mean onenorm down to one val
    # print
    i = 0
    for beg in connections:
        if beg in fname:
            key = connections[beg]
            # print(groundTruth[key])
            groundVal = groundTruth[key][i % 5]
            i += 1;
            df.iloc[:,0] = groundVal
    alpha = df
    alpha[2289] = fname
    dataVector.append(alpha)

len(dataVector)

mean_data = []
std_data = []
SNR_plot_vector = []
sel_val_vector = []
sel_idx_vector = []
X_data = []
y_data = []

stop = 100  # Use this value to specify how many sensors we want to look at.

for i in range(len(dataVector)):
    data = dataVector[i]
    fname1 = data.iloc[:, 2289][0]
    print(fname1)
    gtruth = data.iloc[:, 0][0]
    df_pressure = data.iloc[:, 1:2289]
    # print(df[2289])
    # print(df[2289])
    # mean_data.append(df)
    mr_df = np.mean(df_pressure, axis=0)  # (2288,)
    mean_data.append(mr_df)
    # print(df_pressure.shape)
    # print(mr_df.shape,)
    dev_df = np.square(df_pressure - mr_df)  # (# frames, 2288)
    # We add a very small number so that none of the standard devations are zero for SNR!
    stddev_df = np.sqrt(np.sum(dev_df, axis=0)) + 1e-8
    std_data.append(stddev_df)
    sort_a = sorted(enumerate(mr_df), key=lambda x: x[1], reverse=True)
    idx = [i[0] for i in sort_a if i[1] > 0]
    val = [i[1] for i in sort_a if i[1] > 0]

    SNR = np.divide(mr_df[idx], stddev_df[idx])  # (233,)

    SNR_sort = sorted(enumerate(SNR), key=lambda x: x[1], reverse=True)

    SNR_idx = [i[0] for i in SNR_sort[:stop]]
    SNR_val = [i[1] for i in SNR_sort[:stop]]

    # print(SNR_val[0])
    # print(np.shape(SNR_idx))
    # print(np.shape(idx))
    # sel_idx = idx[0]
    sel_idx = [idx[x] for x in SNR_idx]
    sel_val = [val[x] for x in SNR_idx]
    sel_idx_vector.append(sel_idx)
    sel_val_vector.append(sel_val)
    X_data.append(sel_val)
    y_data.append(gtruth)
    sel_SNR = np.zeros(2888, )
    for i in range(len(SNR_idx)):
        sel_SNR[sel_idx[i]] = SNR_val[i]
    # print(np.shape(sel_SNR))
    SNR_plot_vector.append(sel_SNR)





# fdata = dataVector[[1,2]].apply(pd.to_numeric).values
# X = dataVector[1].apply(pd.to_numeric).values.reshape((20, 1))
# y = np.transpose(dataVector[2].apply(pd.to_numeric).values.reshape(1, -1))
X = np.asarray(X_data)
y = np.asarray(y_data).reshape(-1,1)
# fdata = np.asarray([X,y]).transpose()
indicies = np.asarray(range(20))
np.random.shuffle(indicies)
X = X[indicies]
y = y[indicies]
X_train = X[:15, :]
y_train = y[:15, :]
X_test = X[15:, :]
y_test = y[15:, :]

reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(X_train, y_train)
# reg.coef_
# # fdata.shape
# # f_train.shape
# # print(X[0])
# # print(X_train[0])
# print(reg.intercept_)
print(np.shape(y_train))
np.shape(X_train)

pred = np.transpose(reg.predict(X_test))
print("Shape of Coefficients: {} \n".format(np.shape(reg.coef_)))
print('Coefficients: \n', reg.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((pred - y_test) ** 2))
print('Variance score: %.2f' % reg.score(X_test, y_test))

font = {'size': 20}

mpl.rc('font', **font)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
plt.title(' Most important pixels based on SNR: Top {}'.format(stop))
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="off", right="off", labelleft="on")
plt.xlabel('Ground Truth Data (kg)')
plt.ylabel('Predicted Data (kg)')

yyy = y_test.reshape(-1, 1)
print(yyy)
print(pred.reshape(-1, 1))
plt.scatter(yyy, pred.reshape(-1, 1), 300)

# plt.scatter(X.reshape(-1,1), y.reshape(-1,1), 200, color='yellow')
# plt.scatter(X_train.reshape(-1,1), Y_train.reshape(-1,1), color='black')
# plt.scatter(X_test.reshape(-1,1), Y_test.reshape(-1,1), color='orange')
# pred = np.transpose(reg.predict(X_test))
print(pred.reshape(-1, 1).shape)
# plt.scatter(X_test, pred.transpose(), 140)
# plt.plot(X, reg.predict(X), linewidth=3)

plt.show()