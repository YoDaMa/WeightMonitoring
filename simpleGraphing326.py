import pandas as pd
import os
from PressureMat_PreProc import mainProcessor
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn import linear_model
import numpy as np


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
            # print(groundTruth)
            groundVal = groundTruth[key][i % 5]
            i += 1
    dataVector.append([fname, float(frameVal), float(groundVal)])


data = dlist[0]
fname = data[0]
print("FILENAME: ", fname)
df = data[1][0]
pressure_df = df.iloc[:, 1:]  # (# frames, 2288)
mr_df = np.mean(pressure_df, axis=0) # (2288,)
# print(pressure_df)
# print(mr_df)
dev_df = np.square(pressure_df - mr_df) # (# frames, 2288)
# We add a very small number so that none of the standard devations are zero for SNR!
stddev_df = np.sqrt(np.sum(dev_df, axis=0)) + 1e-8
print(np.min(stddev_df))
stddev_df.shape

sort_a = sorted(enumerate(mr_df), key=lambda x: x[1], reverse=True)
stop = 100 # Use this value to specify how many sensors we want to look at.
idx = [i[0] for i in sort_a if i[1] > 0]
val = [i[1] for i in sort_a if i[1] > 0]

SNR = np.divide(mr_df[idx], stddev_df[idx]) # (233,)

SNR_sort = sorted(enumerate(SNR), key=lambda x: x[1], reverse=True)

SNR_idx = [i[0] for i in SNR_sort[:stop]]
SNR_val = [i[1] for i in SNR_sort[:stop]]

# print(SNR_val[0])
print(np.shape(SNR_idx))
print(np.shape(idx))
sel_idx = idx[0]
sel_idx = [idx[x] for x in SNR_idx]
sel_SNR = np.zeros(2888,)
for i in range(len(SNR_idx)):
    sel_SNR[sel_idx[i]] = SNR_val[i]
print(np.shape(sel_SNR))

np.shape(val)

# Threshould the mean Frame so that values that aren't in the top 20 are zeroed
resolution = (44, 52)
threshframe = np.zeros((2288,))
threshframe[idx] = 1
# threshframe =   threshframe
threshframe = np.reshape(threshframe, (resolution[0], resolution[1]))

font = {'size': 20}

mpl.rc('font', **font)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
# plt.title(file + ' Most important pixels based on MRC')

ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
# plt.ylim(0, 44)
# plt.xlim(0, 52)
plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="off", right="off", labelleft="on")
im = ax.imshow(threshframe, interpolation='nearest', cmap=cm.viridis, alpha=0.8)
cbar = fig.colorbar(im)
cbar.ax.get_yaxis().labelpad = 50
cbar.ax.set_ylabel('Just for Visualization', rotation=270)
plt.show()


# Threshould the mean Frame so that values that aren't in the top 20 are zeroed
resolution = (44, 52)
threshframe = np.zeros((2288,))
threshframe[sel_idx] = sel_SNR[sel_idx]
# threshframe =   threshframe
threshframe = np.reshape(threshframe, (resolution[0], resolution[1]))

font = {'size': 20}

mpl.rc('font', **font)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
# plt.title(file + ' Most important pixels based on MRC')

ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
# plt.ylim(0, 44)
# plt.xlim(0, 52)
plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="off", right="off", labelleft="on")
im = ax.imshow(threshframe, interpolation='nearest', cmap=cm.viridis, alpha=0.8)
cbar = fig.colorbar(im)
cbar.ax.get_yaxis().labelpad = 50
cbar.ax.set_ylabel('Just for Visualization', rotation=270)
plt.show()