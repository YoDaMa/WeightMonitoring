from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from PressureMat_PreProc import mainProcessor

"""
LeeH06_M2.csv - Lee Standing Data
LeeH05_M.csv - Lee Shifting Weight Data
PrioT01_M.csv  - Prio Standing Data
"""
# file = 'LeeH06_M2.csv'
# file = 'PrioT01_M.csv'
file = '../FMAT/MLeo/Task1/O02_M.csv'
df, calib_weight, calib_pressure, hdf = mainProcessor(file)
print(np.nan == calib_weight)
if (type(calib_pressure) != int):
    print("NAN ERROR!!!")
    calib_pressure = 0
    calib_weight = 0


resolution = (44, 52)
dfs = df.shape
print(dfs)
rowdata = np.asarray(df.iloc[:, 1:])
stddevdata = np.std(rowdata, axis=1)

slice = np.reshape(df.iloc[:, 1:].values.transpose(), (resolution[0], resolution[1], dfs[0]))

pressures = np.ravel(df.iloc[:, 0].values)
print(np.max(pressures))
print(calib_pressure)
press_max = int(np.max([np.max(pressures), calib_pressure]))
press_min = int(np.min([np.min(pressures), calib_pressure]))
press_max -= press_max % 100
press_min -= press_min % 100
print("PRESSURES")
print(pressures.shape)





tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)



font = {'size': 20}

mpl.rc('font', **font)




fig1 = plt.figure(figsize=(28, 17))
ax1 = fig1.add_subplot(111)
plt.title(file)
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
plt.ylim(press_min - 100, press_max + 100)
plt.xlim(0, dfs[0])
plt.yticks(range(press_min - 100, press_max + 100, ((press_max-press_min) // 20)),
           [str(x) + " KPa" for x in range(press_min - 100, press_max + 100, ((press_max-press_min) // 20))])
plt.xticks(np.linspace(100, dfs[0], 5), [str(x) + "s" for x in np.linspace(1, (dfs[0] + 1) / 100, 5)])

for y in range(press_min - 100, press_max + 100, ((press_max-press_min) // 20)):
    plt.plot(range(0, dfs[0]), [y] * len(range(0, dfs[0])), "--", lw=0.5, color="black", alpha=0.3)
plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="off", right="off", labelleft="on")

plt.plot(pressures, lw=2.5, color=tableau20[0])
y_pos = pressures[-1] - 0.5
plt.text(dfs[0] + 10, y_pos, 'Standing Pressures', color=tableau20[0])
baseline = np.ravel(np.zeros((1, dfs[0])) + calib_pressure)
print("BASELINE")
print(baseline.shape)
plt.plot(baseline, lw=2.5, color=tableau20[10])
y_pos = baseline[-1] - 0.5
plt.text(dfs[0] + 10, y_pos, 'Baseline Pressure', color=tableau20[10])
plt.text(dfs[0] // 2, press_max + 200,
         "Pressure measured in kPa over samples for subject of weight {}lbs".format(calib_weight)
         , fontsize=28, ha="center")
# plt.text(10,press_min - 300, "Baseline pressure line represents the pressure recorded during calibration"
#                    "\nof the pressure mat. Standing Pressures represents the pressures observed by "
#                    "\nsumming all pressure sensors on the mat at a frequency of 100Hz. ")


plt.savefig("stadingpressureplots.png", bbox_inches="tight")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()


plt.plot(stddevdata)
plt.show()


# plt.tick_params(axis="both", which="both", bottom="off", top="off",
#                 labelbottom="on", left="off", right="off", labelleft="on")
#
#
# # plt.savefig("footgraph.png", bbox_inches="tight")
#
# # ax.scatter3d(X,Y,slice, cmap=cm.coolwarm)
# im = ax.imshow(slice[:,:,1], interpolation='nearest', cmap=cm.coolwarm)
#
# plt.show()
