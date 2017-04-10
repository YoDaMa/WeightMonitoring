import pandas as pd
from io import StringIO
import csv
import numpy as np


def mainProcessor(filename, suppressPrint = False):
    """
    Should take as input csv file generated from TekScan Sensormat. Will process the csv
    into a pandas dataframe where each row corresponds with one grid of data collected.
    When analyzing, the data can be simply reshaped to get the information out.
    To reshape, transpose the matrix then reshape.
    Reshaped matrix should be 44x52xN, where N is the number of data samples collected.
    :type filename: basestring
    :ivar: filename of csv
    :rtype: pandas DataFrame, Union[int,ANY], Union[int,ANY]
    """
    calib_pressure = np.nan
    calib_weight = np.nan

    s = StringIO()
    h = []
    with open(filename) as f:
        # if (~suppressPrint):
        #     print("FILENAME:", filename)
        reader = csv.reader(f, quoting=csv.QUOTE_NONE)
        header = True

        raw_sum = np.nan
        buf = []
        skip1 = False
        for row in reader:
            # Flag set to process the header of the data
            if header:
                ln = list(filter(None, row))
                if len(ln) == 0:
                    header = False
                else:
                    lnsplit = ln[0].split(' ')
                    # if (~suppressPrint):
                    #     print(lnsplit)
                    h.append(lnsplit)
                    if 'CALIBRATION_POINT' in lnsplit[0]:
                        calib_pressure = float(lnsplit[3])
                        calib_weight = float(lnsplit[1])
            # will be triggered once the header has been read
            else:
                # Remove all the blank cells from the row and convert to a list
                ln = list(filter(None, row))
                # This will be ignored because the skip flag is set to false.
                if skip1 == True:
                    # print(ln)
                    skip1 = False
                    continue
                # If empty line, then store values collected into the StreamIO
                if len(ln) == 0:
                    # print(len(np.ravel(buf)))
                    lnstr = ','.join(np.asarray(buf).ravel())
                    sstr = str(raw_sum) + ',' + lnstr + '\n'
                    s.write(sstr)
                    buf = []
                    raw_sum = np.nan
                    continue
                # If the line is a Frame description line, extract the raw sum from it.
                if 'Frame' in ln[0]:
                    # print(ln[0])
                    raw_sum = ln[-1]
                    continue
                # Line of data will have 52 entries (corresponding with sensorpad dimensions)
                if len(ln) == 52:
                    buf.append(ln)
                    continue

    s.seek(0)
    hdf = pd.DataFrame(h)
    df = pd.read_csv(s, sep=',', header=None)
    return df, calib_weight, calib_pressure, hdf


# mainProcessor('LeeH05_M.csv')
