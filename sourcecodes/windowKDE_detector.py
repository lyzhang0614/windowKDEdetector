"""
import numpy as np
from nab.detectors.base import AnomalyDetector
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
import math
import time

def calDistance(v1, v2):  # L1-Distance
    return sum(map(lambda i, j: abs(i - j), v1, v2))


class WindowKDEDetector(AnomalyDetector):

    def __init__(self, *args, **kwargs):
        super(WindowKDEDetector, self).__init__(*args, **kwargs)

        self.big_windowData = []
        self.big_Data = []
        self.big_windowTime = []
        self.big_minVal = None
        self.big_maxVal = None
        self.h = None

        self.vectors = []
        self.distances = []
        self.p = 16
        self.labels = np.zeros(self.big_windowSize - 1)
        self.flag = 0
        self.label = 0
        # self.x = 0
        # self.y = 0
        # self.count = 0
        # self.f2 = False

        self.k = int(math.sqrt(self.small_windowSize))

    def handleRecord(self, inputData):
        finalScore = 0.0
        inputValue = inputData["value"]
        # self.count += 1
        self.big_Data.append(inputValue)

        # get the range of the main sliding window:
        if self.big_maxVal is None or inputValue > self.big_maxVal:
            self.big_maxVal = inputValue
        if self.big_minVal is None or inputValue < self.big_minVal:
            self.big_minVal = inputValue

        if len(self.big_windowData) < self.big_windowSize:
            self.big_windowData.append(inputValue)
            if len(self.big_windowData) == self.big_windowSize:  # initialize the main sliding window:
                finalScore = self._detect()
                self._reset()
        else:
            self.big_windowData.pop(0)
            self.big_windowData.append(inputValue)
            # xs = range(0, 500)
            # ys = self.big_Data[len(self.big_Data) - 500:]
            # if (self.count - self.x == 369) and self.f2:
            #     plt.scatter(130, self.y, marker='o', edgecolors='k', linewidths=2.5, c='w', s=200)
            #     plt.scatter(133, 11206, marker='^', edgecolors='k', linewidths=2.5, c='w', s=200)
            #     plt.legend(['Our algorithm', 'HTM'], loc='best', fontsize=18)
            #     color = []
            #     for i in range(len(ys)):
            #         if i>=53 and i<=239:
            #             color.append('#FF0000')
            #         else:
            #             color.append('#0000FF')
            #     points = np.array([xs, ys]).T.reshape(-1, 1, 2)
            #     segments = np.concatenate([points[:-1], points[1:]], axis=1)
            #     lc = LineCollection(segments, color=color)
            #     ax = plt.axes()
            #     ax.set_xlim(min(xs), max(xs))
            #     ax.set_ylim(min(ys), max(ys))
            #     ax.add_collection(lc)
            #     plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(8000))

                # plt.xlabel('$t$', fontsize=20)
                # plt.ylabel('$x_{t}$', fontsize=20)
                # plt.xticks(fontsize=18)
                # plt.yticks(fontsize=18)
                # plt.show()
                # self.f2 = False
            # s = time.time()
            finalScore = self._detect()
            # e = time.time()
            # print(e-s)
            # if self.label:
            #     self.x = self.count
                # print(self.count)
                # self.y = inputValue
                # if self.count == 3096:  # 2922 3097 3318   4370  AAPL
                #     self.f2 = True
            self._reset()

        return (finalScore,)

    def _getH(self, subwindow=None):
        #main window:
        std = np.std(self.big_windowData, ddof=1)
        if std == 0.0:
            std = 0.000001
        self.h = (4 / (3 * self.big_windowSize)) ** (1 / 5) * std

        #sub-window: fixed
        # std = np.std(subwindow, ddof=1)
        # if std == 0.0:
        #     std = 0.000001
        # self.h = (4 / (3 * self.small_windowSize)) ** (1 / 5) * std

    def _getVectors(self):
        m = (self.big_windowSize - self.small_windowSize) / self.small_windowSize + 1
        for i in range(1, m + 1):
            sub_window = list(
                self.big_windowData[
                self.big_windowSize - (i - 1) * self.small_windowSize - self.small_windowSize:self.big_windowSize - (
                        i - 1) * self.small_windowSize])
            v = self._calVector(sub_window)
            self.vectors.append(v)

    def _calVector(self, set):
        # get the target set T:
        targets = []
        interval = (self.big_maxVal - self.big_minVal) / self.p
        for i in range(0, self.p):
            targets.append(self.big_minVal + (2 * i + 1) / 2 * interval)
        targets = np.array(targets)
        targets = targets.reshape(-1, 1)

        # calculate the descriptor:
        '''def get_h(i,set):
            # sub-window: variable
            x_list =set[:]
            xi = x_list[i]
            del x_list[i]
            distances=[]
            for x in x_list:
                distances.append(abs(xi-x))
            distances = sorted(distances)
            return distances[self.k]

        vector = []
        for yj in targets:
            val = 0
            for i,xi in enumerate(set):
                h = get_h(i,set)
                fenmu = math.sqrt(2*math.pi)*self.small_windowSize*h
                fenzi = math.exp(-( (yj[0]-xi)/h )**2 /2)
                val += fenzi/fenmu
            vector.append(val)'''
        # self._getH(set)
        set = np.array(set)
        set = set.reshape(-1, 1)
        kde = KernelDensity(bandwidth=self.h, kernel='gaussian').fit(set)
        v = kde.score_samples(targets)
        vector = np.exp(v)
        return vector

    def _getDistances(self):
        for i in range(0, len(self.vectors) - 1):
            d = calDistance(np.array(self.vectors[i]), np.array(self.vectors[i + 1]))
            self.distances.append(d)

    def _getDmax(self, vor_distances):
        max_d = max(vor_distances)
        deltas = []
        for i in range(0, len(vor_distances) - 1):
            t_delta = abs(vor_distances[i] - vor_distances[i + 1])
            deltas.append(t_delta)
        max_d = max_d + min(deltas)
        return max_d

    def _getLabel(self):
        finalScore = 0.0
        self.label = 0
        curr_d = self.distances[0]
        ex_d = self.distances[1]
        vor_distances = self.distances[1:]
        max_d = self._getDmax(vor_distances)
        avg_d = np.mean(vor_distances)
        # get label:
        if self.flag == 0:
            if curr_d > max_d and ex_d <= avg_d:
                finalScore = 1.0
                self.label = 1
        # get s:
        if curr_d > max_d and ex_d <= avg_d:
            self.flag = 1
        else:
            self.flag = 0
        self.labels = np.append(self.labels, self.label)
        return finalScore

    def _detect(self):
        self._getH()
        self._getVectors()
        self._getDistances()
        finalScore = self._getLabel()
        return finalScore

    def _reset(self):
        self.vectors = []
        self.distances = []
"""


import numpy as np
from nab.detectors.base import AnomalyDetector
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
import math
import time

def calDistance(v1, v2):  # L1-Distance
    return sum(map(lambda i, j: abs(i - j), v1, v2))


class WindowKDEDetector(AnomalyDetector):

    def __init__(self, *args, **kwargs):
        super(WindowKDEDetector, self).__init__(*args, **kwargs)

        self.big_windowData = []
        self.big_Data = []
        self.big_windowTime = []
        self.big_minVal = None
        self.big_maxVal = None
        self.h = None

        self.vectors = []
        self.distances = []
        self.p = 16
        self.labels = np.zeros(self.big_windowSize - 1)
        self.flag = 0
        self.label = 0

        self.k = int(math.sqrt(self.small_windowSize))

    def handleRecord(self, inputData):
        finalScore = 0.0
        inputValue = inputData["value"]
        self.big_Data.append(inputValue)

        # get the range of the main sliding window:
        if self.big_maxVal is None or inputValue > self.big_maxVal:
            self.big_maxVal = inputValue
        if self.big_minVal is None or inputValue < self.big_minVal:
            self.big_minVal = inputValue

        if len(self.big_windowData) < self.big_windowSize:
            self.big_windowData.append(inputValue)
            if len(self.big_windowData) == self.big_windowSize:  # initialize the main sliding window:
                finalScore = self._detect()
                self._reset()
        else:
            self.big_windowData.pop(0)
            self.big_windowData.append(inputValue)
            finalScore = self._detect()
            self._reset()

        return (finalScore,)

    def _getH(self):
        # main window:
        std = np.std(self.big_windowData)
        if std == 0.0:
            std = 0.000001
        self.h = (4 / (3 * self.big_windowSize)) ** (1 / 5) * std

    def _getVectors(self):
        m = (self.big_windowSize - self.small_windowSize) / self.small_windowSize + 1
        for i in range(1, m + 1):
            sub_window = list(
                self.big_windowData[
                self.big_windowSize - (i - 1) * self.small_windowSize - self.small_windowSize:self.big_windowSize - (
                        i - 1) * self.small_windowSize])
            v = self._calVector(sub_window)
            self.vectors.append(v)

    def _calVector(self, set):
        # get the target set T:
        targets = []
        interval = (self.big_maxVal - self.big_minVal) / self.p
        for i in range(0, self.p):
            targets.append(self.big_minVal + (2 * i + 1) / 2 * interval)
        targets = np.array(targets)
        targets = targets.reshape(-1, 1)

        # calculate the descriptor:
        set = np.array(set)
        set = set.reshape(-1, 1)
        kde = KernelDensity(bandwidth=self.h, kernel='gaussian').fit(set)
        v = kde.score_samples(targets)
        vector = np.exp(v)
        return vector

    def _getDistances(self):
        for i in range(0, len(self.vectors) - 1):
            d = calDistance(np.array(self.vectors[i]), np.array(self.vectors[i + 1]))
            self.distances.append(d)

    def _getDmax(self, vor_distances):
        max_d = max(vor_distances)
        deltas = []
        for i in range(0, len(vor_distances) - 1):
            t_delta = abs(vor_distances[i] - vor_distances[i + 1])
            deltas.append(t_delta)
        max_d = max_d + min(deltas)
        return max_d

    def _getLabel(self):
        finalScore = 0.0
        self.label = 0
        curr_d = self.distances[0]
        ex_d = self.distances[1]
        vor_distances = self.distances[1:]
        max_d = self._getDmax(vor_distances)
        avg_d = np.mean(vor_distances)
        # get label:
        if self.flag == 0:
            if curr_d > max_d and ex_d <= avg_d:
                finalScore = 1.0
                self.label = 1
        # get s:
        if curr_d > max_d and ex_d <= avg_d:
            self.flag = 1
        else:
            self.flag = 0
        self.labels = np.append(self.labels, self.label)
        return finalScore

    def _detect(self):
        self._getH()
        self._getVectors()
        self._getDistances()
        finalScore = self._getLabel()
        return finalScore

    def _reset(self):
        self.vectors = []
        self.distances = []
