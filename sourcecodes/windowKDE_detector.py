import numpy
from nab.detectors.base import AnomalyDetector
from sklearn.neighbors import KernelDensity


def calDistance(v1, v2):  # L1-Distance
    return sum(map(lambda i, j: abs(i - j), v1, v2))


class WindowKDEDetector(AnomalyDetector):

    def __init__(self, *args, **kwargs):
        super(WindowKDEDetector, self).__init__(*args, **kwargs)

        self.big_windowData = []
        self.big_minVal = None
        self.big_maxVal = None
        self.h = None

        self.vectors = []
        self.distances = []
        self.p = 16
        self.labels = numpy.zeros(self.big_windowSize - 1)
        self.flag = 0
        self.label = 0

    def handleRecord(self, inputData):
        finalScore = 0.0
        inputValue = inputData["value"]

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
        std = numpy.std(self.big_windowData)
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
        targets = numpy.array(targets)
        targets = targets.reshape(-1, 1)

        # calculate the descriptor:
        set = numpy.array(set)
        set = set.reshape(-1, 1)
        kde = KernelDensity(bandwidth=self.h, kernel='gaussian').fit(set)
        v = kde.score_samples(targets)
        vector = numpy.exp(v)
        return vector

    def _getDistances(self):
        for i in range(0, len(self.vectors) - 1):
            d = calDistance(numpy.array(self.vectors[i]), numpy.array(self.vectors[i + 1]))
            self.distances.append(d)

    def _getDmax(self, vor_distances):
        max_d = max(vor_distances)
        labels2 = self.labels[len(self.labels) + 1 - self.big_windowSize:]
        deltas = []
        for i in range(0, len(vor_distances) - 1):
            t_delta = abs(vor_distances[i] - vor_distances[i + 1])
            deltas.append(t_delta)
        if 1 in labels2:
            max_d = max_d
        else:
            max_d = max_d + min(deltas)
        return max_d

    def _getLabel(self):
        finalScore = 0.0
        self.label = 0
        curr_d = self.distances[0]
        ex_d = self.distances[1]
        vor_distances = self.distances[1:]
        max_d = self._getDmax(vor_distances)
        avg_d = numpy.mean(vor_distances)
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
        self.labels = numpy.append(self.labels, self.label)
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
