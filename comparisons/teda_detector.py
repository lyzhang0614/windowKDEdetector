from __future__ import division
from nab.detectors.base import AnomalyDetector



class TedaDetector(AnomalyDetector):

    def __init__(self, *args, **kwargs):
        super(TedaDetector, self).__init__(*args, **kwargs)

        self.k = 1
        self.mean_val = 0
        self.mean_val2 = 0
        self.variance_val = 0
        self.n = 3
        length = len(self.dataSet.data.value)
        self.threshold = (self.n**2+1)/(2*length)



    def handleRecord(self, inputData):
        inputValue = inputData["value"]
        finalScore = 0.0

        self.mean_val = self.update_mean_val(inputValue)
        self.mean_val2 = self.update_mean_val2(inputValue)
        self.variance_val = self.update_variance_val()

        if self.k >= 3:
            nor_ecc = self.cal_nor_ecc(inputValue)
            threshold = (self.n ** 2 + 1) / (2 * self.k)
            if nor_ecc > threshold:
                finalScore = 1.0

        self.k += 1

        return (finalScore,)

    def cal_nor_ecc(self, inVal):
        nor_ecc = 1/(2*self.k)+(inVal-self.mean_val)**2/(2*self.variance_val*self.k)
        return nor_ecc

    def update_mean_val(self, inVal):
        mean_val = (self.k-1)/self.k*self.mean_val+inVal/self.k
        return mean_val

    def update_mean_val2(self, inVal):
        mean_val2 = (self.k-1)/self.k*self.mean_val2+inVal**2/self.k
        return mean_val2

    def update_variance_val(self):
        variance_val = self.mean_val2-self.mean_val**2
        return variance_val



