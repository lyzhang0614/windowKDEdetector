from __future__ import division
from nab.detectors.base import AnomalyDetector




class RdeDetector(AnomalyDetector):

    def __init__(self, *args, **kwargs):
        super(RdeDetector, self).__init__(*args, **kwargs)

        self.k = 1
        self.ks = 1
        self.list = [1] # 0: <, 1: >=
        self.status = 'normal'
        self.finalScore = 0.0



    def handleRecord(self, inputData):
        inputValue = inputData["value"]
        finalScore = 0.0

        if self.k == 1:
            self.density = 1.0
            self.mean_dst = self.density
            self.mean_val = inputValue
            self.quantity = inputValue**2
        else:
            self.mean_val = self.update_mean_val(inputValue)
            self.quantity = self.update_quantity(inputValue)
            pre_dst = self.density
            self.density = self.update_dst(inputValue)
            self.delta_dst = abs(pre_dst - self.density)
            self.mean_dst = self.update_mean_dst()

            if self.density<self.mean_dst:
                self.list.append(0)
            else:
                self.list.append(1)

            if self.status == 'normal':
                x = set(self.list[-20:])
                if len(x)==1 and 0 in x:
                    self.status = 'fault'
                    self.finalScore = 1.0
                    finalScore = 1.0
                    self.ks = 0
            else:
                x = set(self.list[-80:])
                if len(x)==1 and 1 in x:
                    self.status = 'normal'
                    self.finalScore = 0.0
                    self.ks = 0
        self.k += 1
        self.ks += 1
        return (self.finalScore,)

    def update_mean_val(self, inVal):
        mean_val = (self.k-1)/self.k*self.mean_val+inVal/self.k
        return mean_val

    def update_quantity(self, inVal):
        quantity = (self.k-1)/self.k*self.quantity+inVal**2/self.k
        return quantity

    def update_dst(self, inVal):
        density = 1/(1+(inVal-self.mean_val)**2+self.quantity-self.mean_val**2)
        return density

    def update_mean_dst(self):
        mean_dst = ((self.ks-1)/self.ks*self.mean_dst+self.density/self.ks)*(1-self.delta_dst) + \
                   self.density*self.delta_dst
        return mean_dst



