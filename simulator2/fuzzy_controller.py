import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import anfis
import membership.mfDerivs
import membership.membershipfunction
import numpy

class FuzzyController:
    #todo
    # write all the fuzzify,inference,defuzzify method in this class
    def __init__(self):
        pass

    def decide(self,anf, left_dist,right_dist):
        inputs = np.expand_dims(np.hstack([right_dist,left_dist]),-1).T
        return anfis.predict(anf,inputs)[0][0]
    