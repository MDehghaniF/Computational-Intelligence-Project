import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import anfis
import membership.mfDerivs
import membership.membershipfunction
import numpy

class FuzzyGasController:
    #todo
    # write all the fuzzify,inference,defuzzify method in this class
    def __init__(self):
        pass

    def decide(self,anf, center):
        inp = center
        if center > 200:
            inp = 200
        inputs = np.expand_dims(np.array([inp]),-1)
        return anfis.predict(anf,inputs)[0][0]