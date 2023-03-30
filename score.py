import math
import numpy as np
from dtaidistance import dtw


class Score(object):

    def percentage_score(self,score):   #To be replaced with a better scoring algorithm, if found in the future
        percentage =  100 - (score* 100)
        return int(percentage)

    def dtwdis(self,model_points,input_points,i,j):
        model_points = model_points.reshape(2*j,)
        input_points = input_points.reshape(2*i,)
        model_points = model_points/ np.linalg.norm(model_points)
        input_points = input_points/np.linalg.norm(input_points)
        return self.percentage_score(dtw.distance(model_points, input_points))

    def cosine_distance(self, A, B):
        A = A.reshape(2,)
        B = B.reshape(2,)
        cosine = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
        return self.percentage_score(math.sqrt(2 * (1 - cosine)))

    def normalize(self,input_test):
        for k in range(0,17):	
            input_test[:,k] = input_test[:,k]/np.linalg.norm(input_test[:,k])	
        return input_test

    def compare(self,ip,model,i,j):
        # ip = self.normalize(ip)
        scores = []
        for k in range(0,6):
            # scores.append(self.dtwdis(ip[:,k],model[:,k],i,j))
            scores.append(self.cosine_distance(ip[:,k], model[:,k]))
        return np.mean(scores),scores
