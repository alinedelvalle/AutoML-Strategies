import numpy as np

import matplotlib.pyplot as plt

from pymoo.indicators.hv import HV


class Point:
    
    def __init__(self, obj1, obj2, norm, meka, weka):
        self.obj1 = obj1
        self.obj2 = obj2
        self.norm = norm
        self.meka = meka
        self.weka = weka
        self.n = 0
        self.rank = 0
        self.S = []
        
        
    def dominate(self, pto):
        flag = False
        
        if (self.obj1 <= pto.obj1 and self.obj2 <= pto.obj2):
            if (self.obj1 < pto.obj1 or self.obj2 < pto.obj2):
                flag = True
                
        return flag
    
    
    def __str__(self):
        return str(self.obj1)+', '+str(self.obj2)
        

# fast non dominated sort        
class FNDS:
    
    def execute(self, list_points):        
        first_froint = []
        
        for i in range(len(list_points)):
            p = list_points[i]
            
            for j in range(len(list_points)):
                
                if i != j:
                    q = list_points[j]

                    if p.dominate(q):
                        p.S.append(q)
                    elif q.dominate(p):
                        p.n += 1

            if p.n == 0:
                p.rank = 1
                first_froint.append(p)
        
        return first_froint

        
    def plot_froint(self, list_points, list_pareto, title, xlabel, ylabel, file_name):
        x_all = []
        y_all = []
        for pto in list_points:
            x_all.append(pto.obj1)
            y_all.append(pto.obj2)
            
        x = []
        y = []
        for pto in list_pareto:
            x.append(pto.obj1)
            y.append(pto.obj2)
            
        plt.figure(figsize=(7, 5))
        plt.scatter(x_all, y_all, s=30, facecolors='none', edgecolors='gray')
        plt.scatter(x, y, s=30, facecolors='none', edgecolors='red')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(file_name)
        
    
# teste dominância
if __name__ == '__main__':
    a = Point(1, 5, True, 'a', 'b')
    b = Point(2, 3, True, 'c', 'd')
    c = Point(4, 1, True, 'e', 'f')
    d = Point(3, 4, False, 'g', 'h')
    e = Point(4, 3, False, 'i', 'j')
    f = Point(5, 5, False, 'l', 'm')
    
    list_points = [a, b, c, d, e, f]

    fnds = FNDS()
    froint = fnds.execute(list_points)
    
    fnds.plot_froint(list_points, froint, 'title', 'xlabel', 'ylabel', 'graph')
    
    for pto in froint:
        print(pto)
        
    F = np.array([[1,2,4,3,4,5], [5,3,1,4,3,5]]).transpose()    
    metric = HV(ref_point=np.array([5.0, 5.0]))
    hv = metric.do(F)
    print(F)
    print(hv)