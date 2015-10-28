import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class Attractor(object):
    def __init__(self,s=10,p=8/3,b=28,start=0.0,end=80.0,points=10000):
        self.params=np.array([s,p,b])
        self.start=start
        self.end=end
        self.points=int(points)
        self.dt=(self.end-self.start)/self.points
        self.k1=np.zeros(3)
        self.k2=np.zeros(3)
        self.k4=np.zeros(3)
        self.x=np.zeros(self.points)
        self.y=np.zeros(self.points)
        self.z=np.zeros(self.points)
        self.t=np.linspace(self.start,self.end,self.points,endpoint=True)
        self.solution=pd.DataFrame([])
           def euler(self,x,y,z):
        r=[x,y,z]
        self.k1[0]=self.params[0]*(r[1] - r[0])
        self.k1[1]=r[0]*(self.params[1] - r[2]) - r[1]
        self.k1[2]=r[0]*r[1]-self.params[2]*r[2]
        return self.k1
    
    def rk2(self,x,y,z):
        self.euler(x,y,z)
        r=[x,y,z]
        r1=np.array([r[0]+self.k1[0]*self.dt/2,r[1]+self.k1[1]*self.dt/2,r[2]+self.k1[2]*self.dt/2])
        self.k2[0]=self.params[0]*(r1[1] - r1[0])
        self.k2[1]=r1[0]*(self.params[1] - r1[2]) - r1[1]
        self.k2[2]=r1[0]*r1[1]-self.params[2]*r1[2]
        return self.k2
    def rk4(self,x,y,z):
        self.rk2(x,y,z)
        r=[x,y,z]
        r2=np.array([r[0]+self.k2[0]*self.dt/2,r[1]+self.k2[1]*self.dt/2,r[2]+self.k2[2]*self.dt/2])
        k3=np.zeros(3)
        k3[0]=self.params[0]*(r2[1] - r2[0])
        k3[1]=r2[0]*(self.params[1] - r2[2]) - r2[1]
        k3[2]=r2[0]*r2[1]-self.params[2]*r2[2]
        r3=np.array([r[0]+k3[0]*self.dt,r[1]+k3[1]*self.dt,r[2]+k3[2]*self.dt])
        self.k4[0]=self.params[0]*(r3[1] - r3[0])
        self.k4[1]=r3[0]*(self.params[1] - r3[2]) - r3[1]
        self.k4[2]=r3[0]*r3[1]-self.params[2]*r3[2]
        return self.k4
    def evolve(self,x0=0.1,y0=0.0,z0=0.0,order=4):
        self.x[0]=x0
        self.y[0]=y0
        self.z[0]=z0
        i=1
        if order==1:
            self.euler(x0,y0,z0)
            while i< self.points:
                self.euler(self.x[i-1],self.y[i-1],self.z[i-1])
                self.x[i]=self.x[i-1]+self.k1[0]*self.dt
                self.y[i]=self.y[i-1]+self.k1[1]*self.dt
                self.z[i]=self.z[i-1]+self.k1[2]*self.dt
                i+=1
        elif order==2:
            while i< self.points:
                self.rk2(self.x[i-1],self.y[i-1],self.z[i-1])
                self.x[i]=self.x[i-1]+2*self.k2[0]*self.dt
                self.y[i]=self.y[i-1]+2*self.k2[1]*self.dt
                self.z[i]=self.z[i-1]+2*self.k2[2]*self.dt
                i+=1
        elif order==4:
            while i< self.points:
                self.rk4(self.x[i-1],self.y[i-1],self.z[i-1])
                self.x[i]=self.x[i-1]+self.k4[0]*self.dt
                self.y[i]=self.y[i-1]+self.k4[1]*self.dt
                self.z[i]=self.z[i-1]+self.k4[2]*self.dt
                i+=1
        sol=pd.DataFrame(data=[self.t,self.x,self.y,self.z],index=['t', 'x', 'y', 'z'])
        self.solution=pd.DataFrame.transpose(sol)
        return self.solution
    def save(self):
        self.solution.to_csv('save_solution.csv')
    def plotx(self):
        plt.plot(self.solution['t'],self.solution['x'])
        plt.show()
    def ploty(self):
        plt.plot(self.solution['t'],self.solution['y'])
        plt.show()
    def plotz(self):
        plt.plot(self.solution['t'],self.solution['z'])
        plt.show()
    def plotxy(self):
        plt.plot(self.solution['x'],self.solution['y'])
        plt.show()
    def plotyz(self):
        plt.plot(self.solution['y'],self.solution['z'])
        plt.show()
    def plotzx(self):
        plt.plot(self.solution['z'],self.solution['x'])
        plt.show()
    def plot3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.solution['x'], self.solution['y'], self.solution['z'])
        plt.show()