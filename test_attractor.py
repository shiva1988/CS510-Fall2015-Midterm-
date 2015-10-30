import os
import numpy as np
from attractor import Attractor
class TestRandomC:
    def test_dt_value():
        a = Attractor()
        assert a.end == a.dt * a.points
    def test_defintions(self):  
        assert self.a.params[0]==self.s
        assert self.a.params[1]==self.p
        assert self.a.params[2]==self.b
        assert self.a.dt==(self.end-self.start)/self.points
   
    def test_euler_shape():
        a = Attractor()
        xyz = np.array([0.0, 0.0, 0.0])
        assert a.euler(xyz).shape == (3, )
    def test_rk2_shape():
        a = Attractor()
        xyz = np.array([0.0, 0.0, 0.0])
        assert a.rk2(xyz).shape == (3, )
    def test_rk4_shape():
        a = Attractor()
        xyz = np.array([0.0, 0.0, 0.0])
        assert a.rk4(xyz).shape == (3,)
    def test_evolve_shape():
        a = Attractor()
        assert a.evolve().shape == (a.points, 3)
    def test_solve(self):
        os.remove('save_solution.csv')
        self.a.save()
        data=open('save_solution.csv','r')
        d=data.read()
        assert len(d)>0
    