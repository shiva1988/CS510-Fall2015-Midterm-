from attractor import Attractor
import random
import numpy as np
from nose import with_setup
import os


class TestRandomValues:
    """Define an Attractor with random values for s, p and b, test interface"""
    def setup(self):
        """Setup fixture is run before every test method separately"""
        s=random.uniform(1,20)
        p=random.uniform(1,10)
        b=random.uniform(1,15)
        self.s=s
        self.p=p
        self.b=b
        start=0
        end=random.uniform(10,50)
        points=random.randint(50,200)
        self.start=start
        self.end=end
        self.points=points
        self.a=Attractor(s,p,b,start,end,points)
    def test_defintions(self):
        """Test the first definitions of the class"""
        assert self.a.params[0]==self.s, "\n we want to test that value of s save correctly,it gives us error if s!=params[0] \n"
        assert self.a.params[1]==self.p, "\n we want to test that value of p save correctly,it gives us error if p!=params[1] \n"
        assert self.a.params[2]==self.b, "\n we want to test that value of b save correctly,it gives us error if b!=params[2] \n"
        assert self.a.dt==(self.end-self.start)/self.points, "\n we want to test the value of s save correctly,it gives us error if dt!=(end-start)/points \n"
    def test_solve(self):
        """Test if solve is printing in the CSV file"""
        os.remove('save_solution.csv')
        self.a.save()
        data=open('save_solution.csv','r')
        d=data.read()
        assert len(d)>0
    def test_euler(self):
        """Test if the methods euler is generating the correct answer and we can do this for rk2 and rk4"""
        self.a=Attractor(5,1,3,0,80,100)
        assert self.a.euler(0,1,0)[0]==5, "\n we want to test that value of euler[0] save correctly,it gives us error if euler[0]!=5 \n"
        assert self.a.euler(0,1,0)[1]==-1, "\n we want to test that value of euler[0] save correctly,it gives us error if euler[1]!=-1 \n"
        assert self.a.euler(0,1,0)[2]==0, "\n we want to test that value of euler[0] save correctly,it gives us error if euler[2]!=0 \n"
        
