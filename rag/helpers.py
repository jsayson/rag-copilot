import numpy as np
import matplotlib.pyplot
import math

class Value:
    def __init__(self, data, ope=""):
        self.data = data
        self.ope = ope

    def __repr__(self):
        return f"the {self.ope} result: {self.data}"
    
    def __add__(self, other):
        return Value(self.data + other.data)
    
    def __sub__(self, other):
        return Value(self.data - other.data)
    
    def __mul__(self, other):
        return Value(self.data * other.data)
    
    def __truediv__(self, other):
        return Value(self.data / other.data)
    
    def dot(self, other):
        assert len(self.data) == len(other.data)
        ini = Value(0)
        for a, b in zip(self, other):
            ini += a * b

        ini.ope = "dot product"
        return ini

    def sqrt(self):
        return Value(math.sqrt(self.data))
    
    def cosine_similarity(self, other):
        dp = Value.dot(self, other)
        res = dp / (((Value.dot(self, self)).sqrt()) * ((Value.dot(other, other)).sqrt()))
        res.ope = "cosine similarity"
        return res 
    

#   [i have a pen] , [i have an apple], [pen pineapple apple pen]
#   [i have a pen an apple pineapple]
if __name__ == "__main__":
    a = np.array([[Value(1), Value(1), Value(1), Value(1), Value(0), Value(0), Value(0)]]).flatten()
    b = np.array([[Value(1), Value(1), Value(0), Value(0), Value(1), Value(1), Value(0)]]).flatten()
    c = np.array([[Value(0), Value(0), Value(0), Value(2), Value(0), Value(1), Value(1)]]).flatten()

    print(f"{Value.cosine_similarity(a, b)}")
    print(f"{Value.cosine_similarity(c, b)}")
    print(f"{Value.cosine_similarity(a, c)}")