import numpy as np
import scipy


class CDPR4:
    def __init__(self, params, approx=1):
        self.params = params # anchor points
        self.approx = approx # check folder /maths_behind_cdpr for approximations description
        
        