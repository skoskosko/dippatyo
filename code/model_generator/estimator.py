
import numpy
from typing import List
import torch


class EstimatorBase():
    
    def estimate(self, groups: List[numpy.ndarray], disparity: torch.Tensor):
        raise NotImplementedError("This needs to be implenmented")
    
    def combine_estimators(self, groups: List[numpy.ndarray], estimated1: torch.Tensor, estimated2: torch.Tensor) -> torch.Tensor:
        output = numpy.copy(torch.Tensor.numpy(estimated1))
        for group in groups:
            Y, X = numpy.where(group==1)

            for i in range(len(Y)):
                color1 = estimated1[0, Y[i], X[i]]
                color2 = estimated2[0, Y[i], X[i]]
                
                color = numpy.rint(numpy.average([color1,color2]))
                output[:, Y[i], X[i]] = (color, color, color)

        return torch.from_numpy(output)
class HorisontalEstimator(EstimatorBase):
    pass

    def estimate(self, groups: List[numpy.ndarray], disparity: torch.Tensor, b: int= 20) -> torch.Tensor:
        output = numpy.copy(torch.Tensor.numpy(disparity))

        for group in groups:
            
            for y in range(group.shape[0]):
                x = numpy.where(group[y]==1)[0]

                if len(x):
                    start = numpy.average(disparity[:, y, x[0]-b:x[0]-1], axis=1)
                    end = numpy.average(disparity[:, y, x[-1]+1:x[-1]+b], axis=1)
                    
                    change = (end[0]-start[0]) / len(x)
                    for i, _x in enumerate(x):
                        color = start[0] + change * i

                        
                        if color < 0:
                            color = 0
                        elif color > 255:
                            color = 255
    
                        output[:, y, _x] = (numpy.rint(color), numpy.rint(color), numpy.rint(color))
        return torch.from_numpy(output)
                
class VerticalEstimator(EstimatorBase):
    pass

    def estimate(self, groups: List[numpy.ndarray], disparity: torch.Tensor, b: int= 20) -> torch.Tensor:
        output = numpy.copy(torch.Tensor.numpy(disparity))

        for group in groups:
            
            for x in range(group.shape[1]):
                y = numpy.where(group[:, x]==1)[0]
                if len(y):
                    start = numpy.average(disparity[:, y[0]-b:y[0]-1, x], axis=1)
                    end = numpy.average(disparity[:, y[-1]+1:y[-1]+b, x], axis=1)
                    
                    change = (end[0]-start[0]) / len(y)
                    for i, _y in enumerate(y):
                        color = start[0] + change * i

                        
                        if color < 0:
                            color = 0
                        elif color > 255:
                            color = 255
    
                        output[:, _y, x] = (numpy.rint(color), numpy.rint(color), numpy.rint(color))
        return torch.from_numpy(output)
    
