
import numpy
from typing import List
import torch


class EstimatorBase():
    
    def estimate(self, groups: List[numpy.ndarray], disparity: torch.Tensor):
        raise NotImplementedError("This needs to be implenmented")

class HorisontalEstimator(EstimatorBase):
    pass

    def estimate(self, groups: List[numpy.ndarray], disparity: torch.Tensor) -> torch.Tensor:
        output = numpy.copy(torch.Tensor.numpy(disparity))

        for group in groups:
            
            for y in range(group.shape[0]):
                x = numpy.where(group[y]==1)[0]

                if len(x):
                    start = numpy.average(disparity[:, y, x[0]-20:x[0]-1], axis=1)
                    end = numpy.average(disparity[:, y, x[-1]+1:x[-1]+20], axis=1)
                    
                    change = (end[0]-start[0]) / len(x)
                    for i, _x in enumerate(x):
                        color = start[0] + change * i

                        
                        if color < 0:
                            color = 0
                        elif color > 255:
                            color = 255
    
                        output[:, y, _x] = (numpy.rint(color), numpy.rint(color), numpy.rint(color))
        return torch.from_numpy(output)
                
