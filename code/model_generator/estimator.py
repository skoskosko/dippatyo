
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
    
class FocalEstimator(EstimatorBase):
    pass

    def get_line_points(self, x1, y1, x2, y2, x_limit = (0, 1024), y_limit = (0, 512)):
        points_x = [x1]
        points_y = [y1]

        x_m = x2-x1
        y_m = y2-y1

        x = x1
        y = y1
        steps = abs(x_m)+abs(y_m)
        x_s = x_m / steps
        y_s = y_m / steps
        for _i in range(steps):

            x += x_s
            y += y_s

            if x_s > 0:
                if int(x) >= x_limit[1]:
                    break
            elif x_s < 0:
                if int(x) <= x_limit[0]:
                    break
            
            if y_s > 0:
                if int(y) >= y_limit[1]:
                    break
            elif y_s < 0:
                if int(y) <= y_limit[0]:
                    break
            
            points_x.append(int(x))
            points_y.append(int(y))

        assert max(points_x) < x_limit[1], f"{max(points_x)} < {x_limit[1]}"
        assert max(points_y) < y_limit[1], f"{max(points_y)} < {y_limit[1]}"

        return numpy.array(points_x), numpy.array(points_y)

    
    def sort_by_distance(self, line_points, point):
        # Stack the x and y coordinates to form an array of shape (n, 2)
        x = line_points[0]
        y = line_points[1]
        coords = numpy.vstack((x, y)).T
        
        # Calculate Euclidean distance for each coordinate from the given point
        distances = numpy.sqrt((coords[:, 0] - point[0])**2 + (coords[:, 1] - point[1])**2)
        
        # Sort the indices by distance in descending order (furthest to closest)
        sorted_indices = numpy.argsort(distances)[::-1]
        
        # Return the sorted x and y coordinates separately
        sorted_x = x[sorted_indices]
        sorted_y = y[sorted_indices]
        
        return sorted_x, sorted_y

    def estimate(self, groups: List[numpy.ndarray], disparity: torch.Tensor) -> torch.Tensor:
        
        # Set focal pojnt
        y = int(disparity.shape[1]/2.5)
        x = int(disparity.shape[2]/2)


        output = numpy.copy(torch.Tensor.numpy(disparity))

        def frame_estimator(_y):
            # 200 value at y
            # 0 value at disparity.shape[1]

            frame_length = disparity.shape[1] - y/2

            _y = _y - y/2

            change = 180 / frame_length


            color = (_y * change)

            if color < 0:
                color = 0
            elif color > 200:
                raise Exception("IMPOSSIBLE!!!")

            return [int(color), int(color), int(color)]

        def set_horisontal_colors(line_points):
            x_point = line_points[0][0] - 5
            if x_point < 0: x_point = 0
            start_color = numpy.average(disparity[:, line_points[1][0], x_point:x_point+5], axis=1)
            if start_color[0] < 2:
                start_color = frame_estimator(line_points[1][0])
            end_color = numpy.average(disparity[:, line_points[1][-1], line_points[0][-1]:line_points[0][-1]+2], axis=1)
            change = (end_color[0]-start_color[0]) / len(line_points[0])
            for i in range(len(line_points[0])):
                color = start_color[0] + change * i
                if color < 0:
                    color = 0
                elif color > 255:
                    color = 255
                output[:, line_points[1][i], line_points[0][i]] = (numpy.rint(color), numpy.rint(color), numpy.rint(color))

        def set_vertical_colors(line_points):
            y_point = line_points[1][0] + 5
            if y_point >= disparity.shape[1]: y_point = disparity.shape[1] -1

            start_color = numpy.average(disparity[:, y_point-5: y_point, line_points[0][0]], axis=1)

            if start_color[0] < 2:
                # print(line_points[1][0])
                start_color = frame_estimator(line_points[1][0])
                # print(start_color)

            end_color = numpy.average(disparity[:, line_points[1][-1]-2:line_points[1][-1], line_points[0][-1]], axis=1)
            # print(end_color)
            change = (end_color[0]-start_color[0]) / len(line_points[0])

            for i in range(len(line_points[0])):
                color = start_color[0] + change * i
                if color < 0:
                    color = 0
                elif color > 255:
                    color = 255
                output[:, line_points[1][i], line_points[0][i]] = (numpy.rint(color), numpy.rint(color), numpy.rint(color))



        for group in groups:
            # Get furthest points
            Y, X = numpy.where(group==1)

            if min(Y) > y/2: # under focal line


                # Fit between thingys
                if max(X) < x and min(X) < x: # completely left side of focal
                    for _y in range(min(Y), max(Y)):
                        line_points = self.get_line_points(min(X), _y, x, y, x_limit=(0, max(X)+ 10))
                        set_horisontal_colors(line_points)

                elif max(X) > x and min(X) > x: # completely rigth side
                    for _y in range(min(Y), max(Y)):
                        line_points = self.get_line_points(max(X), _y, x, y, x_limit=(min(X) - 10, 1024))
                        set_horisontal_colors(line_points)

                else: # Also at the middle
                    # if very at sides also swipe them
                    if min(X) < disparity.shape[2] - (x * 1.5):
                        # raise Exception("HERE I AM!")
                        for _y in range(min(Y), max(Y)):
                            line_points = self.get_line_points(min(X), _y, x, y, x_limit=(0, max(X)+ 10))
                            set_horisontal_colors(line_points)
                    if max(X) > (x * 1.5):
                        for _y in range(min(Y), max(Y)):
                            line_points = self.get_line_points(max(X), _y, x, y, x_limit=(min(X) - 10, 1024))
                            set_horisontal_colors(line_points)
                    for _x in range(min(X), max(X)):
                        line_points = self.get_line_points(_x, max(Y), x, y, x_limit=(min(X) - 10, 1024))
                        set_vertical_colors(line_points)


            else:
                if max(X) < x: # completely left side of focal
                    for _y in range(min(Y), max(Y)):
                        line_points = self.get_line_points(min(X), _y, x, y, x_limit=(0, max(X)+ 10))
                        set_horisontal_colors(line_points)

                elif max(X) > x: # completely rigth side
                    for _y in range(min(Y), max(Y)):
                        line_points = self.get_line_points(max(X), _y, x, y, x_limit=(min(X) - 10, 1024))
                        set_horisontal_colors(line_points)
                for _x in range(min(X), max(X)):
                    line_points = self.get_line_points(_x, max(Y), x, y, x_limit=(min(X) - 10, 1024))
                    set_vertical_colors(line_points)
            # print(min(X))


        return torch.from_numpy(output)
