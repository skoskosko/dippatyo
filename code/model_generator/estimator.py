
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

    def get_line_points(self, x1, y1, x2, y2, start_radius=1, end_radius=1, x_l = 1024, y_l = 512):
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

            points_x.append(int(x))
            points_y.append(int(y))
                
        points_x.append(x2)
        points_y.append(y2)

        if start_radius == end_radius == 1:
            return numpy.array(points_x), numpy.array(points_y)

        thick_points_x = []
        thick_points_y = []


        for i, (x, y) in enumerate(zip(points_x, points_y)):
            # Calculate how far along the line we are (from 0 at the start to 1 at the end)
            t = i / steps  # Normalized position along the line

            # Linearly interpolate radius between start_radius and end_radius
            current_radius = start_radius * (1 - t) + end_radius * t

            # Round and convert radius to an integer
            current_radius = int(current_radius)

            # Generate points around the line point based on the current radius
            for dx in range(-current_radius, current_radius + 1):
                for dy in range(-current_radius, current_radius + 1):
                    # Only keep the points within the current radius (circle shape)
                    if dx**2 + dy**2 <= current_radius**2:
                        new_x = x + dx
                        new_y = y + dy
                        # Ensure the points are within the bounds (optional)
                        if 0 <= new_x < x_l and 0 <= new_y < y_l:
                            thick_points_x.append(new_x)
                            thick_points_y.append(new_y)
        return numpy.array(thick_points_x), numpy.array(thick_points_y)

    # def sort_by_distance(self, coords, point):
    #     # Calculate Euclidean distance for each coordinate from the given point
    #     distances = numpy.sqrt((coords[:, 0] - point[0])**2 + (coords[:, 1] - point[1])**2)
        
    #     # Sort the coordinates by distance in descending order (furthest to closest)
    #     sorted_indices = numpy.argsort(distances)[::-1]
        
    #     # Return the sorted coordinates
    #     return coords[sorted_indices]
    
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
        x = int(disparity.shape[1]/2.5)
        y = int(disparity.shape[2]/2)

        
        
        color = numpy.array([255, 0, 0])[:, None]

        output = numpy.copy(torch.Tensor.numpy(disparity))
        
        # vertical sides
        # radius = 25
        # points = 22
        # for i in range(points):
        #     x2 = int((output.shape[2]-1) * (i / (points-1)))

        #     line_points = self.get_line_points( x2, 0, y, x, start_radius=radius)
        #     line_point= self.sort_by_distance(line_points, (x,y))
        #     color_gradient = numpy.linspace([255, 0, 0], [0, 0, 255], line_points[0].shape[0]).T
        #     output[:, line_points[1],line_points[0]] = color_gradient

            # line_points = self.get_line_points( x2, 511, y, x, start_radius=radius)
            # color_gradient = numpy.linspace([255, 0, 0], [0, 0, 255], line_points[0].shape[0]).T
            # output[:, line_points[1],line_points[0]] = color_gradient

        # # horisontal sides
        # radius = 25
        # points = 12
        # for i in range(points):
        #     y2 = int((output.shape[1]-1) * (i / (points-1)))

        #     line_points = self.get_line_points(0, y2, y, x, start_radius=radius)
        #     color_repeated = numpy.repeat(color, line_points[0].shape[0], axis=1)
        #     output[:, line_points[1],line_points[0]] = color_repeated

        #     line_points = self.get_line_points(1023, y2, y, x, start_radius=radius)
        #     color_repeated = numpy.repeat(color, line_points[0].shape[0], axis=1)
        #     output[:, line_points[1],line_points[0]] = color_repeated



        for group in groups:
            # Get furthest points
            Y, X = numpy.where(group==1)
            
            print(min(Y))
            print(max(Y))

            # if Y min is lower than y we just blend it in
            # if higher we continue with focal

            if min(Y) < y: # under focal line
                # Fit between thingys
                if max(X) < x and min(X) < x: # completely left side of focal
                    print("WESTSIDEIID")


                    line_points = self.get_line_points(min(X), min(Y), y, x)
                    color_repeated = numpy.repeat(color, line_points[0].shape[0], axis=1)
                    output[:, line_points[1],line_points[0]] = color_repeated


                    line_points = self.get_line_points(min(X), max(Y), y, x)
                    color_repeated = numpy.repeat(color, line_points[0].shape[0], axis=1)
                    output[:, line_points[1],line_points[0]] = color_repeated



                elif max(X) > X and min(X) > x: # completely left side
                    print("EASTSIDEIID")

                else:
                    print("KESKUSTA ON MAHTAVA")
            
            # print(min(X))


            break

        return torch.from_numpy(output)
    
