import csv
import math
import numpy as np
import matplotlib.pyplot as plt

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self) -> str:
        return str(self.x) + ' ' + str(self.y)

class Object:
    def __init__(self, features: list[float], values: float):
        self.features = features
        self.values = values

    def print(self):
        for feature in self.features:
            print(feature, end=' ')
        print(self.values)

    def to_point(self)->list[Point]:
        points = []
        for feature in self.features:
            points.add(Point(feature, self.values))
        return points


class LinearRegression:
    @staticmethod
    def plot(gradient: float, lst: list[Point], x_label: str, y_label: str, mx: float):
        if lst != None:
            xpoints = np.array([point.x for point in lst])
            ypoints = np.array([point.y for point in lst])
            plt.scatter(xpoints, ypoints)

        if gradient !=  None:
            x = np.linspace(-1, mx)
            y = math.tan(gradient * math.pi / 180)*x
            plt.plot(x, y, color='black')

        plt.xlabel(x_label)  
        plt.ylabel(y_label)
        plt.show()

    @staticmethod
    def optimize(lst: list[Point])->float:
        mx = math.inf
        my = math.inf
        max_key = -math.inf
        for point in lst:
            mx = min(mx, point.x)
            my = min(my, point.y)
            max_key = max(max_key, point.x)
        for i in range(len(lst)):
            lst[i].x -= mx
            lst[i].y -= my
        return max_key

    @staticmethod
    def yAxisCollision(slope: float, point: Point)->float:
        return point.y - slope * point.x

    @staticmethod
    def toGradient(slope: float)->float:
        return math.tan(slope * math.pi / 180)

    @staticmethod
    def getCollisionPoint(slope: float, vslope: float, y0: float)->Point:
        x = y0 / (slope - vslope)
        y = slope * x
        return Point(x, y)

    @staticmethod
    def getMirroredPointDisToCenter(slope: float, point: Point)->float:
        slope = LinearRegression.toGradient(slope)
        vslope = -1/slope
        y0 = LinearRegression.yAxisCollision(vslope, point)
        vpoint = LinearRegression.getCollisionPoint(slope, vslope, y0)
        return math.sqrt(vpoint.x * vpoint.x + vpoint.y * vpoint.y)

    @staticmethod
    def fitLine(lst: list[Point])->float:
        slope = 0.1
        bestSlope = 0.0
        bestDisSum = -math.inf
        while slope <= 90:
            disSum = sum([LinearRegression.getMirroredPointDisToCenter(slope, point) for point in lst])
            if disSum > bestDisSum:
                bestDisSum = disSum
                bestSlope = slope
            slope += 0.01
        return bestSlope

    @staticmethod
    def loadData(address: str)->list[Point]:
        points = list()
        with open(address, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for line in csvreader:
                points += [Point(float(line[0]), float(line[1]))]
        return points

class MultipleRegression:
    @staticmethod
    def plot(objects: list, slopes: list[float], n_features: list[str], n_values: str, mx: list[float]):
        all_plots = []
        for i in range(len(n_features)):
            points = []
            for object in objects:
                points.append(Point(object.features[i], object.values))
            all_plots.append([slopes[i], points, n_features[i], n_values])
        for index, item in enumerate(all_plots):
            LinearRegression.plot(*item, [mx[index]])

    @staticmethod
    def optimize(objects: list[Object])->list[float]:
        mv = math.inf
        for obj in objects:
            mv = min(mv, obj.values)
        for obj in objects:
            obj.values -= mv
        
        max_keys = []
        n = len(objects[0].features)
        for i in range(n):
            mx = math.inf
            max_key = -math.inf
            for obj in objects:
                mx = min(mx, obj.features[i])
                max_key = max(max_key, obj.features[i])
            for obj in objects:
                obj.features[i] -= mx      
            max_keys.append(max_key)
        return max_keys

    @staticmethod
    def loadData(address: str)->(str, list[str], list[Object]):
        lines = []
        objects = list()
        with open(address, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for line in csvreader:
                lines.append(line)
            values = list(set(line[0] for line in lines[1: ]))
            for line in lines[1:]:
                items = list(map(float, line[1:]))
                objects += [Object(items, values.index(line[0]))]
        return (lines[0][0], lines[0][1:], objects)

    @staticmethod
    def fitlines(objects: list[Object], n: int)->list[float]:
        slopes = []
        for i in range(n):
            points = []
            for obj in objects:
                points.append(Point(obj.features[i], obj.values))
            slopes.append(LinearRegression.fitLine(points))
        return slopes
