# Author: Peilun Kang
# Contact: kangpeilun@nefu.edu.cn
# License: Apache Licence
# Project: VastGaussian
# File: graham_scan.py
# Time: 5/16/24 2:32 PM
# Description: Python implementation of the convex hull algorithm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from shapely.geometry import Polygon, box
from scipy.spatial import ConvexHull


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def distance(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def cross_product(a, b, c):
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def compare_angles(pivot, p1, p2):
    orientation = cross_product(pivot, p1, p2)
    if orientation == 0:
        return distance(pivot, p1) - distance(pivot, p2)
    return -1 if orientation > 0 else 1


def graham_scan(points):
    n = len(points)
    if n < 3:
        return "Convex hull requires at least 3 points."

    pivot = min(points, key=lambda point: (point.y, point.x))
    points = sorted(points, key=lambda point: (np.arctan2(point.y - pivot.y, point.x - pivot.x), -point.y, point.x))

    stack = [points[0], points[1], points[2]]
    for i in range(3, n):
        while len(stack) > 1 and compare_angles(stack[-2], stack[-1], points[i]) > 0:
            stack.pop()
        stack.append(points[i])

    return stack


def plot_convex_hull(points, convex_hull, x, y):
    plt.figure()
    plt.scatter([p.x for p in points], [p.y for p in points], color='b', label="All points")

    # Plot the convex hull
    plt.plot([p.x for p in convex_hull] + [convex_hull[0].x], [p.y for p in convex_hull] + [convex_hull[0].y],
             linestyle='-', color='g', label="Convex hull edges")

    for i in range(len(convex_hull)):
        plt.plot([convex_hull[i].x, convex_hull[(i + 1) % len(convex_hull)].x],
                 [convex_hull[i].y, convex_hull[(i + 1) % len(convex_hull)].y], linestyle='-', color='g')

    plt.plot(x, y)

    plt.show()


def run_graham_scan(points, W, H):
    """Get the convex hull of the region enclosed by 8 points.
    :param points: Projected coordinates of the 8 corner points.
    :param W: Image width.
    :param H: Image height.
    :return: Convex hull point set [x, y].
    """
    # points = [Point(point[0], point[1]) for point in points]
    # convex_hull = graham_scan(points)
    points = np.array(points)
    convex_hull = ConvexHull(np.array(points))
    # convex_hull_polygon = Polygon([(point[0], point[1]) for point in convex_hull])

    convex_hull_list = []
    # plt.plot(points[:, 0], points[:, 1], 'o')
    for i, j in zip(convex_hull.simplices, convex_hull.vertices):
        # plt.plot(points[i, 0], points[i, 1], 'k-')
        convex_hull_list.append(points[j])

    convex_hull_polygon = Polygon(convex_hull_list)
    image_bounds = box(0, 0, W, H)
    # x = [0, W, W, 0, 0]
    # y = [0, 0, H, H, 0]
    # plt.plot(x, y)
    # plt.show()
    # plot_convex_hull(points, convex_hull, x, y)
    # Calculate the intersection of the convex hull and the image bounds
    intersection = convex_hull_polygon.intersection(image_bounds)
    image_area = W * H
    # Calculate the ratio of the intersection area to the image area
    intersection_rate = intersection.area / image_area

    # print("intersection_area: ", intersection.area, " image_area: ", image_area, " intersection_rate: ", intersection_rate)
    return {
        "intersection_area": intersection.area,
        "image_area": image_area,
        "intersection_rate": intersection_rate,
    }
