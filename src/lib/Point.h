//
// Created by dbara on 24/05/21.
//

#ifndef TSP_GA_POINT_H
#define TSP_GA_POINT_H

#include <cmath>
#include <iostream>

// Class to represent points in space.
class Point {
private:
    double xval, yval;
public:
    // Constructor uses default arguments to allow calling with zero, one,
    // or two values.
    explicit Point(double x = 0.0, double y = 0.0) {
        xval = x;
        yval = y;
    }

    // Extractors.
    double x() { return xval; }

    double y() { return yval; }

    // Distance to another point.  Pythagorean thm.
    double dist(Point other) {
        double xd = xval - other.xval;
        double yd = yval - other.yval;
        return sqrt(xd * xd + yd * yd);
    }

    // Add or subtract two points.
    Point add(Point b) {
        return Point(xval + b.xval, yval + b.yval);
    }

    Point sub(Point b) {
        return Point(xval - b.xval, yval - b.yval);
    }

    // Move the existing point.
    void move(double a, double b) {
        xval += a;
        yval += b;
    }

    // Print the point on the stream.  The class ostream is a base class
    // for output streams of various types.
    void print(std::ostream &strm) {
        strm << "(" << xval << "," << yval << ")";
    }
};

#endif //TSP_GA_POINT_H
