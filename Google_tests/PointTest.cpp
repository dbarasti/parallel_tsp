//
// Created by dbara on 23/05/21.
//

#include "gtest/gtest.h"
#include "../src/lib/Point.h"

TEST(PointTestSuite, CorrectDistance){
    Point p1{4,1};
    Point p2{1.2, 4.8};
    double dst = p1.dist(p2);

EXPECT_NEAR(dst,4.720169, 0.00001);
}

TEST(PointTestSuite, CorrectExtractors){
    Point p1{5.6, 7.1};

EXPECT_EQ(p1.x(), 5.6);
EXPECT_EQ(p1.y(), 7.1);
}

TEST(PointTestSuite, AddSubtract){
    Point p1{4.2, 8.5};
    Point toAdd{31.4, 12.4};
    Point resAdd = p1.add(toAdd);

// no side effect expected
EXPECT_EQ(p1.x(), 4.2);
EXPECT_EQ(p1.y(), 8.5);
EXPECT_EQ(toAdd.x(), 31.4);
EXPECT_EQ(toAdd.y(), 12.4);

EXPECT_DOUBLE_EQ(resAdd.x(), 35.6);
EXPECT_DOUBLE_EQ(resAdd.y(), 20.9);

    Point p2{4.2, 8.5};
    Point toSub{31.4, 12.4};
    Point resSub = p1.sub(toSub);

// no side effect expected
EXPECT_EQ(p2.x(), 4.2);
EXPECT_EQ(p2.y(), 8.5);
EXPECT_EQ(toSub.x(), 31.4);
EXPECT_EQ(toSub.y(), 12.4);

EXPECT_DOUBLE_EQ(resSub.x(), -27.2);
EXPECT_DOUBLE_EQ(resSub.y(), -3.9);
}

TEST(PointTestSuite, MovePoint){
     Point p1{4.5, 8.2};
     p1.move(2.1, -4.6);

EXPECT_DOUBLE_EQ(p1.x(), 6.6);
EXPECT_DOUBLE_EQ(p1.y(), 3.6);

}