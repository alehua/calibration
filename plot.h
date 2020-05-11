#pragma once

#include<opencv2/core/core.hpp> 
#include<opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

void plot_circle(Mat img, Point center, int r, Scalar color, int thick);
void plot_ellipse(Mat img, Point center, Size axes, Scalar color, int thick);
void plot_rectangle(Mat img, Point s1, Point s2, Scalar color, int thick);
void demo();