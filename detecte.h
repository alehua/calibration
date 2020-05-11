#ifndef DETECTE_H
#define DETECTE_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

class FindCorners
{
public:
	FindCorners(Mat img);
	~FindCorners(){};

public:
	void detectCorners(Mat &Src, vector<Point> &resultCornors, float threshhold);

private:
	//��̬�ֲ�
	float normpdf(float dist, float mu, float sigma);
	//��ȡ��Сֵ
	void getMin(Mat src1, Mat src2, Mat &dst);
	//��ȡ���ֵ
	void getMax(Mat src1, Mat src2, Mat &dst);
	//��ȡ�ݶȽǶȺ�Ȩ��
	void getImageAngleAndWeight(Mat img, Mat &imgDu, Mat &imgDv, Mat &imgAngle, Mat &imgWeight);
	//estimate edge orientations
	void edgeOrientations(Mat imgAngle, Mat imgWeight, int index);
	//find modes of smoothed histogram
	void findModesMeanShift(vector<float> hist, vector<float> &hist_smoothed, vector<pair<float, int>> &modes, float sigma);
	//score corners
	void scoreCorners(Mat img, Mat imgAngle, Mat imgWeight, vector<Point> &cornors, vector<int> radius, vector<float> &score);
	//compute corner statistics
	void cornerCorrelationScore(Mat img, Mat imgWeight, vector<Point2f> cornersEdge, float &score);
	//�����ؾ����ҽǵ�
	void refineCorners(vector<Point> &cornors, Mat imgDu, Mat imgDv, Mat imgAngle, Mat imgWeight, float radius);
	//���ɺ�
	void createkernel(float angle1, float angle2, int kernelSize, Mat &kernelA, Mat &kernelB, Mat &kernelC, Mat &kernelD);
	//�Ǽ���ֵ����
	void nonMaximumSuppression(Mat& inputCorners, vector<Point>& outputCorners, float threshold, int margin, int patchSize);

private:
	vector<Point2f> templateProps;
	vector<int> radius;
	vector<Point> cornerPoints;
	std::vector<std::vector<float> > cornersEdge1;
	std::vector<std::vector<float> > cornersEdge2;
	std::vector<cv::Point* > cornerPointsRefined;

};

#endif 
