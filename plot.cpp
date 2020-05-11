#include "plot.h"

void demo()
{
	Mat image2 = Mat::zeros(600, 850, CV_8UC3);//����һ��850x600�Ĵ���
	plot_circle(image2, Point(200, 300), 100, Scalar(225, 0, 225), 7);
	plot_circle(image2, Point(350, 300), 100, Scalar(225, 0, 225), 7);
	plot_circle(image2, Point(500, 300), 100, Scalar(225, 0, 225), 7);
	plot_circle(image2, Point(650, 300), 100, Scalar(225, 0, 225), 7);
	imshow("Audi", image2);
	waitKey(0);

	Mat	image1 = Mat::zeros(900, 900, CV_8UC3);//900x900�Ĵ���
	plot_ellipse(image1, Point(450, 450), Size(400, 250), Scalar(0, 0, 225), 5);//���Ƶ�һ����Բ������Բ����ɫΪ��ɫ
	ellipse(image1, Point(450, 450), Size(250, 110), 90, 0, 360, Scalar(0, 0, 225), 5, 8);//���Ƶڶ�����Բ������Բ
	plot_ellipse(image1, Point(450, 320), Size(280, 120), Scalar(0, 0, 225), 5);//���Ƶ�������Բ��С��Բ���ᣩ
	imshow("Toyota", image1);
	waitKey(0);

	Mat	image3 = Mat::zeros(800, 800, CV_8UC3);//����һ��800x800�Ĵ���
	Rect rec1 = Rect(100, 300, 600, 200);
	Rect rec2 = Rect(300, 100, 200, 600);
	plot_rectangle(image3, rec1.tl(), rec1.br(), Scalar(0, 0, 255), -1);//�����
	plot_rectangle(image3, rec2.tl(), rec2.br(), Scalar(0, 0, 255), -1);//������
	plot_rectangle(image3, Point(300, 300), Point(500, 500), Scalar(0, 0, 255), 3);//��ɫ�����θ��ǣ����룩
	imshow("Cross", image3);
	waitKey(0);
}

void plot_circle(Mat img, Point center, int r, Scalar color, int thick)
{
	circle(img, center, r, color, thick);
}
void plot_ellipse(Mat img, Point center, Size axes, Scalar color, int thick)
{
	ellipse(img, center, axes, 0, 0, 360, color, thick);
}

void plot_rectangle(Mat img, Point s1, Point s2, Scalar color, int thick)
{
	rectangle(img, s1, s2, color, thick);
}