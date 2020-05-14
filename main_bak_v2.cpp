#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>            
#include <string>
#include <sstream>
#include <cstdlib>
#include <fstream>
#include <io.h>
#include<time.h>
#include "exif.h"
#include "detecte.h"
#include "plot.h"
using namespace std;
using namespace cv;

#define RED Scalar(0,0,255)
#define GREEN Scalar(0,255,0)


//************************文件夹图片获取*****************************//
//image_path：图片路径
//image_list：每一张图片路径
//*************************************************************************//
void interprate(vector<string>& image_list,string image_path);
Mat next_img(vector<string>& image_list,int idx);
Mat subplot(vector<Mat> array, int m, int n, int type);
// 矩阵计算部分
void TransposeMatrix(double *m1, double *m2, int row, int col);
void InverseMatrix(double *a, int n); 
void MultiplyMatrix(double *m1, double *m2, double *result, int m, int n, int l); 

int main()
{
#if 0
	/*
	没有提取角点执行该部分代码
	*/
	vector<string> image_list;
	string img_path = "data\\sample\\";
	interprate(image_list, img_path);
	vector<vector<Point2f> > imagePoints;
	Mat cameraMatrix, distCoeffs;
	for (int i = 0; i < image_list.size(); i++)
	{
		Mat view, view_gray,view_blur;
		view = next_img(image_list,i);
		if (!view.empty())
		{
			resize(view, view, Size(600, 800));
			cvtColor(view, view_gray, CV_RGB2GRAY);
			medianBlur(view_gray, view_blur, 5);
			vector<Point> corners;
			FindCorners corner_detector(view_blur);
			corner_detector.detectCorners(view_blur, corners, 0.14);   // 0.15 需要调整
			ofstream fout("create\\position.txt");
			fout <<"角点数量：" <<corners.size() << endl;
			for (auto i : corners)
			{
				fout << i << endl;
				plot_circle(view, i, 5, RED, 2);
			}
			fout << "finished...." << endl;
			imshow("result", view);
			waitKey(0);
			imwrite("create\\result.jpg", view);
		}
	}
#endif
#if 0
	/*
	已经提取好角点
	读取角点 position.yml
	*/
	vector<Point2f> corners_buf, corners_detected;
	FileStorage fs("create\\position.yml", FileStorage::READ); 
	if (!fs.isOpened())
	{
		cout << "Could not open the configuration file: \"" << "create\\position.yml" << endl;
		return -1;
	}
	fs["img_corners"] >> corners_buf;
	fs.release();
	/*
	读入图像
	*/
	Mat view, view_gray, view_blur;
	vector<string> image_list;
	image_list.push_back("data\\sample\\src.jpg");
	view = next_img(image_list, 0);
	if (!view.empty())
	{
		resize(view, view, Size(600, 800));
		cvtColor(view, view_gray, CV_RGB2GRAY);
		medianBlur(view_gray, view_blur, 5);
	}
	cornerSubPix(view_blur, corners_buf, Size(11, 11),
		Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
	for (auto i : corners_buf)
	{
		if (i.x == 0.0)
			continue;
		corners_detected.push_back(i);
	}
	FileStorage fot("create\\position_new.yml", FileStorage::WRITE);//写XML文件
	fot << "img_corners" << corners_detected;
	fot << "image" << view;
	fot.release();
#endif
#if 0
	//ofstream fout("create\\points.txt");
	vector<Point2f> corners_buf;
	Mat image;
	FileStorage fs("create\\position_new.yml", FileStorage::READ);
	if (!fs.isOpened())
	{
		cout << "Could not open the configuration file: \"" << "create\\position.yml" << endl;
		return -1;
	}
	fs["image"] >> image;
	fs["img_corners"] >> corners_buf;
	fs.release();
	// text parameters
	int font_face = FONT_HERSHEY_COMPLEX;
	double font_scale = 0.35;
	int thickness = 1;
	int baseline = 0;
	Size text_size;
	for (int idx = 0; idx < corners_buf.size(); idx++)
	{
		//fout << idx + 1 << " " << corners_buf[idx].x << " " << corners_buf[idx].y << endl;
		plot_circle(image, corners_buf[idx], 2, GREEN, 1);
		string text;
		text = to_string(idx + 1);
		
		text_size = getTextSize(text, font_face, font_scale, thickness, &baseline);
		Point origin;
		origin.x = corners_buf[idx].x - text_size.width;
		origin.y = corners_buf[idx].y + text_size.height;
		putText(image, text, origin, font_face, font_scale, RED);
	}
	//fout.close();
	imwrite("create\\cordinate.jpg", image);
#endif
#if 1
	double temp;
	vector<double> world;
	ifstream infile;
	infile.open("data\\world.txt");
	while (infile >> temp) world.push_back(temp*0.02);
	infile.close();
	vector<double> corner;
	ifstream infile1;
	infile1.open("data\\image.txt");
	while (infile1 >> temp) corner.push_back(temp);
	infile1.close();

	int num = corner.size() / 2;
	// AL = C
	double *A = new double[num * 2 * 11]; //每一组角点 2*11个数据
	double *C = new double[num * 2];
	for (int i = 0; i < num; i++)
	{
		A[0 + 22 * i] = world[3 * i + 0];
		A[1 + 22 * i] = world[3 * i + 1];
		A[2 + 22 * i] = world[3 * i + 2];
		A[3 + 22 * i] = 1;
		A[4 + 22 * i] = 0;
		A[5 + 22 * i] = 0;
		A[6 + 22 * i] = 0;
		A[7 + 22 * i] = 0;
		A[8 + 22 * i] = -world[3 * i + 0] * corner[2 * i + 0];
		A[9 + 22 * i] = -world[3 * i + 1] * corner[2 * i + 0];
		A[10 + 22 * i] = -world[3 * i + 2] * corner[2 * i + 0];
		A[11 + 22 * i] = 0;
		A[12 + 22 * i] = 0;
		A[13 + 22 * i] = 0;
		A[14 + 22 * i] = 0;
		A[15 + 22 * i] = world[3 * i + 0];
		A[16 + 22 * i] = world[3 * i + 1];
		A[17 + 22 * i] = world[3 * i + 2];
		A[18 + 22 * i] = 1;
		A[19 + 22 * i] = -world[3 * i + 0] * corner[2 * i + 1];
		A[20 + 22 * i] = -world[3 * i + 1] * corner[2 * i + 1];
		A[21 + 22 * i] = -world[3 * i + 2] * corner[2 * i + 1];

		C[0 + 2 * i] = corner[2 * i + 0];
		C[1 + 2 * i] = corner[2 * i + 1];
	}
	// L = (ATA)(-1)ATC
	double *AT = new double[11 * 2 * num]; //大小11*2*num
	double ATA[11 * 11] = { 0.0 };    //大小11*11
	double ATC[11] = { 0.0 };         // 11*1
	double XX[11] = { 0.0 };
	TransposeMatrix(A, AT, 2*num, 11);           // A'    11*12
	MultiplyMatrix(AT, A, ATA, 11, 2 * num, 11);   // A'A   11*11
	MultiplyMatrix(AT, C, ATC, 11, 2 * num, 1);    // A'C   11*1
	InverseMatrix(ATA, 11);                   // A'A-1 11*11
	MultiplyMatrix(ATA, ATC, XX, 11, 11, 1);  // XX 就是结果
#if 0
	//输出结果
	double x0, y0;										//这地方公式有问题
	x0 = XX[0] * XX[8] + XX[1] * XX[9] + XX[2] * XX[10];/// (XX[8] * XX[8] + XX[9] * XX[9] + XX[10] * XX[10]);;
	y0 = XX[4] * XX[8] + XX[5] * XX[9] + XX[6] * XX[10];//  (XX[8] * XX[8] + XX[9] * XX[9] + XX[10] * XX[10]);;
	//计算fx,fy
	double fx, fy;  //x向主距，y向主距
	double tmp1 = 0, tmp2 = 0, tmp3 = 0;

	tmp1 = XX[0] * XX[0] + XX[1] * XX[1] + XX[2] * XX[2];
	tmp3 = XX[4] * XX[4] + XX[5] * XX[5] + XX[6] * XX[6];
	fx = sqrt(tmp1 - x0*x0);
	fy = sqrt(tmp3 - y0*y0);
#else 
	//内方位元素x0 y0解算
	double x0, y0;
	x0 = -(XX[0] * XX[8] + XX[1] * XX[9] + XX[2] * XX[10]) / (XX[8] * XX[8] + XX[9] * XX[9] + XX[10] * XX[10]);
	y0 = -(XX[4] * XX[8] + XX[5] * XX[9] + XX[6] * XX[10]) / (XX[8] * XX[8] + XX[9] * XX[9] + XX[10] * XX[10]);

	//计算fx,fy
	double fx, fy;  //x向主距，y向主距
	double tem1 = 0, tem2 = 0, tem3 = 0;
	//double *c = new double[num];

	tem1 = XX[0] * XX[0] + XX[1] * XX[1] + XX[2] * XX[2];
	tem2 = XX[8] * XX[8] + XX[9] * XX[9] + XX[10] * XX[10];
	tem3 = XX[4] * XX[4] + XX[5] * XX[5] + XX[6] * XX[6];

	fx = sqrt(tem1 / tem2 - x0*x0);

	double r3 = 0, ds = 0;
	r3 = 1 / (XX[8] * XX[8] + XX[9] * XX[9] + XX[10] * XX[10]);
	ds = sqrt((r3*r3*tem1 - x0*x0) / (r3*r3*tem3 - y0*y0)) - 1;
	fy = fx / (1 + ds);
#endif
	FileStorage fout("create\\params.yml", FileStorage::WRITE);
	fout << "u0" << x0;
	fout << "v0" << y0;
	fout << "fx" << fx;
	fout << "fy" << fy;
	fout.release();
#endif
	system("pause");
	return 0;
}


void interprate(vector<string>& image_list, string image_path)
{
	string to_serach = image_path + "*.jpg";
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	hFile = _findfirst(to_serach.c_str(), &fileinfo);
	if (hFile == -1) return;
	image_list.push_back(image_path + fileinfo.name);
	while (!_findnext(hFile, &fileinfo))
	{
		image_list.push_back(image_path + fileinfo.name);
	}
	_findclose(hFile);
}
Mat next_img(vector<string>& image_list, int idx)
{
	Mat result;
	if (idx < image_list.size())
	{
		fstream fin(image_list[idx].c_str(), ifstream::in | ifstream::binary);
		result = imread(image_list[idx]);
		int orientation = IMAGE_ORIENTATION_TL;
		ExifReader reader(fin);
		if (reader.parse())
		{
			ExifEntry_t entry = reader.getTag(ORIENTATION);
			if (entry.tag != INVALID_TAG)
			{
				orientation = entry.field_u16;
			}
		}
		switch (orientation)
		{
		case    IMAGE_ORIENTATION_TL: //0th row == visual top, 0th column == visual left-hand side
			//do nothing, the image already has proper orientation
			break;
		case    IMAGE_ORIENTATION_TR: //0th row == visual top, 0th column == visual right-hand side
			flip(result, result, 1); //flip horizontally
			break;
		case    IMAGE_ORIENTATION_BR: //0th row == visual bottom, 0th column == visual right-hand side
			flip(result, result, -1);//flip both horizontally and vertically
			break;
		case    IMAGE_ORIENTATION_BL: //0th row == visual bottom, 0th column == visual left-hand side
			flip(result, result, 0); //flip vertically
			break;
		case    IMAGE_ORIENTATION_LT: //0th row == visual left-hand side, 0th column == visual top
			transpose(result, result);
			break;
		case    IMAGE_ORIENTATION_RT: //0th row == visual right-hand side, 0th column == visual top
			transpose(result, result);
			flip(result, result, 1); //flip horizontally
			break;
		case    IMAGE_ORIENTATION_RB: //0th row == visual right-hand side, 0th column == visual bottom
			transpose(result, result);
			flip(result, result, -1); //flip both horizontally and vertically
			break;
		case    IMAGE_ORIENTATION_LB: //0th row == visual left-hand side, 0th column == visual bottom
			transpose(result, result);
			flip(result, result, 0); //flip vertically
			break;
		default:
			//by default the image read has normal (JPEG_ORIENTATION_TL) orientation
			break;
		}
	}
	return result;
}
Mat subplot(vector<Mat> array, int m, int n, int type){
	int all = array.size();
	int col = array[0].cols; // 列  
	int row = array[0].rows; //行  
	int new_col, new_row;
	new_col = col*n;
	new_row = row*m;
	Mat newImage(new_row, new_col, type);
	int x = 0;
	int y = 0;
	for (auto img : array){
		Mat imageROI = newImage(Rect(x*col, y*row, col, row));
		img.copyTo(imageROI);
		x++;
		if (x == n){
			y++;
			x = 0;
		}
		if (y == m){
			return newImage;
		}
	}
}

void TransposeMatrix(double *m1, double *m2, int row, int col) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			m2[j * row + i] = m1[i * col + j];
		}
	}
}
void MultiplyMatrix(double *m1, double *m2, double *result, int m, int n, int l) {
	for (int i = 0; i<m; i++) {
		for (int j = 0; j<l; j++) {
			result[i*l + j] = 0.0;							//输出矩阵初始化
			for (int k = 0; k<n; k++)
				result[i*l + j] += m1[i*n + k] * m2[j + k*l];		//输出矩阵赋值
		}
	}
}
void InverseMatrix(double *a, int n) {
	int i, j, k;
	for (k = 0; k<n; k++) {
		for (i = 0; i<n; i++) {
			if (i != k)
				*(a + i*n + k) = -*(a + i*n + k) / (*(a + k*n + k));
		}
		*(a + k*n + k) = 1 / (*(a + k*n + k));
		for (i = 0; i<n; i++) {
			if (i != k) {
				for (j = 0; j<n; j++) {
					if (j != k)
						*(a + i*n + j) += *(a + k*n + j)* *(a + i*n + k);
				}
			}
		}
		for (j = 0; j<n; j++) {
			if (j != k)
				*(a + k*n + j) *= *(a + k*n + k);
		}
	}
}
