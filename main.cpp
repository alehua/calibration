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
#if 1
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
		plot_circle(image, corners_buf[idx], 2, GREEN, 1);
		string text;
		text = to_string(idx + 1);
		
		text_size = getTextSize(text, font_face, font_scale, thickness, &baseline);
		Point origin;
		origin.x = corners_buf[idx].x - text_size.width;
		origin.y = corners_buf[idx].y + text_size.height;
		putText(image, text, origin, font_face, font_scale, RED);
	}
	imwrite("create\\cordinate.jpg", image);
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


