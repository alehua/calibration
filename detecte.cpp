#include"detecte.h"

FindCorners::FindCorners(Mat img)
{
	radius.push_back(4);
	radius.push_back(8);
	radius.push_back(12);
	templateProps.push_back(Point2f((float)0, (float)CV_PI / 2));
	templateProps.push_back(Point2f((float)CV_PI / 4, (float)-CV_PI / 4));
	templateProps.push_back(Point2f((float)0, (float)CV_PI / 2));
	templateProps.push_back(Point2f((float)CV_PI / 4, (float)-CV_PI / 4));
	templateProps.push_back(Point2f((float)0, (float)CV_PI / 2));
	templateProps.push_back(Point2f((float)CV_PI / 4, (float)-CV_PI / 4));
}
//**************************正太分布*****************************//
//dist：距离中心距离
//sigma：方差
//*************************************************************************//
float FindCorners::normpdf(float dist, float mu, float sigma){
	return exp(-0.5*(dist - mu)*(dist - mu) / (sigma*sigma)) / (std::sqrt(2 * CV_PI)*sigma);
}
//**************************生成核*****************************//
//angle代表核类型：45度核和90度核
//kernelSize代表核大小（最终生成的核的大小为kernelSize*2+1）
//kernelA...kernelD是生成的核
//*************************************************************************//
void FindCorners::createkernel(float angle1, float angle2, int kernelSize, Mat &kernelA, Mat &kernelB, Mat &kernelC, Mat &kernelD){

	int width = (int)kernelSize * 2 + 1;
	int height = (int)kernelSize * 2 + 1;
	kernelA = cv::Mat::zeros(height, width, CV_32F);
	kernelB = cv::Mat::zeros(height, width, CV_32F);
	kernelC = cv::Mat::zeros(height, width, CV_32F);
	kernelD = cv::Mat::zeros(height, width, CV_32F);

	for (int u = 0; u<width; ++u){
		for (int v = 0; v<height; ++v){
			float vec[] = { u - kernelSize, v - kernelSize };//相当于将坐标原点移动到核中心
			float dis = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1]);//相当于计算到中心的距离
			float side1 = vec[0] * (-sin(angle1)) + vec[1] * cos(angle1);//相当于将坐标原点移动后的核进行旋转，以此产生四种核
			float side2 = vec[0] * (-sin(angle2)) + vec[1] * cos(angle2);//X=X0*cos+Y0*sin;Y=Y0*cos-X0*sin
			if (side1 <= -0.1&&side2 <= -0.1){
				kernelA.ptr<float>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
			if (side1 >= 0.1&&side2 >= 0.1){
				kernelB.ptr<float>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
			if (side1 <= -0.1&&side2 >= 0.1){
				kernelC.ptr<float>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
			if (side1 >= 0.1&&side2 <= -0.1){
				kernelD.ptr<float>(v)[u] = normpdf(dis, 0, kernelSize / 2);
			}
		}
	}
	//归一化
	kernelA = kernelA / cv::sum(kernelA)[0];
	kernelB = kernelB / cv::sum(kernelB)[0];
	kernelC = kernelC / cv::sum(kernelC)[0];
	kernelD = kernelD / cv::sum(kernelD)[0];

}
// 获取src1与src2 对应位置元素最小值，输出到dst
void FindCorners::getMin(Mat src1, Mat src2, Mat &dst){
	int rowsLeft = src1.rows;
	int colsLeft = src1.cols;
	int rowsRight = src2.rows;
	int colsRight = src2.cols;
	if (rowsLeft != rowsRight || colsLeft != colsRight)return;

	int channels = src1.channels();

	int nr = rowsLeft;
	int nc = colsLeft;
	if (src1.isContinuous()){
		nc = nc*nr;
		nr = 1;
		//std::cout<<"continue"<<std::endl;
	}
	for (int i = 0; i<nr; i++){
		const float* dataLeft = src1.ptr<float>(i);
		const float* dataRight = src2.ptr<float>(i);
		float* dataResult = dst.ptr<float>(i);
		for (int j = 0; j<nc*channels; ++j){
			dataResult[j] = (dataLeft[j]<dataRight[j]) ? dataLeft[j] : dataRight[j];
		}
	}
}
// 获取src1与src2 对应位置元素最大值，输出到dst
void FindCorners::getMax(Mat src1, Mat src2, Mat &dst){
	int rowsLeft = src1.rows;
	int colsLeft = src1.cols;
	int rowsRight = src2.rows;
	int colsRight = src2.cols;
	if (rowsLeft != rowsRight || colsLeft != colsRight)return;

	int channels = src1.channels();

	int nr = rowsLeft;
	int nc = colsLeft;
	if (src1.isContinuous()){
		nc = nc*nr;
		nr = 1;
		//std::cout<<"continue"<<std::endl;
	}
	for (int i = 0; i<nr; i++){
		const float* dataLeft = src1.ptr<float>(i);
		const float* dataRight = src2.ptr<float>(i);
		float* dataResult = dst.ptr<float>(i);
		for (int j = 0; j<nc*channels; ++j){
			dataResult[j] = (dataLeft[j] >= dataRight[j]) ? dataLeft[j] : dataRight[j];
		}
	}
}

//**************************获取梯度角度和权重*****************************//
//imgDu：x方向梯度
//imgDv：y方向梯度
//imgAngle：梯度角度
//imgWeight：梯度权重
//*************************************************************************//
void FindCorners::getImageAngleAndWeight(Mat img, Mat &imgDu, Mat &imgDv, Mat &imgAngle, Mat &imgWeight){
	Mat sobelKernel(3, 3, CV_32F);
	Mat sobelKernelTrs(3, 3, CV_32F);
	//soble滤波器算子核
	sobelKernel.col(0).setTo(cv::Scalar(-1));
	sobelKernel.col(1).setTo(cv::Scalar(0));
	sobelKernel.col(2).setTo(cv::Scalar(1));

	sobelKernelTrs = sobelKernel.t();

	filter2D(img, imgDu, CV_32F, sobelKernel);
	filter2D(img, imgDv, CV_32F, sobelKernelTrs);

	if (imgDu.size() != imgDv.size())return;

	for (int i = 0; i < imgDu.rows; i++)
	{
		float* dataDv = imgDv.ptr<float>(i);
		float* dataDu = imgDu.ptr<float>(i);
		float* dataAngle = imgAngle.ptr<float>(i);
		float* dataWeight = imgWeight.ptr<float>(i);
		for (int j = 0; j < imgDu.cols; j++)
		{
			if (dataDu[j]>0.000001)
			{
				dataAngle[j] = atan2((float)dataDv[j], (float)dataDu[j]);
				if (dataAngle[j] < 0)dataAngle[j] = dataAngle[j] + CV_PI;
				else if (dataAngle[j] > CV_PI)dataAngle[j] = dataAngle[j] - CV_PI;
			}
			dataWeight[j] = std::sqrt((float)dataDv[j] * (float)dataDv[j] + (float)dataDu[j] * (float)dataDu[j]);
		}
	}
}

//**************************非极大值抑制*****************************//
//inputCorners是输入角点，outputCorners是非极大值抑制后的角点
//threshold是设定的阈值
//margin是进行非极大值抑制时检查方块与输入矩阵边界的距离，patchSize是该方块的大小
//*************************************************************************//
void FindCorners::nonMaximumSuppression(Mat& inputCorners, vector<Point>& outputCorners, float threshold, int margin, int patchSize)
{
	if (inputCorners.size <= 0)
	{
		cout << "The imput mat is empty!" << endl; return;
	}
	for (int i = margin + patchSize; i < inputCorners.cols - (margin + patchSize); i = i + patchSize + 1)//移动检查方块，每次移动一个方块的大小
	{
		for (int j = margin + patchSize; j < inputCorners.rows - (margin + patchSize); j = j + patchSize + 1)
		{
			float maxVal = inputCorners.ptr<float>(j)[i];
			int maxX = i; int maxY = j;
			for (int m = i; m < i + patchSize + 1; m++)//找出该检查方块中的局部最大值
			{
				for (int n = j; n < j + patchSize + 1; n++)
				{
					float temp = inputCorners.ptr<float>(n)[m];
					if (temp>maxVal)
					{
						maxVal = temp; maxX = m; maxY = n;
					}
				}
			}
			if (maxVal < threshold)continue;//若该局部最大值小于阈值则不满足要求
			int flag = 0;
			for (int m = maxX - patchSize; m < min(maxX + patchSize, inputCorners.cols - margin); m++)//二次检查
			{
				for (int n = maxY - patchSize; n < min(maxY + patchSize, inputCorners.rows - margin); n++)
				{
					if (inputCorners.ptr<float>(n)[m]>maxVal && (m<i || m>i + patchSize || n<j || n>j + patchSize))
					{
						flag = 1; break;
					}
				}
				if (flag)break;
			}
			if (flag)continue;
			outputCorners.push_back(Point(maxX, maxY));
			std::vector<float> e1(2, 0.0);
			std::vector<float> e2(2, 0.0);
			cornersEdge1.push_back(e1);
			cornersEdge2.push_back(e2);
		}
	}
}


//find modes of smoothed histogram
void FindCorners::findModesMeanShift(vector<float> hist, vector<float> &hist_smoothed, vector<pair<float, int>> &modes, float sigma){
	//efficient mean - shift approximation by histogram smoothing
	//compute smoothed histogram
	bool allZeros = true;
	for (int i = 0; i < hist.size(); i++)
	{
		float sum = 0;
		for (int j = -(int)round(2 * sigma); j <= (int)round(2 * sigma); j++)
		{
			int idx = 0;
			if ((i + j) < 0)idx = i + j + hist.size();
			else if ((i + j) >= 32)idx = i + j - hist.size();
			else idx = (i + j);
			sum = sum + hist[idx] * normpdf(j, 0, sigma);
		}
		hist_smoothed[i] = sum;
		if (abs(hist_smoothed[i] - hist_smoothed[0])>0.0001)allZeros = false;// check if at least one entry is non - zero
		//(otherwise mode finding may run infinitly)
	}
	if (allZeros)return;
	for (int i = 0; i<hist.size(); ++i){
		int j = i;
		int curLeft = (j - 1)<0 ? j - 1 + hist.size() : j - 1;
		int curRight = (j + 1)>hist.size() - 1 ? j + 1 - hist.size() : j + 1;
		if (hist_smoothed[curLeft]<hist_smoothed[i] && hist_smoothed[curRight]<hist_smoothed[i]){
			modes.push_back(std::make_pair(hist_smoothed[i], i));
		}
	}
	std::sort(modes.begin(), modes.end());
}
//estimate edge orientations
void FindCorners::edgeOrientations(Mat imgAngle, Mat imgWeight, int index){
	//number of bins (histogram parameter)
	int binNum = 32;

	//convert images to vectors
	if (imgAngle.size() != imgWeight.size())return;
	vector<float> vec_angle, vec_weight;
	for (int i = 0; i < imgAngle.cols; i++)
	{
		for (int j = 0; j < imgAngle.rows; j++)
		{
			// convert angles from normals to directions
			float angle = imgAngle.ptr<float>(j)[i] + CV_PI / 2;
			angle = angle>CV_PI ? (angle - CV_PI) : angle;
			vec_angle.push_back(angle);

			vec_weight.push_back(imgWeight.ptr<float>(j)[i]);
		}
	}

	//create histogram
	vector<float> angleHist(binNum, 0);
	for (int i = 0; i < vec_angle.size(); i++)
	{
		int bin = max(min((int)floor(vec_angle[i] / (CV_PI / binNum)), binNum - 1), 0);
		angleHist[bin] = angleHist[bin] + vec_weight[i];
	}

	// find modes of smoothed histogram
	vector<float> hist_smoothed(angleHist);
	vector<std::pair<float, int> > modes;
	findModesMeanShift(angleHist, hist_smoothed, modes, 1);

	// if only one or no mode = > return invalid corner
	if (modes.size() <= 1)return;

	//extract 2 strongest modes and compute orientation at modes
	std::pair<float, int> most1 = modes[modes.size() - 1];
	std::pair<float, int> most2 = modes[modes.size() - 2];
	float most1Angle = most1.second*CV_PI / binNum;
	float most2Angle = most2.second*CV_PI / binNum;
	float tmp = most1Angle;
	most1Angle = (most1Angle>most2Angle) ? most1Angle : most2Angle;
	most2Angle = (tmp>most2Angle) ? most2Angle : tmp;

	// compute angle between modes
	float deltaAngle = min(most1Angle - most2Angle, most2Angle + (float)CV_PI - most1Angle);

	// if angle too small => return invalid corner
	if (deltaAngle <= 0.3)return;

	//set statistics: orientations
	cornersEdge1[index][0] = cos(most1Angle);
	cornersEdge1[index][1] = sin(most1Angle);
	cornersEdge2[index][0] = cos(most2Angle);
	cornersEdge2[index][1] = sin(most2Angle);
}
//亚像素精度找角点
void FindCorners::refineCorners(vector<Point> &cornors, Mat imgDu, Mat imgDv, Mat imgAngle, Mat imgWeight, float radius){
	// image dimensions
	int width = imgDu.cols;
	int height = imgDu.rows;
	// for all corners do
	for (int i = 0; i < cornors.size(); i++)
	{
		//extract current corner location
		int cu = cornors[i].x;
		int cv = cornors[i].y;
		// estimate edge orientations
		int startX, startY, ROIwidth, ROIheight;
		startX = max(cu - radius, (float)0);
		startY = max(cv - radius, (float)0);
		ROIwidth = min(cu + radius, (float)width - 1) - startX;
		ROIheight = min(cv + radius, (float)height - 1) - startY;

		Mat roiAngle, roiWeight;
		roiAngle = imgAngle(Rect(startX, startY, ROIwidth, ROIheight));
		roiWeight = imgWeight(Rect(startX, startY, ROIwidth, ROIheight));
		edgeOrientations(roiAngle, roiWeight, i);

		// continue, if invalid edge orientations
		if (cornersEdge1[i][0] == 0 && cornersEdge1[i][1] == 0 || cornersEdge2[i][0] == 0 && cornersEdge2[i][1] == 0)continue;
	}
}


//**************************角点检测函数*****************************//
//Src：输入图像
//resultCornors：检测到角点
//threshhold：非极大值抑制大小
//*************************************************************************// 
void FindCorners::detectCorners(Mat &Src, vector<Point> &resultCornors, float threshhold){
	Mat gray, imageNorm;
	gray = Src.clone();
	normalize(gray, imageNorm, 0, 1, cv::NORM_MINMAX, CV_32F);//对灰度图进行归一化

	Mat imgCorners = Mat::zeros(imageNorm.size(), CV_32F);//卷积核得出的点
	for (int i = 0; i < 6; i++)
	{
		//按照论文步骤，第一步：用卷积核进行卷积的方式找出可能是棋盘格角点的点
		Mat kernelA1, kernelB1, kernelC1, kernelD1;
		createkernel(templateProps[i].x, templateProps[i].y, radius[i / 2], kernelA1, kernelB1, kernelC1, kernelD1);//1.1 产生四种核


		Mat imgCornerA1(imageNorm.size(), CV_32F);
		Mat imgCornerB1(imageNorm.size(), CV_32F);
		Mat imgCornerC1(imageNorm.size(), CV_32F);
		Mat imgCornerD1(imageNorm.size(), CV_32F);
		filter2D(imageNorm, imgCornerA1, CV_32F, kernelA1);//1.2 用所产生的核对图像做卷积
		filter2D(imageNorm, imgCornerB1, CV_32F, kernelB1);
		filter2D(imageNorm, imgCornerC1, CV_32F, kernelC1);
		filter2D(imageNorm, imgCornerD1, CV_32F, kernelD1);

		Mat imgCornerMean(imageNorm.size(), CV_32F);
		imgCornerMean = (imgCornerA1 + imgCornerB1 + imgCornerC1 + imgCornerD1) / 4;//1.3 按照公式进行计算
		Mat imgCornerA(imageNorm.size(), CV_32F);
		Mat imgCornerB(imageNorm.size(), CV_32F);
		Mat imgCorner1(imageNorm.size(), CV_32F);
		Mat imgCorner2(imageNorm.size(), CV_32F);

		getMin(imgCornerA1 - imgCornerMean, imgCornerB1 - imgCornerMean, imgCornerA);
		getMin(imgCornerMean - imgCornerC1, imgCornerMean - imgCornerD1, imgCornerB);
		getMin(imgCornerA, imgCornerB, imgCorner1);

		getMin(imgCornerMean - imgCornerA1, imgCornerMean - imgCornerB1, imgCornerA);
		getMin(imgCornerC1 - imgCornerMean, imgCornerD1 - imgCornerMean, imgCornerB);
		getMin(imgCornerA, imgCornerB, imgCorner2);

		getMax(imgCorners, imgCorner1, imgCorners);
		getMax(imgCorners, imgCorner2, imgCorners);
	}
	nonMaximumSuppression(imgCorners, resultCornors, threshhold, 5, 3);//1.5 非极大值抑制算法进行过滤，获取棋盘格角点初步结果
	
	//**************角点提取完成***********************
//提取角点包含很多重复
//下一步亚像素角点提取
//去重
//*************************************************

#if 1
	//算两个方向的梯度
	Mat imageDu(gray.size(), CV_32F);
	Mat imageDv(gray.size(), CV_32F);
	Mat img_angle(gray.size(), CV_32F);
	Mat img_weight(gray.size(), CV_32F);
	//获取梯度角度和权重
	getImageAngleAndWeight(gray, imageDu, imageDv, img_angle, img_weight);
	//subpixel refinement
	refineCorners(resultCornors, imageDu, imageDv, img_angle, img_weight, 10);
	if (resultCornors.size()>0)
	{
		for (int i = 0; i < resultCornors.size(); i++)
		{
			if (cornersEdge1[i][0] == 0 && cornersEdge1[i][0] == 0)
			{
				resultCornors[i].x = 0; resultCornors[i].y = 0;
			}

		}
	}
	FileStorage fs2("create\\position.yml", FileStorage::WRITE);//写XML文件
	fs2 << "img_corners" << resultCornors;
	fs2.release();
#endif
}