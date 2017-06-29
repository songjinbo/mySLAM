#pragma once
#include <sstream>
#include <fstream>
#include <vector>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>

class CameraParam; //类的前置声明
struct FRAME
{
	int frameID;
	cv::Mat left;
	cv::Mat right;
	cv::Mat depth;
	cv::Mat depthColor;
	cv::Mat desp;
	std::vector<cv::KeyPoint> kp;
};

struct MOTION
{
	cv::Mat rvec,tvec;
	int inliers;
};

void computeKeyPointsAndDesp(FRAME& frame);

MOTION motionEstimation(FRAME& frame1,FRAME& frame2,CameraParam& camera,\
		double good_match_threshold,int min_good_match,int min_inliers,std::ofstream& out);
