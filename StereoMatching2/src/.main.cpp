/*
*  stereo_Match.cpp
*  calibration
*
*  Created by Victor  Eruhimov on 1/18/10.
*  Copyright 2010 Argus Corp. All rights reserved.
*
*/
#include <iostream>
#include <string>
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

#include "generatePointCloud.hpp"
#include "stereoMatching.hpp"
#include "readParam.hpp"
#include "motionEstimation.hpp"

//#include <Eigen/Core>
//#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

int main()
{
	ParameterReader pd;
	CameraParam camera(atof(pd.getData("baseline").c_str()),atof(pd.getData("factor").c_str()),atof(pd.getData("cx").c_str()),\
			atof(pd.getData("cy").c_str()),atof(pd.getData("fx").c_str()),atof(pd.getData("fy").c_str()));
	string path = pd.getData("path");
	int startIndex = atoi(pd.getData("start_index").c_str());
	int endIndex = atoi(pd.getData("end_index").c_str());
	int currIndex = startIndex;//当前索引为currIndex

	double good_match_threshold = atof(pd.getData("good_match_threshold").c_str());
	int min_good_match = atoi(pd.getData("min_good_match").c_str());

	//读取left和right文件
	const string llast = path + "left_"+itos(lastIndex)+".png"; 
	const string rlast = path + "right_"+itos(lastIndex)+".png"; 
	const string lcurrent = path + "left_"+itos(currIndex)+".png"; 
	const string rcurrent = path + "right_"+itos(currIndex)+".png"; 
	FRAME frame1,frame2;//frame1代表前一帧，frame2代表当前帧
	frame1.left = cv::imread(llast.c_str(), -1);
	frame1.right = cv::imread(rlast.c_str(), -1);
	frame2.left = cv::imread(lcurrent.c_str(), -1);
	frame2.right = cv::imread(rcurrent.c_str(), -1);

	//计算depth和depthColor;
	int alg = STEREO_BM;
	int SADWindowSize = 0, numberOfDisparities = 0;
	stereoMatching(frame1,camera,alg,SADWindowSize,numberOfDisparities); 
	stereoMatching(frame2,camera,alg,SADWindowSize,numberOfDisparities); 

	//计算位姿估计
	computeKeyPointsAndDesp(frame1);	
	computeKeyPointsAndDesp(frame2);	
	MOTION motion = motionEstimation(frame1,frame2,camera,good_match_threshold,min_good_match);	
	cv::Mat imgshow;
	cv::drawKeypoints(frame1.left,frame1.kp,imgshow,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imshow("keypoints",imgshow);
	cv::waitKey(0);
	std::cout<<"inliers:"<<motion.inliers<<std::endl;
	std::cout<<"R="<<motion.rvec<<std::endl;
	std::cout<<"T="<<motion.tvec<<std::endl;	
	
	//结合点云
	Eigen::Isometry3d T = cvMat2Eigen(motion);

	std::cout<<"converting image to clouds"<<endl;
	PointCloud::Ptr cloud1 = generatePointCloud(frame1.depthColor,frame1.depth,camera);
	PointCloud::Ptr output(new PointCloud());

	output = joinPointCloud(cloud1,frame2,T,camera);
	pcl::io::savePCDFile("../data/result.pcd",*output);
	std::cout<<"Final result saved."<<std::endl;

	pcl::visualization::CloudViewer viewer("viewer");
	viewer.showCloud(output);

	return 0;
}
