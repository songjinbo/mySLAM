/*
*  stereo_Match.cpp
*  calibration
*
*  Created by Victor  Eruhimov on 1/18/10.
*  Copyright 2010 Argus Corp. All rights reserved.
*
*/
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
//#include <unistd.h>
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>

#include "generatePointCloud.hpp"
#include "stereoMatching.hpp"
#include "readParam.hpp"
#include "motionEstimation.hpp"

bool readFrame(FRAME &frame,int index,std::string path);

int main()
{
	//读取参数文件
	ParameterReader pd;
	CameraParam camera(atof(pd.getData("baseline").c_str()),atof(pd.getData("factor").c_str()),atof(pd.getData("cx").c_str()),\
			atof(pd.getData("cy").c_str()),atof(pd.getData("fx").c_str()),atof(pd.getData("fy").c_str()));
	string path = pd.getData("path");
	int startIndex = atoi(pd.getData("start_index").c_str());
	int endIndex = atoi(pd.getData("end_index").c_str());
	int lastIndex= startIndex;

	double good_match_threshold = atof(pd.getData("good_match_threshold").c_str());
	int min_good_match = atoi(pd.getData("min_good_match").c_str());

	bool isVisualize=pd.getData("visualize_pointcloud")==string("yes");
	int min_inliers = atoi(pd.getData("min_inliers").c_str());
	double max_norm = atof(pd.getData("max_norm").c_str());

	pcl::visualization::CloudViewer viewer("viewer");

	double gridsize = atof(pd.getData("voxel_grid").c_str());
	int alg = STEREO_BM;
	int SADWindowSize = 0, numberOfDisparities = 0;
	
	//生成初始点云
	FRAME frame1,frame2;//frame1代表前一帧，frame2代表当前帧

	readFrame(frame1,lastIndex,path);
	stereoMatching(frame1,camera,alg,SADWindowSize,numberOfDisparities); 
	computeKeyPointsAndDesp(frame1);	
	PointCloud::Ptr cloud = generatePointCloud(frame1.depthColor,frame1.depth,camera);


	//g2o优化部分
	typedef g2o::BlockSolver_6_3 SlamBlockSolver;
	typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;
	for(int currIndex = startIndex+1;currIndex<endIndex;currIndex++)
	{
		bool isExit = readFrame(frame2,currIndex,path);
		if(isExit==0)
			continue;
		//计算depth和depthColor;
		std::cout<<"读取第"<<currIndex<<"帧"<<std::endl;
		stereoMatching(frame2,camera,alg,SADWindowSize,numberOfDisparities); 
		
		//计算位姿估计
		computeKeyPointsAndDesp(frame2);	
		MOTION motion = motionEstimation(frame1,frame2,camera,good_match_threshold,min_good_match,min_inliers);	

		if(motion.inliers < 0) //判断是否匹配不成功
		{
			std::cout<<"inliers数量不满足"<<std::endl;
			continue;
		}
		double norm = fabs(min(cv::norm(motion.rvec),\
					2*M_PI-cv::norm(motion.rvec)))+fabs(cv::norm(motion.tvec));
		if(norm>=max_norm)//判断是否运动范围太大
		{
			std::cout<<"norm距离不满足"<<std::endl;
			continue;
		}
		//结合点云
		std::cout<<"匹配成功"<<std::endl;
		Eigen::Isometry3d T = cvMat2Eigen(motion);
		cloud = joinPointCloud(cloud,frame2,T,camera,gridsize);

		if(isVisualize==true)
			viewer.showCloud(cloud);

		frame1 = frame2;
	}

	pcl::io::savePCDFile("result.pcd",*cloud);
	std::cout<<"保存成功"<<std::endl;
	return 0;
}

std::string itos(double i)
{
	std::stringstream ss;

	ss << i;

	return ss.str();
}
bool readFrame(FRAME &frame,int index,std::string path)
{
	//读取left和right文件
	std::string leftPath = path+"left_"+itos(index)+".png";
	std::string rightPath = path+"right_"+itos(index)+".png";
	
	fstream _file;
	_file.open(leftPath.c_str(),ios::in);
	if(!_file)
	{
		_file.close();
		std::cout<<leftPath<<"不存在"<<std::endl;
		return 0;
	}
	_file.close();
	_file.open(rightPath.c_str(),ios::in);
	if(!_file)
	{
		_file.close();
		std::cout<<rightPath<<"不存在"<<std::endl;
		return 0;
	}
	_file.close();

	frame.left = cv::imread(leftPath.c_str(), -1);
	frame.right = cv::imread(rightPath.c_str(), -1);
	return 1;
}
