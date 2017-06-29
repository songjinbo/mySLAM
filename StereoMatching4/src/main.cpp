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
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

#include "generatePointCloud.hpp"
#include "stereoMatching.hpp"
#include "readParam.hpp"
#include "motionEstimation.hpp"

bool readFrame(FRAME &frame,int index,std::string path,ofstream&);
/*
enum CHECK_RESULT
{
	NOT_MATCHED = 0,
	TOO_FAR_AWAY,
	TOO_CLOSE,
	KEYFRAME
};
//检测是否是关键帧
CHECK_RESULT checkKeyFrames(FRAME &f1,FRAME &f2,g2o::SparseOptimizer &opti,\
		bool is_loops,CameraParam &camera);
//检测近距离回环
void checkNearbyLoops(vector<FRAME> &frames,FRAME &currFrame,g2o::SparseOptimizer &opti,int nearby_loops,\
		CameraParam &camera);
//检测近随机回环
void checkRandomLoops(vector<FRAME> &frames,FRAME& currFrame,g2o::SparseOptimizer &opti,int random_loops,\
		CameraParam &camera);
*/

int main()
{
	time_t start,stop;
	start = time(NULL);
	ofstream out("out.txt");
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
	double keyframe_threshold = atof(pd.getData("keyframe_threshold").c_str());
	bool check_loop_closure = pd.getData("check_loop_closure")==string("yes");
	int random_loops = atoi(pd.getData("random_loops").c_str());
	int nearby_loops = atoi(pd.getData("nearby_loops").c_str());
	pcl::visualization::CloudViewer viewer("viewer");

	double gridsize = atof(pd.getData("voxel_grid").c_str());
	int alg = STEREO_BM;
	int SADWindowSize = 0, numberOfDisparities = 0;
	
	//生成初始点云
	FRAME lastFrame,currFrame;//lastFrame代表前一帧，currFrame代表当前帧
	readFrame(lastFrame,lastIndex,path,out);
	stereoMatching(lastFrame,camera,alg,SADWindowSize,numberOfDisparities); 
	computeKeyPointsAndDesp(lastFrame);	
    //cv::imwrite( "./data/keypoints.png", imgShow );
	system("pause");

	PointCloud::Ptr cloud = generatePointCloud(lastFrame.depthColor,lastFrame.depth,camera);

	//实时处理
	for(int currIndex = startIndex+1;currIndex<endIndex;currIndex++)
	{
		bool isExit = readFrame(currFrame,currIndex,path,out);
		if(isExit==0)
			continue;
		//计算depth和depthColor;
		std::cout<<"读取第"<<currIndex<<"帧"<<std::endl;
		out<<"读取第"<<currIndex<<"帧"<<std::endl;
		stereoMatching(currFrame,camera,alg,SADWindowSize,numberOfDisparities); 
		//计算位姿估计
		computeKeyPointsAndDesp(currFrame);	
		MOTION result = motionEstimation(lastFrame,currFrame,camera,good_match_threshold,\
				min_good_match,min_inliers,out);
		if(result.inliers==-1)
		{
			std::cout<<"good matches太少"<<std::endl;
			out<<"good matches太少"<<std::endl;
			continue;
		}
		else if(result.inliers==-2)
		{
			std::cout<<"inliers太少"<<std::endl;
			out<<"inliers太少"<<std::endl;
			continue;
		}
		double norm = fabs(min(cv::norm(result.rvec), 2*M_PI-cv::norm(result.rvec)))+ fabs(cv::norm(result.tvec));
		std::cout<<"norm = "<<norm<<std::endl;
		out<<"norm = "<<norm<<std::endl;
		if(norm>=max_norm)
		{
			std::cout<<"两帧距离太远"<<std::endl;
			out<<"两帧距离太远"<<std::endl;
			continue;
		}
		else if(norm<=keyframe_threshold)
		{
			std::cout<<"不是关键帧"<<std::endl;
			out<<"不是关键帧"<<std::endl;
			continue;
		}
		Eigen::Isometry3d T = cvMat2Eigen(result);
		std::cout<<"T = "<<T.matrix()<<std::endl;
		out<<"T = "<<T.matrix()<<std::endl;
				
		std::ofstream fout("../data/path.txt",std::ios::app);
		fout<<T.matrix()<<std::endl;
		fout.close();

		cloud = joinPointCloud(cloud,currFrame,T,camera,gridsize);
/*
		if(isVisualize = true)
			viewer.showCloud(cloud);
*/
		lastFrame = currFrame;
	}
		
	pcl::io::savePCDFile("result.pcd",*cloud);
	std::cout<<"saving succeed!"<<std::endl;
	stop = time(NULL);
	std::cout<<"共花费"<<stop-start<<"秒"<<std::endl;
	return 0;
}

std::string itos(double i)
{
	std::stringstream ss;

	ss << i;

	return ss.str();
}
bool readFrame(FRAME &frame,int index,std::string path,ofstream &out)
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
		out<<leftPath<<"不存在"<<std::endl;
		return 0;
	}
	_file.close();
	_file.open(rightPath.c_str(),ios::in);
	if(!_file)
	{
		_file.close();
		std::cout<<rightPath<<"不存在"<<std::endl;
		out<<rightPath<<"不存在"<<std::endl;
		return 0;
	}
	_file.close();

	frame.left = cv::imread(leftPath.c_str(), -1);
	frame.right = cv::imread(rightPath.c_str(), -1);
	frame.frameID = index;
	return 1;
}
