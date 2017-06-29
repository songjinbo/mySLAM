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

typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

bool readFrame(FRAME &frame,int index,std::string path,ofstream&);

enum CHECK_RESULT
{
	NOT_MATCHED = 0,
	TOO_FAR_AWAY,
	TOO_CLOSE,
	KEYFRAME
};
//检测是否是关键帧
CHECK_RESULT checkKeyFrames(FRAME &f1,FRAME &f2,g2o::SparseOptimizer &opti,\
		bool is_loops,CameraParam &camera,ofstream&);
//检测近距离回环
void checkNearbyLoops(vector<FRAME> &frames,FRAME &currFrame,g2o::SparseOptimizer &opti,int nearby_loops,\
		CameraParam &camera,ofstream&);
//检测近随机回环
void checkRandomLoops(vector<FRAME> &frames,FRAME& currFrame,g2o::SparseOptimizer &opti,int random_loops,\
		CameraParam &camera,ofstream&);

int main()
{
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
//	pcl::visualization::CloudViewer viewer("viewer");

	double gridsize = atof(pd.getData("voxel_grid").c_str());
	int alg = STEREO_BM;
	int SADWindowSize = 0, numberOfDisparities = 0;
	
	//生成初始点云
	FRAME lastFrame,currFrame;//lastFrame代表前一帧，currFrame代表当前帧
	readFrame(lastFrame,lastIndex,path,out);
	stereoMatching(lastFrame,camera,alg,SADWindowSize,numberOfDisparities); 
	computeKeyPointsAndDesp(lastFrame);	
	PointCloud::Ptr cloud = generatePointCloud(lastFrame.depthColor,lastFrame.depth,camera);

	//g2o优化部分
	SlamLinearSolver *linearSolver = new SlamLinearSolver();
	linearSolver->setBlockOrdering(false);
	SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
	g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);

	g2o::SparseOptimizer globalOptimizer;
	globalOptimizer.setAlgorithm(solver);
	globalOptimizer.setVerbose(false);//不输出调试信息
	//向globalOptimizer增加一个顶点
	g2o::VertexSE3* v = new g2o::VertexSE3();
	v->setId(lastIndex);
	v->setEstimate(Eigen::Isometry3d::Identity());
	v->setFixed(true);
	globalOptimizer.addVertex(v);
	
	vector<FRAME> keyframes;
	keyframes.push_back(lastFrame);

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
		CHECK_RESULT result = checkKeyFrames(keyframes.back(),currFrame,globalOptimizer,false,camera,out);
		switch(result)
		{
			case NOT_MATCHED:
				std::cout<<"Not enough inliers."<<std::endl;
				out<<"Not enough inliers."<<std::endl;
				break;
			case TOO_FAR_AWAY:
				std::cout<<"Too far away,may be an error."<<std::endl;
				out<<"Too far away,may be an error."<<std::endl;
				break;
			case TOO_CLOSE:
				std::cout<<"Too close,not a keyframe."<<std::endl;
				out<<"Too close,not a keyframe."<<std::endl;
				break;
			case KEYFRAME:
				std::cout<<"This is a keyframe."<<std::endl;
				out<<"This is a keyframe."<<std::endl;
				if(check_loop_closure)
				{
					checkNearbyLoops(keyframes,currFrame,globalOptimizer,nearby_loops,camera,out);
					checkRandomLoops(keyframes,currFrame,globalOptimizer,random_loops,camera,out);
				}
				keyframes.push_back(currFrame);
				break;
			default:
				break;
		}
	}
		
	std::cout<<"optimizing pose graph, vertices: "<<globalOptimizer.vertices().size()<<std::endl;
    out<<"optimizing pose graph, vertices: "<<globalOptimizer.vertices().size()<<std::endl;
    globalOptimizer.save("./data/result_before.g2o");
	globalOptimizer.initializeOptimization();
	globalOptimizer.optimize(100);
	globalOptimizer.save("./data/result_before.g2o");
	std::cout<<"Optimization done."<<std::endl;
	out<<"Optimization done."<<std::endl;
	//拼接点云图
	std::cout<<"saving the point cloud map..."<<std::endl;
	out<<"saving the point cloud map..."<<std::endl;
	PointCloud::Ptr output(new PointCloud());//全局地图
	PointCloud::Ptr tmp(new PointCloud());
	pcl::VoxelGrid<PointT> voxel;//网格滤波器，调整地图分辨率
	pcl::PassThrough<PointT> pass;

	pass.setFilterFieldName("z");
	pass.setFilterLimits(0.0,4.0);

	voxel.setLeafSize(gridsize,gridsize,gridsize);

	for(size_t i = 0;i<keyframes.size();i++)
	{
		g2o::VertexSE3 *vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex(keyframes[i].frameID));
		Eigen::Isometry3d pose = vertex->estimate();
		PointCloud::Ptr newCloud = generatePointCloud(keyframes[i].depthColor,keyframes[i].depth,camera);

		voxel.setInputCloud(newCloud);
		voxel.filter(*tmp);
		pass.setInputCloud(tmp);
		
		pass.filter(*newCloud);
		pcl::transformPointCloud(*newCloud,*tmp,pose.matrix());

		*output += *tmp;
		tmp->clear();
		newCloud->clear();
	}

	voxel.setInputCloud(output);
	voxel.filter(*tmp);

	pcl::io::savePCDFile("result.pcd",*tmp);
	std::cout<<"Final map is saved."<<std::endl;
	out<<"Final map is saved."<<std::endl;
	out.close();

	globalOptimizer.clear();
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

CHECK_RESULT checkKeyFrames(FRAME &f1,FRAME &f2,g2o::SparseOptimizer &opti,\
		bool is_loops,CameraParam &camera,ofstream &out)
{
	static ParameterReader pd;
	static int min_inliers = atoi(pd.getData("min_inliers").c_str());
	static int min_good_match = atoi(pd.getData("min_good_match").c_str());
	static double max_norm = atof(pd.getData("max_norm").c_str());
	static double keyframe_threshold = atof(pd.getData("keyframe_threshold").c_str());
	static double max_norm_lp = atof(pd.getData("max_norm_lp").c_str());
	static double good_match_threshold = atof(pd.getData("good_match_threshold").c_str());
	static g2o::RobustKernel* robustKernel = g2o::RobustKernelFactory::instance()->construct("Cauchy");

	MOTION result = motionEstimation(f1,f2,camera,good_match_threshold,min_good_match,min_inliers,out);	
	if(result.inliers < 0)
		return NOT_MATCHED;
	double norm = fabs(min(cv::norm(result.rvec),\
				2*M_PI-cv::norm(result.rvec)))+fabs(cv::norm(result.tvec));
	if(is_loops == false)
	{
		if(norm>=max_norm)
			return TOO_FAR_AWAY;
	}
	else
	{
		if(norm>=max_norm_lp)
			return TOO_FAR_AWAY;
	}

	if(norm<=keyframe_threshold)
		return TOO_CLOSE;

	if(is_loops == false)
	{
		g2o::VertexSE3 *v = new g2o::VertexSE3();
		v->setId(f2.frameID);
		v->setEstimate(Eigen::Isometry3d::Identity());
		opti.addVertex(v);
	}
	
	g2o::EdgeSE3* edge = new g2o::EdgeSE3();

	edge->vertices()[0] = opti.vertex(f1.frameID);
	edge->vertices()[1] = opti.vertex(f2.frameID);
	edge->setRobustKernel(robustKernel);

	Eigen::Matrix<double ,6,6> information = Eigen::Matrix<double,6,6>::Identity();

	information(0,0) = information(1,1) = information(2,2) = 100;
	information(3,3) = information(4,4) = information(5,5) = 100;

	edge->setInformation(information);
	Eigen::Isometry3d T = cvMat2Eigen(result);

	edge->setMeasurement(T.inverse());

	opti.addEdge(edge);
	return KEYFRAME;
}

void checkNearbyLoops(vector<FRAME> &frames,FRAME &currFrame,g2o::SparseOptimizer &opti,int nearby_loops,\
		CameraParam &camera,ofstream &out)
{
	if(frames.size()<=nearby_loops)
	{
		for(size_t i = 0;i<frames.size();i++)
		{
			checkKeyFrames(frames[i],currFrame,opti,true,camera,out);
		}
	}

	else
	{
		for(size_t i = frames.size()-nearby_loops;i<frames.size();i++)
		{
			checkKeyFrames(frames[i],currFrame,opti,true,camera,out);
		}
	}
}

void checkRandomLoops(vector<FRAME> &frames,FRAME& currFrame,g2o::SparseOptimizer &opti,int random_loops,\
		CameraParam &camera,ofstream &out)
{
	srand((unsigned int) time(NULL));

	if(frames.size()<=random_loops)
	{
		for(size_t i = 0;i<frames.size();i++)
		{
			checkKeyFrames(frames[i],currFrame,opti,true,camera,out);
		}
	}
	else
	{
		for(int i = 0;i<random_loops;i++)
		{
			int index = rand()%frames.size();
			checkKeyFrames(frames[index],currFrame,opti,true,camera,out);
		}
	}
}
