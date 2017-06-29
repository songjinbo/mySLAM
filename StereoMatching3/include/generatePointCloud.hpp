#pragma once

#include <pcl-1.7/pcl/io/pcd_io.h>
#include <pcl-1.7/pcl/point_cloud.h>
#include <pcl-1.7/pcl/impl/point_types.hpp>
#include <eigen3/Eigen/src/Geometry/Transform.h>

struct MOTION;
struct FRAME;

namespace cv
{
	class Mat;
}

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

class CameraParam
{
public:
	const double baseline;
	const double camera_factor; 
	const double camera_cx;
	const double camera_cy;
	const double camera_fx;
	const double camera_fy;
public:
	CameraParam(double x1,double x2,double x3,double x4,double x5,double x6)\
		:baseline(x1),camera_factor(x2),camera_cx(x3),camera_cy(x4),camera_fx(x5),camera_fy(x6)
		{}
};

PointCloud::Ptr generatePointCloud(cv::Mat & ,cv::Mat &,CameraParam &);

Eigen::Isometry3d cvMat2Eigen(MOTION& motion);

PointCloud::Ptr joinPointCloud(PointCloud::Ptr original,FRAME &newFrame,\
Eigen::Isometry3d T,CameraParam& camera,double);
