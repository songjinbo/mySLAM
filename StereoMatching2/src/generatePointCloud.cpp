#include "generatePointCloud.hpp"

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <pcl-1.7/pcl/io/pcd_io.h>
#include <pcl-1.7/pcl/visualization/cloud_viewer.h>
#include <pcl-1.7/pcl/common/transforms.h>
#include <pcl-1.7/pcl/filters/voxel_grid.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include "motionEstimation.hpp"
#include "readParam.hpp"

PointCloud::Ptr generatePointCloud(cv::Mat &rgb,cv::Mat &depth ,CameraParam& camera)
{
	if(!(rgb.depth()==0 && rgb.channels()==3 && depth.depth()==5 && depth.channels() == 1))
		assert(0);
	PointCloud::Ptr cloud(new PointCloud);
	for(int m = 0;m<depth.rows;m++)
		for(int n=0;n<depth.cols;n++)
		{
			float d=depth.at<float>(m,n);
			if(d==0)
				continue;
			PointT p;
			p.z=double(d)/camera.camera_factor;
			p.x=(n-camera.camera_cx)*p.z/camera.camera_fx;
			p.y=(m-camera.camera_cy)*p.z/camera.camera_fy;

			p.b=rgb.at<cv::Vec3b>(m,n)[0];
			p.g=rgb.at<cv::Vec3b>(m,n)[1];
			p.r=rgb.at<cv::Vec3b>(m,n)[2];

			cloud->points.push_back(p);
		}

	cloud->height=1;
	cloud->width=cloud->points.size();
	cloud->is_dense=false;
	return cloud;
}

Eigen::Isometry3d cvMat2Eigen(MOTION& motion)
{
    cv::Mat R;
    cv::Rodrigues( motion.rvec, R );
    Eigen::Matrix3d r;
    cv::cv2eigen(R, r);

	//将平移向量和旋转矩阵转换成变换矩阵
	Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
	Eigen::AngleAxisd angle(r);
	T = angle;
	T(0,3) = motion.tvec.at<double>(0,0);
	T(1,3) = motion.tvec.at<double>(0,1);
	T(2,3) = motion.tvec.at<double>(0,2);

	return T;
}

PointCloud::Ptr joinPointCloud(PointCloud::Ptr original,FRAME &newFrame,\
Eigen::Isometry3d T,CameraParam& camera,double gridsize)
{
	PointCloud::Ptr newCloud = generatePointCloud(newFrame.depthColor,newFrame.depth,camera);

	PointCloud::Ptr output(new PointCloud());
	pcl::transformPointCloud(*original,*output,T.matrix());
	*newCloud += *output;

	static pcl::VoxelGrid<PointT> voxel;
	static ParameterReader pd;
	voxel.setLeafSize(gridsize,gridsize,gridsize);
	voxel.setInputCloud(newCloud);
	PointCloud::Ptr tmp(new PointCloud);
	voxel.filter(*tmp);
	return tmp;
}
