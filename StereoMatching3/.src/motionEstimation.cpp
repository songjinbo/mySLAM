#include "motionEstimation.hpp"

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "generatePointCloud.hpp"

//#include "generatePointCloud.hpp"

cv::Point3f point2dTo3d( cv::Point3f& point, CameraParam& camera )
{
    cv::Point3f p; // 3D 点
    p.z = double( point.z ) / camera.camera_factor;
    p.x = ( point.x - camera.camera_cx) * p.z / camera.camera_fx;
    p.y = ( point.y - camera.camera_cy) * p.z / camera.camera_fy;
    return p;
}

void computeKeyPointsAndDesp(FRAME& frame)
{
	cv::Ptr<cv::ORB> detectorAndDescriptor = cv::ORB::create();
	detectorAndDescriptor->detectAndCompute(frame.left,cv::noArray(),frame.kp,frame.desp);
}

MOTION motionEstimation(FRAME &frame1,FRAME &frame2,CameraParam& camera,\
		double good_match_threshold,int min_good_match,int min_inliers)
{
	if(!(frame1.depth.depth()==5&&frame1.depth.channels()==1))
		assert(0);

    MOTION result;
	std::vector< cv::DMatch > matches;
    cv::BFMatcher matcher;
    matcher.match( frame1.desp, frame2.desp, matches );
	std::cout<<"find total "<<matches.size()<<" matches."<<std::endl;

	std::vector< cv::DMatch > goodMatches;
    double minDis = 9999;
    for ( size_t i=0; i<matches.size(); i++ )
    {
		if(matches[i].distance == 0)
			continue;
        else if( matches[i].distance < minDis )
			minDis = matches[i].distance;
    }
	std::cout<<"minDis: "<<minDis<<std::endl;
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < good_match_threshold*minDis)
            goodMatches.push_back( matches[i] );
    }
	std::cout<<"good matches: "<<goodMatches.size()<<std::endl;
	if(goodMatches.size() < min_good_match)
	{
		result.inliers = -1;
		return result;
	}
	
	std::vector<cv::Point3f> pts_obj;// 第一个帧的三维点
	std::vector< cv::Point2f > pts_img;// 第二个帧的图像点
    for (size_t i=0; i<goodMatches.size(); i++)
    {
        // query 是第一个, train 是第二个
        cv::Point2f p = frame1.kp[goodMatches[i].queryIdx].pt;
        // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
        ushort d = frame1.depth.ptr<float>( int(p.y) )[ int(p.x) ];
        if (d == 0)
            continue;
        pts_img.push_back( cv::Point2f( frame2.kp[goodMatches[i].trainIdx].pt ) );

        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, camera );
        pts_obj.push_back( pd );
    }

    // 相机内参
    double camera_matrix_data[3][3] = {
        {camera.camera_fx, 0, camera.camera_cx},
        {0, camera.camera_fy, camera.camera_cy},
        {0, 0, 1}
    };
	std::cout<<"solving pnp"<<std::endl;
    // 构建相机矩阵
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    // 求解pnp
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec,false,100,8.0,0.99,inliers);

    result.rvec = rvec;
    result.tvec = tvec;
    result.inliers = inliers.rows;

	if(result.inliers < min_inliers)
	{
		result.inliers = -2;
		return result;
	}		
    return result;
}
