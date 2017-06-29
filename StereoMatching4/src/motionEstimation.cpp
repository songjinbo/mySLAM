#include "motionEstimation.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>//用来检测SIFT特征
#include <opencv2/highgui.hpp>

#include "generatePointCloud.hpp"

//#include "generatePointCloud.hpp"

std::string itos(double i);
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
//	cv::Ptr<cv::ORB> detectorAndDescriptor = cv::ORB::create();
//	detectorAndDescriptor->detectAndCompute(frame.left,cv::noArray(),frame.kp,frame.desp);
	cv::Ptr<cv::xfeatures2d::SIFT> feature = cv::xfeatures2d::SIFT::create();
	feature->detectAndCompute(frame.left,cv::noArray(),frame.kp,frame.desp);
	std::cout<<"特征点数量： "<<frame.kp.size()<<std::endl;
//	cv::Ptr<cv::xfeatures2d::SURF> feature = cv::xfeatures2d::SURF::create();
//	feature->detectAndCompute(frame.left,cv::noArray(),frame.kp,frame.desp);
}

MOTION motionEstimation(FRAME &frame1,FRAME &frame2,CameraParam& camera,\
		double good_match_threshold,int min_good_match,int min_inliers,std::ofstream& out)
{
	std::cout<<"两帧ID分别为： "<<frame1.frameID<<" , "<<frame2.frameID<<std::endl;		
	if(!(frame1.depth.depth()==5&&frame1.depth.channels()==1))
		assert(0);

    MOTION result;
	std::vector< cv::DMatch > matches;
    cv::BFMatcher matcher;
    matcher.match( frame1.desp, frame2.desp, matches );
//	std::cout<<"find total "<<matches.size()<<" matches."<<std::endl;
	/*
	cv::Mat imgMatches;
    cv::drawMatches( frame1.left, frame1.kp, frame2.left, frame2.kp, matches, imgMatches );
    cv::imshow( "matches", imgMatches );
	cv::waitKey(0);
	*/
	std::vector< cv::DMatch > goodMatches;
    double minDis = 9999;
    for ( size_t i=0; i<matches.size(); i++ )
    {
		if(matches[i].distance == 0)
			continue;
        else if( matches[i].distance < minDis )
			minDis = matches[i].distance;
    }
//	std::cout<<"minDis: "<<minDis<<std::endl;
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if (matches[i].distance < good_match_threshold*minDis)
            goodMatches.push_back( matches[i] );
    }
	std::cout<<"the size of good matches is : "<<goodMatches.size()<<std::endl;
	/*
	if(frame1.frameID == 178)
	{
		cv::drawMatches( frame1.left, frame1.kp, frame2.left, frame2.kp, goodMatches, imgMatches );
		cv::imshow( "good matches", imgMatches );
		cv::waitKey(0);	
	}
	*/
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
	std::cout<<"the size of pts_obj is : "<<pts_obj.size()<<std::endl;
	out<<"the size of pts_obj is : "<<pts_obj.size()<<std::endl;
	if(pts_obj.size() < min_good_match)
	{
		result.inliers = -1;
		return result;
	}
    // 相机内参
    double camera_matrix_data[3][3] = {
        {camera.camera_fx, 0, camera.camera_cx},
        {0, camera.camera_fy, camera.camera_cy},
        {0, 0, 1}
    };
//	std::cout<<"solving pnp"<<std::endl;
    // 构建相机矩阵
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    // 求解pnp
	
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec,false,100,2.0,0.90,inliers);
    result.rvec = rvec;
    result.tvec = tvec;
    result.inliers = inliers.rows;
	std::cout<<"the number of inliers is : "<<result.inliers<<std::endl;
	out<<"the number of inliers is : "<<result.inliers<<std::endl;
	if(result.inliers < min_inliers)
	{
		result.inliers = -2;
		return result;
	}		

    return result;
}
