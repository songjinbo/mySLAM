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

	cv::Mat imgMatches;
    cv::drawMatches( frame1.left, frame1.kp, frame2.left, frame2.kp, matches, imgMatches );
    cv::imshow( "matches", imgMatches );
	//cv::waitKey(0);

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
	//if(frame1.frameID == 178)
	{
		cv::drawMatches( frame1.left, frame1.kp, frame2.left, frame2.kp, goodMatches, imgMatches );
		cv::imshow( "good matches", imgMatches );
		cv::waitKey(0);	
	}
	std::vector<cv::Point2f> pts_img1;// 第一个帧的三维点
	std::vector<cv::Point2f> pts_img2;// 第二个帧的图像点
    for (size_t i=0; i<goodMatches.size(); i++)
    {
        pts_img1.push_back( cv::Point2f( frame1.kp[goodMatches[i].trainIdx].pt ) );
        pts_img2.push_back( cv::Point2f( frame2.kp[goodMatches[i].trainIdx].pt ) );
    }
    // 相机内参
    double camera_matrix_data[3][3] = {
        {camera.camera_fx, 0, camera.camera_cx},
        {0, camera.camera_fy, camera.camera_cy},
        {0, 0, 1}
    };
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
	cv::Mat E, mask;
    E = findEssentialMat(pts_img1, pts_img2, cameraMatrix, cv::RANSAC, 0.999, 1.0, mask);
    recoverPose(E, pts_img1, pts_img2, cameraMatrix, rvec, tvec, mask);

	std::cout<<rvec<<std::endl;
	std::cout<<tvec<<std::endl;
	std::cout<<"the norm of tvec is : "<<norm(tvec,CV_L2)<<std::endl;
	cv::waitKey(0);

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
