#include "stereoMatching.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "motionEstimation.hpp"
#include "generatePointCloud.hpp"

#define img_b(img,y,x) img.at<cv::Vec3b>(y,x)[0]
#define img_g(img,y,x) img.at<cv::Vec3b>(y,x)[1]
#define img_r(img,y,x) img.at<cv::Vec3b>(y,x)[2]

void Pseudocolor(cv::Mat &img8u,cv::Mat &img_color)
{
	if(img8u.depth()==0 && img8u.channels()==1 &&\
			img_color.depth()==5 && img_color.channels()==3)
		assert(0);
	uchar tmp= 0;
	for(int y =0;y<img8u.rows;y++)
	{
		for(int x = 0;x<img8u.cols;x++)	
		{
			tmp = img8u.at<uchar>(y,x);
			if(tmp<=51)
			{
				img_b(img_color,y,x) = 255;
				img_g(img_color,y,x) = tmp*5;
				img_r(img_color,y,x) = 0;
			}
			else if(tmp<=102)
			{
				tmp -=51;
			    img_b(img_color,y,x) = 255-tmp*5;
				img_g(img_color,y,x) = 255;
				img_r(img_color,y,x) = 0;
			}
			else if(tmp<=153)
			{
				tmp-=102;
				img_b(img_color,y,x) = 0;
				img_g(img_color,y,x) = 255;
				img_r(img_color,y,x) = tmp*5;
			}
			else if(tmp<=204)
			{
				tmp -= 153;
				img_b(img_color,y,x) = 0;
				img_g(img_color,y,x) = 255-uchar(128.0*tmp/51.0+0.5);
				img_r(img_color,y,x) = 255;
			}
			else
			{
				tmp -= 204;
				img_b(img_color,y,x) = 0;
				img_g(img_color,y,x) = 127-uchar(127.0*tmp/51.0+0.5);
				img_r(img_color,y,x) = 255;
			}
		}
	}
}

cv::Mat stereoMatching(FRAME &frame,CameraParam &camera,\
		int alg,int SADWindowSize,int numberOfDisparities)
{
	cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16,9);
	numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((frame.left.size().width / 8) + 15) & -16;
	cv::Rect roi1, roi2;
    bm->setROI1(roi1);
    bm->setROI2(roi2);
    bm->setPreFilterCap(63);
    bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
    bm->setMinDisparity(0);
    bm->setNumDisparities(numberOfDisparities);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(15);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(1);

	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create();
	int cn = frame.left.channels();
	sgbm->setPreFilterCap(63);
	sgbm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 5);
    sgbm->setP1(8 * cn*sgbm->getBlockSize()*sgbm->getBlockSize());
    sgbm->setP2(32 * cn*sgbm->getBlockSize()*sgbm->getBlockSize());
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(16);
//	std::cout<<sgbm->getNumDisparities()<<std::endl;
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(bm->getSpeckleWindowSize());
    sgbm->setSpeckleRange(bm->getSpeckleRange());
    sgbm->setDisp12MaxDiff(1);
    if(alg==STEREO_HH)
        sgbm->setMode(cv::StereoSGBM::MODE_HH);
    else if(alg==STEREO_SGBM)
        sgbm->setMode(cv::StereoSGBM::MODE_SGBM);

	cv::Mat disp;
	int64 t = cv::getTickCount();
	if (alg == STEREO_BM)
		bm->compute(frame.left, frame.right, disp);
	else if (alg == STEREO_SGBM || alg == STEREO_HH)
		sgbm->compute(frame.left,frame.right, disp);//------

	t = cv::getTickCount() - t;
	//printf("Time elapsed: %fms\n", t * 1000 / cv::getTickFrequency());

	cv::Mat depth(frame.left.rows,frame.left.cols,CV_32F);
	for(int y = 0;y<disp.rows;y++)
	{
		for(int x = 0;x<disp.cols;x++)
		{
			float tmp = disp.at<short>(y,x);
			if(tmp<=0)
			{
				depth.at<float>(y,x) = 0;
			}
			else 
			{
				depth.at<float>(y,x) = camera.camera_fx*camera.baseline/tmp;
			}
		}
	}
	frame.depth = depth;
	cv::Mat depth8;	
	cv::Mat depth_color(frame.left.rows,frame.left.cols,CV_8UC3);
	depth.convertTo(depth8,CV_8U);
	Pseudocolor(depth8,depth_color);
//	frame.depthColor = frame.left;
			
	return disp;
}
