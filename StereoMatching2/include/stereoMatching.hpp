#pragma once

#include <opencv2/core/mat.hpp>

struct FRAME;
struct CameraParam;
enum { STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2}; 
cv::Mat stereoMatching(FRAME &frame,CameraParam &camera,\
		int alg,int SADWindowSize,int numberOfDisparities);
