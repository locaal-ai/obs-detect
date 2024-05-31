#ifndef EDGEYOLO_TYPES_HPP
#define EDGEYOLO_TYPES_HPP

#include <opencv2/core/types.hpp>
#include <opencv2/video/tracking.hpp>

#ifdef _WIN32
#define file_name_t std::wstring
#else
#define file_name_t std::string
#endif

struct Object {
	cv::Rect_<float> rect;
	int label;
	float prob;
	uint64_t id;
	uint64_t unseenFrames;
	cv::KalmanFilter kf;
};

struct GridAndStride {
	int grid0;
	int grid1;
	int stride;
};

#endif
