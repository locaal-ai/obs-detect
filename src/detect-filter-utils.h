#ifndef DETECT_FILTER_UTILS_H
#define DETECT_FILTER_UTILS_H

#include <opencv2/core/types.hpp>

void drawDashedLine(cv::Mat &img, cv::Point pt1, cv::Point pt2, cv::Scalar color, int thickness = 1,
		    int lineType = 8, int dashLength = 10);

void drawDashedRectangle(cv::Mat &img, cv::Rect rect, cv::Scalar color, int thickness = 1,
			 int lineType = 8, int dashLength = 10);

#endif // DETECT_FILTER_UTILS_H
