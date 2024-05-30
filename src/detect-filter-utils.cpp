#include <opencv2/opencv.hpp>
using namespace cv;

void drawDashedLine(Mat &img, Point pt1, Point pt2, Scalar color, int thickness, int lineType,
		    int dashLength)
{
	double lineLength = norm(pt1 - pt2);
	double angle = atan2(pt2.y - pt1.y, pt2.x - pt1.x);

	Point p1 = pt1;
	Point p2;
	bool draw = true;

	for (double d = 0; d < lineLength; d += dashLength) {
		if (draw) {
			p2.x = pt1.x +
			       static_cast<int>(cos(angle) * std::min(d + dashLength, lineLength));
			p2.y = pt1.y +
			       static_cast<int>(sin(angle) * std::min(d + dashLength, lineLength));
			line(img, p1, p2, color, thickness, lineType);
		}
		p1.x = pt1.x + static_cast<int>(cos(angle) * (d + dashLength));
		p1.y = pt1.y + static_cast<int>(sin(angle) * (d + dashLength));
		draw = !draw;
	}
}

void drawDashedRectangle(Mat &img, Rect rect, Scalar color, int thickness, int lineType,
			 int dashLength)
{
	Point pt1(rect.x, rect.y);
	Point pt2(rect.x + rect.width, rect.y);
	Point pt3(rect.x + rect.width, rect.y + rect.height);
	Point pt4(rect.x, rect.y + rect.height);

	drawDashedLine(img, pt1, pt2, color, thickness, lineType, dashLength);
	drawDashedLine(img, pt2, pt3, color, thickness, lineType, dashLength);
	drawDashedLine(img, pt3, pt4, color, thickness, lineType, dashLength);
	drawDashedLine(img, pt4, pt1, color, thickness, lineType, dashLength);
}
