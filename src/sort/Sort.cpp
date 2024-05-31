#include "Sort.h"

#include "sort/munkres-cpp/matrix_base.h"
#include "sort/munkres-cpp/adapters/matrix_std_2d_vector.h"
#include "sort/munkres-cpp/munkres.h"

#include <cmath>
#include <limits>
#include <algorithm>

#include "plugin-support.h"

#include <obs.h>

#define INF std::numeric_limits<float>::infinity()

// Constructor
Sort::Sort(size_t maxUnseenFrames_) : nextTrackID(0), maxUnseenFrames(maxUnseenFrames_) {}

// Destructor
Sort::~Sort() {}

// Initialize the Kalman filter for a new object
void Sort::initializeKalmanFilter(cv::KalmanFilter &kf, const cv::Rect_<float> &bbox)
{
	// Linear motion model with dimension: [x, y, width, height, dx, dy, dwidth, dheight]
	kf.init(8, 4, 0);

	// State vector: [x, y, width, height, dx, dy, dwidth, dheight]
	kf.statePre.at<float>(0) = bbox.x;
	kf.statePre.at<float>(1) = bbox.y;
	kf.statePre.at<float>(2) = bbox.width;
	kf.statePre.at<float>(3) = bbox.height;

	// Transition matrix
	kf.transitionMatrix = (cv::Mat_<float>(8, 8) << 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
			       0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
			       0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			       0, 0, 1);

	// Measurement matrix
	kf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
	for (int i = 0; i < 4; ++i) {
		kf.measurementMatrix.at<float>(i, i) = 1;
	}

	// Process noise covariance matrix
	const float q = 1e-1f;
	kf.processNoiseCov = (cv::Mat_<float>(8, 8) << 0.25, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.25, 0, 0,
			      0, 0.5, 0, 0, 0, 0, 0.25, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.25, 0, 0, 0,
			      0.5, 0.5, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0,
			      0, 0, 1, 0, 0, 0, 0, 0.5, 0, 0, 0, 1) *
			     q;

	// Measurement noise covariance matrix
	kf.measurementNoiseCov =
		(cv::Mat_<float>(4, 4) << 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 10, 0, 0, 0, 0, 10);

	// Error covariance matrix
	cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1e3));

	// Correct the state vector with the initial measurement
	cv::Mat measurement(4, 1, CV_32F);
	measurement.at<float>(0) = bbox.x;
	measurement.at<float>(1) = bbox.y;
	measurement.at<float>(2) = bbox.width;
	measurement.at<float>(3) = bbox.height;
	kf.correct(measurement);
}

// Predict the next state of the object
cv::Rect_<float> Sort::predict(cv::KalmanFilter &kf)
{
	cv::Mat prediction = kf.predict();
	return cv::Rect_<float>(prediction.at<float>(0), prediction.at<float>(1),
				prediction.at<float>(2), prediction.at<float>(3));
}

// Update the Kalman filter with a new measurement
cv::Rect_<float> Sort::updateKalmanFilter(cv::KalmanFilter &kf, const cv::Rect_<float> &bbox)
{
	cv::Mat measurement(4, 1, CV_32F);
	measurement.at<float>(0) = bbox.x;
	measurement.at<float>(1) = bbox.y;
	measurement.at<float>(2) = bbox.width;
	measurement.at<float>(3) = bbox.height;
	const auto corrected = kf.correct(measurement);
	return cv::Rect_<float>(corrected.at<float>(0), corrected.at<float>(1),
				corrected.at<float>(2), corrected.at<float>(3));
}

// Compute the Intersection over Union (IoU) between two rectangles
float computeIoU(const cv::Rect_<float> &rect1, const cv::Rect_<float> &rect2)
{
	float intersectionArea = (rect1 & rect2).area();
	float unionArea = rect1.area() + rect2.area() - intersectionArea;
	return intersectionArea / unionArea;
}

// Update the tracking with detected objects
std::vector<Object> Sort::update(const std::vector<Object> &detections)
{
	if (detections.empty()) {
		std::vector<Object> newTrackedObjects;

		// No detections, predict the next state of the existing tracks and update unseen frames
		for (size_t i = 0; i < trackedObjects.size(); ++i) {
			trackedObjects[i].rect = predict(trackedObjects[i].kf);
			trackedObjects[i].unseenFrames++;

			// Remove lost tracks
			if (trackedObjects[i].unseenFrames < this->maxUnseenFrames) {
				newTrackedObjects.push_back(trackedObjects[i]);
			}
		}
		trackedObjects = newTrackedObjects;
		return trackedObjects;
	}

	if (trackedObjects.empty()) {
		// No existing tracks, create new tracks for all detections
		for (const auto &detection : detections) {
			cv::KalmanFilter kf;
			initializeKalmanFilter(kf, detection.rect);
			trackedObjects.push_back(detection);
			trackedObjects.back().kf = kf; // store the Kalman filter in the object
			trackedObjects.back().id = nextTrackID++;
			trackedObjects.back().unseenFrames = 0;
		}
		return trackedObjects;
	}

	// Predict new locations of existing tracked objects
	for (size_t i = 0; i < trackedObjects.size(); ++i) {
		trackedObjects[i].rect = predict(trackedObjects[i].kf);
	}

	// Build the cost matrix for the Hungarian algorithm
	size_t numDetections = detections.size();
	std::vector<std::vector<float>> costMatrix(trackedObjects.size(),
						   std::vector<float>(numDetections, 0));

	for (size_t i = 0; i < trackedObjects.size(); ++i) {
		for (size_t j = 0; j < numDetections; ++j) {
			const float iou = computeIoU(trackedObjects[i].rect, detections[j].rect);
			costMatrix[i][j] = iou > 0.0f ? 1.0f - iou : 10.0f;
		}
	}

	// Solve the assignment problem using the Hungarian algorithm
	munkres_cpp::matrix_std_2d_vector costMatrixAdapter(costMatrix);
	munkres_cpp::Munkres<float, munkres_cpp::matrix_std_2d_vector> solver(costMatrixAdapter);

	// Update Kalman filters with associated detections
	std::vector<bool> detectionUsed(numDetections, false);
	std::vector<bool> trackedObjectUsed(trackedObjects.size(), false);
	for (size_t i = 0; i < trackedObjects.size(); ++i) {
		for (size_t j = 0; j < numDetections; ++j) {
			if (costMatrix[i][j] == 0) {
				const float iou =
					computeIoU(trackedObjects[i].rect, detections[j].rect);
				if (iou == 0) {
					// prevent matching detections without any overlap
					continue;
				}
				// update the tracked object with the new detection
				trackedObjects[i].rect = updateKalmanFilter(trackedObjects[i].kf,
									    detections[j].rect);
				trackedObjects[i].unseenFrames = 0;
				trackedObjects[i].label = detections[j].label;
				trackedObjects[i].prob = detections[j].prob;
				// mark the detection and the tracked object as used
				detectionUsed[j] = true;
				trackedObjectUsed[i] = true;
				break;
			}
		}
	}

	// Create new tracks for unmatched detections
	for (size_t j = 0; j < numDetections; ++j) {
		if (!detectionUsed[j]) {
			cv::KalmanFilter kf;
			initializeKalmanFilter(kf, detections[j].rect);
			trackedObjects.push_back(detections[j]);
			trackedObjects.back().kf = kf; // store the Kalman filter in the object
			trackedObjects.back().id = nextTrackID++;
			trackedObjects.back().unseenFrames = 0;
			// resize trackedObjectUsed to match the new size of trackedObjects
			trackedObjectUsed.resize(trackedObjects.size(), true);
		}
	}

	// Remove lost tracks
	std::vector<Object> newTrackedObjects;
	std::vector<int> newTrackIDs;
	for (size_t i = 0; i < trackedObjects.size(); ++i) {
		if (trackedObjectUsed[i] ||
		    trackedObjects[i].unseenFrames < this->maxUnseenFrames) {
			newTrackedObjects.push_back(trackedObjects[i]);
			if (!trackedObjectUsed[i]) {
				newTrackedObjects.back().unseenFrames++;
			}
		}
	}
	trackedObjects = newTrackedObjects;

	return trackedObjects;
}

// Get the current tracked objects and their tracking id
std::vector<Object> Sort::getTrackedObjects() const
{
	return trackedObjects;
}
