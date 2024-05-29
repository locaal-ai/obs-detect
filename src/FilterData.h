#ifndef FILTERDATA_H
#define FILTERDATA_H

#include <obs-module.h>
#include "edgeyolo/edgeyolo_onnxruntime.hpp"
#include "sort/Sort.h"

/**
  * @brief The filter_data struct
  *
  * This struct is used to store the base data needed for ORT filters.
  *
*/
struct filter_data {
	std::string useGPU;
	uint32_t numThreads;
	float conf_threshold;
	std::string modelSize;

	int objectCategory;
	bool maskingEnabled;
	std::string maskingType;
	int maskingColor;
	int maskingBlurRadius;
	bool trackingEnabled;
	float zoomFactor;
	float zoomSpeedFactor;
	std::string zoomObject;
	obs_source_t *trackingFilter;
	cv::Rect2f trackingRect;
	int lastDetectedObjectId;
	bool sortTracking;
	bool showUnseenObjects;
	std::string saveDetectionsPath;

	// create SORT tracker
	Sort tracker;

	obs_source_t *source;
	gs_texrender_t *texrender;
	gs_stagesurf_t *stagesurface;
	gs_effect_t *kawaseBlurEffect;
	gs_effect_t *maskingEffect;

	cv::Mat inputBGRA;
	cv::Mat outputPreviewBGRA;
	cv::Mat outputMask;

	bool isDisabled;
	bool preview;

	std::mutex inputBGRALock;
	std::mutex outputLock;
	std::mutex modelMutex;

	std::unique_ptr<edgeyolo_cpp::EdgeYOLOONNXRuntime> edgeyolo;
	std::vector<std::string> classNames;

#if _WIN32
	std::wstring modelFilepath;
#else
	std::string modelFilepath;
#endif
};

#endif /* FILTERDATA_H */
