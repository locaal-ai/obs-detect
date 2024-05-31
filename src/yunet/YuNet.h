#ifndef YUNET_ONNX_H
#define YUNET_ONNX_H

#include <opencv2/core/types.hpp>
#include <onnxruntime_cxx_api.h>

#include <vector>
#include <array>
#include <string>
#include <tuple>

#include "ort-model/ONNXRuntimeModel.h"

namespace yunet {

static const std::vector<std::string> FACE_CLASSES = {"face"};

class YuNetONNX : public ONNXRuntimeModel {
public:
	YuNetONNX(file_name_t path_to_model, int intra_op_num_threads, int keep_topk = 50,
		  int inter_op_num_threads = 1, const std::string &use_gpu_ = "", int device_id = 0,
		  bool use_parallel = false, float nms_th = 0.45f, float conf_th = 0.3f);

	std::vector<Object> inference(const cv::Mat &frame) override;

private:
	std::tuple<std::vector<cv::Rect>, std::vector<std::array<cv::Point2f, 5>>,
		   std::vector<float>>
	inference_internal(const cv::Mat &image);

	cv::Mat preprocess(const cv::Mat &image);
	std::vector<Object> postProcess(const std::vector<Ort::Value> &result);

	struct Detections {
		std::vector<cv::Rect> bboxes;
		std::vector<std::array<cv::Point2f, 5>> landmarks;
		std::vector<float> scores;
	};

	int keep_topk;
	int divisor;
	int padH;
	int padW;
	std::vector<int> strides;
};

} // namespace yunet

#endif // YUNET_ONNX_H
