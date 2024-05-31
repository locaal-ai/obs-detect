#ifndef ONNXRUNTIME_MODEL_H
#define ONNXRUNTIME_MODEL_H

#include <opencv2/core/types.hpp>
#include <onnxruntime_cxx_api.h>

#include <vector>
#include <array>
#include <string>
#include <tuple>

#include "types.hpp"

// Generic class for ONNXRuntime models
class ONNXRuntimeModel {
public:
	// default constructor
	ONNXRuntimeModel() {}
	// constructor with parameters
	ONNXRuntimeModel(file_name_t path_to_model, int intra_op_num_threads, int num_classes,
			 int inter_op_num_threads = 1, const std::string &use_gpu_ = "",
			 int device_id = 0, bool use_parallel = false, float nms_th = 0.45f,
			 float conf_th = 0.3f);
	// virtual destructor
	virtual ~ONNXRuntimeModel() {}

	void setBBoxConfThresh(float thresh) { this->bbox_conf_thresh_ = thresh; }
	void setNmsThresh(float thresh) { this->nms_thresh_ = thresh; }

	virtual std::vector<Object> inference(const cv::Mat &frame) = 0;

protected:
	cv::Mat static_resize(const cv::Mat &img, const int input_index);
	void blobFromImage(const cv::Mat &img, float *blob_data);
	void blobFromImage_nhwc(const cv::Mat &img, float *blob_data);
	float intersection_area(const Object &a, const Object &b);
	void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right);
	void qsort_descent_inplace(std::vector<Object> &objects);
	void nms_sorted_bboxes(const std::vector<Object> &objects, std::vector<int> &picked,
			       const float nms_threshold);

	// run inference on the model with the given frame that should go in the input index
	void inference(const cv::Mat &frame, const int input_index);

	std::vector<int> input_w_;
	std::vector<int> input_h_;
	float nms_thresh_;
	float bbox_conf_thresh_;
	int num_classes_;
	bool use_parallel_;
	int inter_op_num_threads_;
	int intra_op_num_threads_;
	int device_id_;
	std::string use_gpu;
	const std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
	const std::vector<float> std_ = {0.229f, 0.224f, 0.225f};

	Ort::Session session_{nullptr};
	Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "Default"};

	std::vector<Ort::Value> input_tensor_;
	std::vector<Ort::Value> output_tensor_;
	std::vector<std::string> input_name_;
	std::vector<std::string> output_name_;
	std::vector<std::unique_ptr<uint8_t[]>> input_buffer_;
	std::vector<std::unique_ptr<uint8_t[]>> output_buffer_;
	std::vector<Ort::ShapeInferContext::Ints> output_shapes_;
};

#endif // ONNXRUNTIME_MODEL_H
