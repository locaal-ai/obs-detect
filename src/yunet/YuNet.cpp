#include "YuNet.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>
#include <array>
#include <string>
#include <tuple>

#include "plugin-support.h"

#include <obs.h>

namespace yunet {

YuNetONNX::YuNetONNX(file_name_t path_to_model, int intra_op_num_threads, int keep_topk,
		     int inter_op_num_threads, const std::string &use_gpu_, int device_id,
		     bool use_parallel, float nms_th, float conf_th)
	: ONNXRuntimeModel(path_to_model, intra_op_num_threads, 1, inter_op_num_threads, use_gpu_,
			   device_id, use_parallel, nms_th, conf_th),
	  keep_topk(keep_topk),
	  strides({8, 16, 32}),
	  divisor(32)
{
	padW = (int((this->input_w_[0] - 1) / divisor) + 1) * divisor;
	padH = (int((this->input_h_[0] - 1) / divisor) + 1) * divisor;
}

std::vector<Object> YuNetONNX::inference(const cv::Mat &frame)
{
	ONNXRuntimeModel::inference(frame, 0);

	// Postprocessing
	std::vector<Object> objects = postProcess(this->output_tensor_);

	const float scale = std::fminf((float)input_w_[0] / (float)frame.cols,
				       (float)input_h_[0] / (float)frame.rows);

	// adjust scale to original image
	for (auto &obj : objects) {
		obj.rect.x = obj.rect.x / scale;
		obj.rect.y = obj.rect.y / scale;
		obj.rect.width = obj.rect.width / scale;
		obj.rect.height = obj.rect.height / scale;
	}

	return objects;
}

// Adapted from https://github.com/opencv/opencv/blob/98b8825031f19f47b1e33a9b9c062208f8d4acb5/modules/objdetect/src/face_detect.cpp#L161
std::vector<Object> YuNetONNX::postProcess(const std::vector<Ort::Value> &result)
{
	std::vector<Object> faces;
	for (size_t i = 0; i < this->strides.size(); ++i) {
		const float stride = (float)strides[i];
		int cols = int(this->padW / stride);
		int rows = int(this->padH / stride);

		// Extract from output_blobs
		const Ort::Value &cls = result[i];
		const Ort::Value &obj = result[i + this->strides.size() * 1];
		const Ort::Value &bbox = result[i + this->strides.size() * 2];
		const Ort::Value &kps = result[i + this->strides.size() * 3];

		// Decode from predictions
		const float *cls_v = cls.GetTensorData<float>();
		const float *obj_v = obj.GetTensorData<float>();
		const float *bbox_v = bbox.GetTensorData<float>();
		const float *kps_v = kps.GetTensorData<float>();

		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				size_t idx = r * cols + c;

				// Get score
				const float cls_score = std::clamp(cls_v[idx], 0.f, 1.f);
				const float obj_score = std::clamp(obj_v[idx], 0.f, 1.f);

				Object face;
				face.prob = std::sqrt(cls_score * obj_score);

				if (face.prob < this->bbox_conf_thresh_) {
					continue;
				}

				// Get bounding box
				const float cx = ((float)c + bbox_v[idx * 4 + 0]) * stride;
				const float cy = ((float)r + bbox_v[idx * 4 + 1]) * stride;
				const float w = exp(bbox_v[idx * 4 + 2]) * stride;
				const float h = exp(bbox_v[idx * 4 + 3]) * stride;
				const float x1 = cx - w / 2.f;
				const float y1 = cy - h / 2.f;

				face.rect = cv::Rect2f(x1, y1, w, h);
				face.label = 0;

				// TODO Get landmarks
				// for(int n = 0; n < 5; ++n) {
				//     face.at<float>(0, 4 + 2 * n) = (kps_v[idx * 10 + 2 * n] + c) * strides[i];
				//     face.at<float>(0, 4 + 2 * n + 1) = (kps_v[idx * 10 + 2 * n + 1]+ r) * strides[i];
				// }

				faces.push_back(face);
			}
		}
	}

	// run NMS
	ONNXRuntimeModel::qsort_descent_inplace(faces);
	std::vector<int> picked;
	ONNXRuntimeModel::nms_sorted_bboxes(faces, picked, this->nms_thresh_);

	// Keep topk
	if (this->keep_topk < picked.size()) {
		picked.resize(this->keep_topk);
	}

	std::vector<Object> faces_nms;
	for (size_t i = 0; i < picked.size(); ++i) {
		faces_nms.push_back(faces[picked[i]]);
	}

	return faces_nms;
}

} // namespace yunet
