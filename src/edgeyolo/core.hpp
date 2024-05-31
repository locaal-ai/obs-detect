#ifndef _EdgeYOLO_CPP_CORE_HPP
#define _EdgeYOLO_CPP_CORE_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "ort-model/ONNXRuntimeModel.h"

namespace edgeyolo_cpp {
/**
 * @brief Define names based depends on Unicode path support
 */
#define tcout std::cout

#define imread_t cv::imread

class AbcEdgeYOLO : public ONNXRuntimeModel {
public:
	AbcEdgeYOLO(file_name_t path_to_model, int intra_op_num_threads, int inter_op_num_threads,
		    const std::string &use_gpu_, int device_id, bool use_parallel, float nms_th,
		    float conf_th, int num_classes = 80)
		: ONNXRuntimeModel(path_to_model, intra_op_num_threads, num_classes,
				   inter_op_num_threads, use_gpu_, device_id, use_parallel, nms_th,
				   conf_th)
	{
		this->num_array_ = 1;
		for (size_t i = 0; i < this->output_shapes_[0].size(); i++) {
			this->num_array_ *= (int)(this->output_shapes_[0][i]);
		}
		this->num_array_ /= (5 + this->num_classes_);
	}

protected:
	int num_array_;

	void generate_edgeyolo_proposals(const int num_array, const float *feat_ptr,
					 const float prob_threshold, std::vector<Object> &objects)
	{

		for (int idx = 0; idx < num_array; ++idx) {
			const int basic_pos = idx * (num_classes_ + 5);

			float box_objectness = feat_ptr[basic_pos + 4];
			int class_id = 0;
			float max_class_score = 0.0;
			for (int class_idx = 0; class_idx < num_classes_; ++class_idx) {
				float box_cls_score = feat_ptr[basic_pos + 5 + class_idx];
				float box_prob = box_objectness * box_cls_score;
				if (box_prob > max_class_score) {
					class_id = class_idx;
					max_class_score = box_prob;
				}
			}
			if (max_class_score > prob_threshold) {
				float x_center = feat_ptr[basic_pos + 0];
				float y_center = feat_ptr[basic_pos + 1];
				float w = feat_ptr[basic_pos + 2];
				float h = feat_ptr[basic_pos + 3];
				float x0 = x_center - w * 0.5f;
				float y0 = y_center - h * 0.5f;

				Object obj;
				obj.rect.x = x0;
				obj.rect.y = y0;
				obj.rect.width = w;
				obj.rect.height = h;
				obj.label = class_id;
				obj.prob = max_class_score;
				objects.push_back(obj);
			}
		}
	}

	void decode_outputs(const float *prob, const int num_array, std::vector<Object> &objects,
			    const float bbox_conf_thresh, const float scale, const int img_w,
			    const int img_h)
	{

		std::vector<Object> proposals;
		generate_edgeyolo_proposals(num_array, prob, bbox_conf_thresh, proposals);

		qsort_descent_inplace(proposals);

		std::vector<int> picked;
		nms_sorted_bboxes(proposals, picked, nms_thresh_);

		int count = (int)(picked.size());
		objects.clear();

		for (int i = 0; i < count; ++i) {
			// adjust offset to original unpadded
			float x0 = (proposals[picked[i]].rect.x) / scale;
			float y0 = (proposals[picked[i]].rect.y) / scale;
			float x1 = (proposals[picked[i]].rect.x + proposals[picked[i]].rect.width) /
				   scale;
			float y1 =
				(proposals[picked[i]].rect.y + proposals[picked[i]].rect.height) /
				scale;

			// clip
			x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
			y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
			x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
			y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

			proposals[picked[i]].rect.x = x0;
			proposals[picked[i]].rect.y = y0;
			proposals[picked[i]].rect.width = x1 - x0;
			proposals[picked[i]].rect.height = y1 - y0;
			proposals[picked[i]].id = objects.size() + 1;

			objects.push_back(proposals[picked[i]]);
		}
	}
};
} // namespace edgeyolo_cpp
#endif
