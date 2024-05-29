#ifndef _EdgeYOLO_CPP_CORE_HPP
#define _EdgeYOLO_CPP_CORE_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "types.hpp"

namespace edgeyolo_cpp {
/**
 * @brief Define names based depends on Unicode path support
 */
#define tcout std::cout

#ifdef _WIN32
#define file_name_t std::wstring
#else
#define file_name_t std::string
#endif

#define imread_t cv::imread

class AbcEdgeYOLO {
public:
	AbcEdgeYOLO() {}
	AbcEdgeYOLO(float nms_th = 0.45f, float conf_th = 0.3f, int num_classes = 80)
		: nms_thresh_(nms_th),
		  bbox_conf_thresh_(conf_th),
		  num_classes_(num_classes)
	{
	}
	virtual std::vector<Object> inference(const cv::Mat &frame) = 0;

	void setBBoxConfThresh(float thresh) { this->bbox_conf_thresh_ = thresh; }

protected:
	int input_w_;
	int input_h_;
	float nms_thresh_;
	float bbox_conf_thresh_;
	int num_classes_;
	int num_array_;
	const std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
	const std::vector<float> std_ = {0.229f, 0.224f, 0.225f};

	cv::Mat static_resize(const cv::Mat &img)
	{
		float r = std::fminf((float)input_w_ / (float)img.cols,
				     (float)input_h_ / (float)img.rows);
		// r = std::min(r, 1.0f);
		int unpad_w = (int)(r * (float)img.cols);
		int unpad_h = (int)(r * (float)img.rows);
		cv::Mat re(unpad_h, unpad_w, CV_8UC3);
		cv::resize(img, re, re.size());
		cv::Mat out(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
		re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
		return out;
	}

	// for NCHW
	void blobFromImage(const cv::Mat &img, float *blob_data)
	{
		size_t channels = 3;
		size_t img_h = img.rows;
		size_t img_w = img.cols;
		for (size_t c = 0; c < channels; ++c) {
			for (size_t h = 0; h < img_h; ++h) {
				for (size_t w = 0; w < img_w; ++w) {
					blob_data[(int)(c * img_w * img_h + h * img_w + w)] =
						(float)img.ptr<cv::Vec3b>((int)h)[(int)w][(int)c];
				}
			}
		}
	}

	// for NHWC
	void blobFromImage_nhwc(const cv::Mat &img, float *blob_data)
	{
		size_t channels = 3;
		size_t img_h = img.rows;
		size_t img_w = img.cols;
		for (size_t i = 0; i < img_h * img_w; ++i) {
			for (size_t c = 0; c < channels; ++c) {
				blob_data[i * channels + c] = (float)img.data[i * channels + c];
			}
		}
	}

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

	float intersection_area(const Object &a, const Object &b)
	{
		cv::Rect_<float> inter = a.rect & b.rect;
		return inter.area();
	}

	void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right)
	{
		int i = left;
		int j = right;
		float p = faceobjects[(left + right) / 2].prob;

		while (i <= j) {
			while (faceobjects[i].prob > p)
				++i;

			while (faceobjects[j].prob < p)
				--j;

			if (i <= j) {
				std::swap(faceobjects[i], faceobjects[j]);

				++i;
				--j;
			}
		}
		if (left < j)
			qsort_descent_inplace(faceobjects, left, j);
		if (i < right)
			qsort_descent_inplace(faceobjects, i, right);
	}

	void qsort_descent_inplace(std::vector<Object> &objects)
	{
		if (objects.empty())
			return;

		qsort_descent_inplace(objects, 0, (int)(objects.size() - 1));
	}

	void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked,
			       const float nms_threshold)
	{
		picked.clear();

		const size_t n = faceobjects.size();

		std::vector<float> areas(n);
		for (size_t i = 0; i < n; ++i) {
			areas[i] = faceobjects[i].rect.area();
		}

		for (size_t i = 0; i < n; ++i) {
			const Object &a = faceobjects[i];
			const size_t picked_size = picked.size();

			int keep = 1;
			for (size_t j = 0; j < picked_size; ++j) {
				const Object &b = faceobjects[picked[j]];

				// intersection over union
				float inter_area = intersection_area(a, b);
				float union_area = areas[i] + areas[picked[j]] - inter_area;
				// float IoU = inter_area / union_area
				if (inter_area / union_area > nms_threshold)
					keep = 0;
			}

			if (keep)
				picked.push_back((int)i);
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
