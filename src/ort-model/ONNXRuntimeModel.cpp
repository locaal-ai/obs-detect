#include "ONNXRuntimeModel.h"

#ifdef _WIN32
#include <dml_provider_factory.h>
#endif

#include "plugin-support.h"

#include <obs.h>

ONNXRuntimeModel::ONNXRuntimeModel(file_name_t path_to_model, int intra_op_num_threads,
				   int num_classes, int inter_op_num_threads,
				   const std::string &use_gpu_, int device_id, bool use_parallel,
				   float nms_th, float conf_th)
	: intra_op_num_threads_(intra_op_num_threads),
	  inter_op_num_threads_(inter_op_num_threads),
	  use_gpu(use_gpu_),
	  device_id_(device_id),
	  use_parallel_(use_parallel),
	  nms_thresh_(nms_th),
	  bbox_conf_thresh_(conf_th),
	  num_classes_(num_classes)
{
	try {
		Ort::SessionOptions session_options;

		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		if (this->use_parallel_) {
			session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
			session_options.SetInterOpNumThreads(this->inter_op_num_threads_);
		} else {
			session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
		}
		session_options.SetIntraOpNumThreads(this->intra_op_num_threads_);

#ifdef _WIN32
		if (this->use_gpu == "cuda") {
			OrtCUDAProviderOptions cuda_option;
			cuda_option.device_id = this->device_id_;
			session_options.AppendExecutionProvider_CUDA(cuda_option);
		}
		if (this->use_gpu == "dml") {
			auto &api = Ort::GetApi();
			OrtDmlApi *dmlApi = nullptr;
			Ort::ThrowOnError(api.GetExecutionProviderApi("DML", ORT_API_VERSION,
								      (const void **)&dmlApi));
			Ort::ThrowOnError(dmlApi->SessionOptionsAppendExecutionProvider_DML(
				session_options, 0));
		}
#endif

		this->session_ = Ort::Session(this->env_, path_to_model.c_str(), session_options);
	} catch (std::exception &e) {
		obs_log(LOG_ERROR, "Cannot load model: %s", e.what());
		throw e;
	}

	Ort::AllocatorWithDefaultOptions ort_alloc;

	// number of inputs
	size_t num_input = this->session_.GetInputCount();

	for (size_t i = 0; i < num_input; i++) {
		auto input_info = this->session_.GetInputTypeInfo(i);
		auto input_shape_info = input_info.GetTensorTypeAndShapeInfo();
		auto input_shape = input_shape_info.GetShape();
		auto input_tensor_type = input_shape_info.GetElementType();

		// assume input shape is NCHW
		this->input_h_.push_back((int)(input_shape[2]));
		this->input_w_.push_back((int)(input_shape[3]));

		// Allocate input memory buffer
		this->input_name_.push_back(
			std::string(this->session_.GetInputNameAllocated(i, ort_alloc).get()));
		size_t input_byte_count = sizeof(float) * input_shape_info.GetElementCount();
		std::unique_ptr<uint8_t[]> input_buffer =
			std::make_unique<uint8_t[]>(input_byte_count);
		auto input_memory_info =
			Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

		this->input_tensor_.push_back(Ort::Value::CreateTensor(
			input_memory_info, input_buffer.get(), input_byte_count, input_shape.data(),
			input_shape.size(), input_tensor_type));
		this->input_buffer_.push_back(std::move(input_buffer));

		obs_log(LOG_INFO, "Input name: %s", this->input_name_[i].c_str());
		obs_log(LOG_INFO, "Input shape: %d %d %d %d", input_shape[0],
			input_shape.size() > 1 ? input_shape[1] : 0,
			input_shape.size() > 2 ? input_shape[2] : 0,
			input_shape.size() > 3 ? input_shape[3] : 0);
	}

	// number of outputs
	size_t num_output = this->session_.GetOutputCount();

	for (size_t i = 0; i < num_output; i++) {
		auto output_info = this->session_.GetOutputTypeInfo(i);
		auto output_shape_info = output_info.GetTensorTypeAndShapeInfo();
		auto output_shape = output_shape_info.GetShape();
		auto output_tensor_type = output_shape_info.GetElementType();

		this->output_shapes_.push_back(output_shape);

		// Allocate output memory buffer
		size_t output_byte_count = sizeof(float) * output_shape_info.GetElementCount();
		std::unique_ptr<uint8_t[]> output_buffer =
			std::make_unique<uint8_t[]>(output_byte_count);
		auto output_memory_info =
			Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

		this->output_tensor_.push_back(Ort::Value::CreateTensor(
			output_memory_info, output_buffer.get(), output_byte_count,
			output_shape.data(), output_shape.size(), output_tensor_type));
		this->output_buffer_.push_back(std::move(output_buffer));

		this->output_name_.push_back(
			std::string(this->session_.GetOutputNameAllocated(i, ort_alloc).get()));

		obs_log(LOG_INFO, "Output name: %s", this->output_name_[i].c_str());
		obs_log(LOG_INFO, "Output shape: %d %d %d %d", output_shape[0],
			output_shape.size() > 1 ? output_shape[1] : 0,
			output_shape.size() > 2 ? output_shape[2] : 0,
			output_shape.size() > 3 ? output_shape[3] : 0);
	}
}

cv::Mat ONNXRuntimeModel::static_resize(const cv::Mat &img, const int input_index)
{
	float r = std::fminf((float)input_w_[input_index] / (float)img.cols,
			     (float)input_h_[input_index] / (float)img.rows);
	// r = std::min(r, 1.0f);
	int unpad_w = (int)(r * (float)img.cols);
	int unpad_h = (int)(r * (float)img.rows);
	cv::Mat re(unpad_h, unpad_w, CV_8UC3);
	cv::resize(img, re, re.size());
	cv::Mat out(input_h_[input_index], input_w_[input_index], CV_8UC3,
		    cv::Scalar(114, 114, 114));
	re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
	return out;
}

// for NCHW
void ONNXRuntimeModel::blobFromImage(const cv::Mat &img, float *blob_data)
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
void ONNXRuntimeModel::blobFromImage_nhwc(const cv::Mat &img, float *blob_data)
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

float ONNXRuntimeModel::intersection_area(const Object &a, const Object &b)
{
	cv::Rect_<float> inter = a.rect & b.rect;
	return inter.area();
}

void ONNXRuntimeModel::qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right)
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

void ONNXRuntimeModel::qsort_descent_inplace(std::vector<Object> &objects)
{
	if (objects.empty())
		return;

	qsort_descent_inplace(objects, 0, (int)(objects.size() - 1));
}

void ONNXRuntimeModel::nms_sorted_bboxes(const std::vector<Object> &objects,
					 std::vector<int> &picked, const float nms_threshold)
{
	picked.clear();

	const size_t n = objects.size();

	std::vector<float> areas(n);
	for (size_t i = 0; i < n; ++i) {
		areas[i] = objects[i].rect.area();
	}

	for (size_t i = 0; i < n; ++i) {
		const Object &a = objects[i];
		const size_t picked_size = picked.size();

		int keep = 1;
		for (size_t j = 0; j < picked_size; ++j) {
			const Object &b = objects[picked[j]];

			// intersection over union
			float inter_area = this->intersection_area(a, b);
			float union_area = areas[i] + areas[picked[j]] - inter_area;
			// float IoU = inter_area / union_area
			if (inter_area / union_area > nms_threshold)
				keep = 0;
		}

		if (keep)
			picked.push_back((int)i);
	}
}

void ONNXRuntimeModel::inference(const cv::Mat &frame, const int input_index)
{
	// preprocess
	cv::Mat pr_img = this->static_resize(frame, input_index);

	float *blob_data = (float *)(this->input_buffer_[input_index].get());
	blobFromImage(pr_img, blob_data);

	// input names
	std::vector<const char *> input_names;
	for (size_t i = 0; i < this->input_name_.size(); i++) {
		input_names.push_back(this->input_name_[i].c_str());
	}

	// output names
	std::vector<const char *> output_names;
	for (size_t i = 0; i < this->output_name_.size(); i++) {
		output_names.push_back(this->output_name_[i].c_str());
	}

	// Inference
	Ort::RunOptions run_options;
	this->session_.Run(run_options, input_names.data(), this->input_tensor_.data(),
			   this->input_tensor_.size(), output_names.data(),
			   this->output_tensor_.data(), this->output_tensor_.size());
}
