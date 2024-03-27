#include "edgeyolo_onnxruntime.hpp"

#ifdef _WIN32
#include <dml_provider_factory.h>
#endif

namespace edgeyolo_cpp {

EdgeYOLOONNXRuntime::EdgeYOLOONNXRuntime(file_name_t path_to_model,
					 int intra_op_num_threads,
					 int inter_op_num_threads,
					 const std::string& use_gpu, int device_id,
					 bool use_parallel, float nms_th,
					 float conf_th, int num_classes)
	: AbcEdgeYOLO(nms_th, conf_th, num_classes),
	  intra_op_num_threads_(intra_op_num_threads),
	  inter_op_num_threads_(inter_op_num_threads),
	  use_gpu(use_gpu),
	  device_id_(device_id),
	  use_parallel_(use_parallel)
{
	try {
		Ort::SessionOptions session_options;

		session_options.SetGraphOptimizationLevel(
			GraphOptimizationLevel::ORT_ENABLE_ALL);
		if (this->use_parallel_) {
			session_options.SetExecutionMode(
				ExecutionMode::ORT_PARALLEL);
			session_options.SetInterOpNumThreads(
				this->inter_op_num_threads_);
		} else {
			session_options.SetExecutionMode(
				ExecutionMode::ORT_SEQUENTIAL);
		}
		session_options.SetIntraOpNumThreads(
			this->intra_op_num_threads_);

		if (this->use_gpu == "cuda") {
			OrtCUDAProviderOptions cuda_option;
			cuda_option.device_id = this->device_id_;
			session_options.AppendExecutionProvider_CUDA(
				cuda_option);
		}
#ifdef _WIN32
        if (this->use_gpu == "dml") {
			auto &api = Ort::GetApi();
			OrtDmlApi *dmlApi = nullptr;
			Ort::ThrowOnError(api.GetExecutionProviderApi(
				"DML", ORT_API_VERSION,
				(const void **)&dmlApi));
			Ort::ThrowOnError(
				dmlApi->SessionOptionsAppendExecutionProvider_DML(
					session_options, 0));
        }
#endif

		this->session_ = Ort::Session::Session(
			this->env_, path_to_model.c_str(), session_options);
	} catch (std::exception &e) {
		std::cerr << e.what() << std::endl;
		throw e;
	}

	Ort::AllocatorWithDefaultOptions ort_alloc;

	// Allocate input memory buffer
	std::cout << "input:" << std::endl;
	this->input_name_ = std::string(
		this->session_.GetInputNameAllocated(0, ort_alloc).get());
	// this->input_name_ = this->session_.GetInputName(0, ort_alloc);
	std::cout << " name: " << this->input_name_ << std::endl;
	auto input_info = this->session_.GetInputTypeInfo(0);
	auto input_shape_info = input_info.GetTensorTypeAndShapeInfo();
	std::vector<int64_t> input_shape = input_shape_info.GetShape();
	ONNXTensorElementDataType input_tensor_type =
		input_shape_info.GetElementType();
	this->input_h_ = (int)(input_shape[2]);
	this->input_w_ = (int)(input_shape[3]);

	std::cout << " shape:" << std::endl;
	for (size_t i = 0; i < input_shape.size(); i++) {
		std::cout << "   - " << input_shape[i] << std::endl;
	}
	std::cout << " tensor_type: " << input_tensor_type << std::endl;

	size_t input_byte_count =
		sizeof(float) * input_shape_info.GetElementCount();
	std::unique_ptr<uint8_t[]> input_buffer =
		std::make_unique<uint8_t[]>(input_byte_count);
	// auto input_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
	auto input_memory_info =
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	this->input_tensor_ = Ort::Value::CreateTensor(
		input_memory_info, input_buffer.get(), input_byte_count,
		input_shape.data(), input_shape.size(), input_tensor_type);
	this->input_buffer_.emplace_back(std::move(input_buffer));

	// Allocate output memory buffer
	std::cout << "outputs" << std::endl;
	this->output_name_ = std::string(
		this->session_.GetOutputNameAllocated(0, ort_alloc).get());
	// this->output_name_ = this->session_.GetOutputName(0, ort_alloc);
	std::cout << " name: " << this->output_name_ << std::endl;

	auto output_info = this->session_.GetOutputTypeInfo(0);
	auto output_shape_info = output_info.GetTensorTypeAndShapeInfo();
	auto output_shape = output_shape_info.GetShape();
	auto output_tensor_type = output_shape_info.GetElementType();

	this->num_array_ = 1;
	std::cout << " shape:" << std::endl;
	for (size_t i = 0; i < output_shape.size(); i++) {
		std::cout << "   - " << output_shape[i] << std::endl;
		this->num_array_ *= (int)(output_shape[i]);
	}
	std::cout << " tensor_type: " << output_tensor_type << std::endl;
	this->num_array_ /= (5 + this->num_classes_);

	size_t output_byte_count =
		sizeof(float) * output_shape_info.GetElementCount();
	std::unique_ptr<uint8_t[]> output_buffer =
		std::make_unique<uint8_t[]>(output_byte_count);
	// auto output_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
	auto output_memory_info =
		Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	this->output_tensor_ = Ort::Value::CreateTensor(
		output_memory_info, output_buffer.get(), output_byte_count,
		output_shape.data(), output_shape.size(), output_tensor_type);
	this->output_buffer_.emplace_back(std::move(output_buffer));
}

std::vector<Object> EdgeYOLOONNXRuntime::inference(const cv::Mat &frame)
{
	// preprocess
	cv::Mat pr_img = static_resize(frame);

	float *blob_data = (float *)(this->input_buffer_[0].get());
	blobFromImage(pr_img, blob_data);

	const char *input_names_[] = {this->input_name_.c_str()};
	const char *output_names_[] = {this->output_name_.c_str()};

	// Inference
	Ort::RunOptions run_options;
	this->session_.Run(run_options, input_names_, &this->input_tensor_, 1,
			   output_names_, &this->output_tensor_, 1);

	float *net_pred = (float *)this->output_buffer_[0].get();

	// post process
	float scale = std::fminf(input_w_ / (frame.cols * 1.0f),
				 input_h_ / (frame.rows * 1.0f));
	std::vector<Object> objects;
	decode_outputs(net_pred, this->num_array_, objects,
		       this->bbox_conf_thresh_, scale, frame.cols, frame.rows);
	return objects;
}

} // namespace edgeyolo_cpp
