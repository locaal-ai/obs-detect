#include <iostream>

#include "core.hpp"
#include "edgeyolo_onnxruntime.hpp"
#include "utils.hpp"
#include "coco_names.hpp"

#include <opencv2/core.hpp>

cv::Mat colorImageCallback(const cv::Mat &frame,
			   std::unique_ptr<edgeyolo_cpp::EdgeYOLOONNXRuntime> &edgeyolo_)
{
	// fps
	auto now = std::chrono::system_clock::now();

	auto objects = edgeyolo_->inference(frame);

	auto end = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - now);
	printf("Inference: %f FPS\n", 1000.0f / elapsed.count());
	printf("OBJECTS: %ld\n", objects.size());

	cv::Mat out_frame = frame.clone();
	edgeyolo_cpp::utils::draw_objects(out_frame, objects);

	return out_frame;
}

int main(int argc, char **argv)
{

	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <model_path>\n";
		return 1;
	}

	std::wstring model_path_ = std::wstring(argv[1], argv[1] + strlen(argv[1]));

	// parameters
	int onnxruntime_intra_op_num_threads_ = 4;
	int onnxruntime_inter_op_num_threads_ = 4;
	bool onnxruntime_use_cuda_ = false;
	int onnxruntime_device_id_ = 0;
	bool onnxruntime_use_parallel_ = false;
	float nms_th_ = 0.45;
	float conf_th_ = 0.3;
	int num_classes_ = edgeyolo_cpp::COCO_CLASSES.size();

	// Load model
	auto edgeyolo_ = std::make_unique<edgeyolo_cpp::EdgeYOLOONNXRuntime>(
		model_path_, onnxruntime_intra_op_num_threads_, onnxruntime_inter_op_num_threads_,
		onnxruntime_use_cuda_, onnxruntime_device_id_, onnxruntime_use_parallel_, nms_th_,
		conf_th_, num_classes_);

	cv::VideoCapture cap("wqctLW0Hb_0.mp4");
	// grab frames from camera and run inference
	cv::Mat frame;
	while (cap.read(frame)) {
		cv::Mat frame_draw = colorImageCallback(frame, edgeyolo_);
		cv::imshow("frame", frame_draw);
		if (cv::waitKey(1) == 27)
			break;
	}

	return 0;
}
