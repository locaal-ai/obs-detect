#ifndef _EdgeYOLO_CPP_EdgeYOLO_ONNX_HPP
#define _EdgeYOLO_CPP_EdgeYOLO_ONNX_HPP

#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <onnxruntime_cxx_api.h>

#include "core.hpp"
#include "coco_names.hpp"

namespace edgeyolo_cpp {
class EdgeYOLOONNXRuntime : public AbcEdgeYOLO {
public:
	EdgeYOLOONNXRuntime(file_name_t path_to_model, int intra_op_num_threads,
			    int num_classes = 80, int inter_op_num_threads = 1,
			    const std::string &use_gpu_ = "", int device_id = 0,
			    bool use_parallel = false, float nms_th = 0.45f, float conf_th = 0.3f);
	std::vector<Object> inference(const cv::Mat &frame) override;
};

} // namespace edgeyolo_cpp

#endif
