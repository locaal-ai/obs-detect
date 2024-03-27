#include "detect-filter.h"

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <wchar.h>
#include <windows.h>
#endif // _WIN32

#include <opencv2/imgproc.hpp>

#include <numeric>
#include <memory>
#include <exception>
#include <fstream>
#include <new>
#include <mutex>
#include <regex>
#include <thread>

#include <plugin-support.h>
#include "FilterData.h"
#include "consts.h"
#include "obs-utils/obs-utils.h"
#include "edgeyolo/utils.hpp"

struct detect_filter : public filter_data {};

const char *detect_filter_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("Detect");
}

/**                   PROPERTIES                     */

static bool visible_on_bool(obs_properties_t *ppts, obs_data_t *settings,
			    const char *bool_prop, const char *prop_name)
{
	const bool enabled = obs_data_get_bool(settings, bool_prop);
	obs_property_t *p = obs_properties_get(ppts, prop_name);
	obs_property_set_visible(p, enabled);
	return true;
}

static bool enable_advanced_settings(obs_properties_t *ppts, obs_property_t *p,
				     obs_data_t *settings)
{
	const bool enabled = obs_data_get_bool(settings, "advanced");

	for (const char *prop_name :
	     {"threshold", "useGPU", "preview", "numThreads"}) {
		p = obs_properties_get(ppts, prop_name);
		obs_property_set_visible(p, enabled);
	}

	return true;
}

obs_properties_t *detect_filter_properties(void *data)
{
	obs_properties_t *props = obs_properties_create();

	obs_property_t *advanced = obs_properties_add_bool(
		props, "advanced", obs_module_text("Advanced"));

	// If advanced is selected show the advanced settings, otherwise hide them
	obs_property_set_modified_callback(advanced, enable_advanced_settings);

	obs_properties_add_float_slider(props, "threshold",
					obs_module_text("Threshold"), 0.0, 1.0,
					0.025);

	/* GPU, CPU and performance Props */
	obs_property_t *p_use_gpu = obs_properties_add_list(
		props, "useGPU", obs_module_text("InferenceDevice"),
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);

	obs_property_list_add_string(p_use_gpu, obs_module_text("CPU"),
				     USEGPU_CPU);
#if defined(__linux__) && defined(__x86_64__)
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPUTensorRT"),
				     USEGPU_TENSORRT);
#endif
#if _WIN32
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPUDirectML"),
				     USEGPU_DML);
#endif
#if defined(__APPLE__)
	obs_property_list_add_string(p_use_gpu, obs_module_text("CoreML"),
				     USEGPU_COREML);
#endif

	obs_properties_add_int_slider(props, "numThreads",
				      obs_module_text("NumThreads"), 0, 8, 1);

	obs_properties_add_bool(props, "preview", obs_module_text("Preview"));

	// Add a informative text about the plugin
	// replace the placeholder with the current version
	// use std::regex_replace instead of QString::arg because the latter doesn't work on Linux
	std::string basic_info = std::regex_replace(
		PLUGIN_INFO_TEMPLATE, std::regex("%1"), PLUGIN_VERSION);
	// Check for update
	// if (get_latest_version() != nullptr) {
	// 	basic_info += std::regex_replace(
	// 		PLUGIN_INFO_TEMPLATE_UPDATE_AVAILABLE, std::regex("%1"),
	// 		get_latest_version());
	// }
	obs_properties_add_text(props, "info", basic_info.c_str(),
				OBS_TEXT_INFO);

	UNUSED_PARAMETER(data);
	return props;
}

void detect_filter_defaults(obs_data_t *settings)
{
	obs_data_set_default_bool(settings, "advanced", false);
#if _WIN32
	obs_data_set_default_string(settings, "useGPU", USEGPU_DML);
#elif defined(__APPLE__)
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#else
	// Linux
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
#endif
	obs_data_set_default_int(settings, "numThreads", 1);
	obs_data_set_default_bool(settings, "preview", false);
}

void detect_filter_update(void *data, obs_data_t *settings)
{
	obs_log(LOG_INFO, "Detect filter updated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	tf->isDisabled = true;

	tf->preview = obs_data_get_bool(settings, "preview");

	const std::string newUseGpu = obs_data_get_string(settings, "useGPU");
	const uint32_t newNumThreads =
		(uint32_t)obs_data_get_int(settings, "numThreads");

	if (tf->useGPU != newUseGpu || tf->numThreads != newNumThreads) {
		// lock modelMutex
		std::unique_lock<std::mutex> lock(tf->modelMutex);

		char *modelFilepath_rawPtr = obs_module_file(
			"models/edgeyolo_tiny_lrelu_coco_256x416.onnx");

		if (modelFilepath_rawPtr == nullptr) {
			obs_log(LOG_ERROR,
				"Unable to get model filename from plugin.");
			return;
		}

		std::string modelFilepath_s(modelFilepath_rawPtr);

#if _WIN32
		int outLength = MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED,
						    modelFilepath_rawPtr, -1,
						    nullptr, 0);
		tf->modelFilepath = std::wstring(outLength, L'\0');
		MultiByteToWideChar(CP_ACP, MB_PRECOMPOSED,
				    modelFilepath_rawPtr, -1,
				    tf->modelFilepath.data(), outLength);
#else
		tf->modelFilepath = std::string(modelFilepath_rawPtr);
#endif

		// Re-initialize model if it's not already the selected one or switching inference device
		tf->useGPU = newUseGpu;
		tf->numThreads = newNumThreads;

		// parameters
		int onnxruntime_device_id_ = 0;
		bool onnxruntime_use_parallel_ = false;
		float nms_th_ = 0.45f;
		float conf_th_ = 0.3f;
		int num_classes_ = (int)edgeyolo_cpp::COCO_CLASSES.size();

		// Load model
		try {
			tf->edgeyolo = std::make_unique<
				edgeyolo_cpp::EdgeYOLOONNXRuntime>(
				tf->modelFilepath, tf->numThreads,
				tf->numThreads, tf->useGPU, onnxruntime_device_id_,
				onnxruntime_use_parallel_, nms_th_, conf_th_,
				num_classes_);
		} catch (const std::exception &e) {
			obs_log(LOG_ERROR, "Failed to load model: %s",
				e.what());
			// disable filter
			tf->isDisabled = true;
			tf->edgeyolo.reset();
			return;
		}
	}

	// Log the currently selected options
	obs_log(LOG_INFO, "Detect Filter Options:");
	// name of the source that the filter is attached to
	obs_log(LOG_INFO, "  Source: %s", obs_source_get_name(tf->source));
	obs_log(LOG_INFO, "  Inference Device: %s", tf->useGPU.c_str());
	obs_log(LOG_INFO, "  Num Threads: %d", tf->numThreads);
	obs_log(LOG_INFO, "  Disabled: %s", tf->isDisabled ? "true" : "false");
#ifdef _WIN32
	obs_log(LOG_INFO, "  Model file path: %S", tf->modelFilepath.c_str());
#else
	obs_log(LOG_INFO, "  Model file path: %s", tf->modelFilepath.c_str());
#endif

	// enable
	tf->isDisabled = false;
}

void detect_filter_activate(void *data)
{
	obs_log(LOG_INFO, "Detect filter activated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	tf->isDisabled = false;
}

void detect_filter_deactivate(void *data)
{
	obs_log(LOG_INFO, "Detect filter deactivated");
	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);
	tf->isDisabled = true;
}

/**                   FILTER CORE                     */

void *detect_filter_create(obs_data_t *settings, obs_source_t *source)
{
	obs_log(LOG_INFO, "Detect filter created");
	void *data = bmalloc(sizeof(struct detect_filter));
	struct detect_filter *tf = new (data) detect_filter();

	tf->source = source;
	tf->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	tf->effect = obs_get_base_effect(OBS_EFFECT_OPAQUE);

	detect_filter_update(tf, settings);

	return tf;
}

void detect_filter_destroy(void *data)
{
	obs_log(LOG_INFO, "Detect filter destroyed");

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf) {
		tf->isDisabled = true;

		obs_enter_graphics();
		gs_texrender_destroy(tf->texrender);
		if (tf->stagesurface) {
			gs_stagesurface_destroy(tf->stagesurface);
		}
		obs_leave_graphics();
		tf->~detect_filter();
		bfree(tf);
	}
}

void detect_filter_video_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(seconds);

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf->isDisabled) {
		return;
	}

	if (!obs_source_enabled(tf->source)) {
		return;
	}

	if (!tf->edgeyolo) {
		obs_log(LOG_ERROR, "Model is not initialized");
		return;
	}

	cv::Mat imageBGRA;
	{
		std::unique_lock<std::mutex> lock(tf->inputBGRALock,
						  std::try_to_lock);
		if (!lock.owns_lock()) {
			// No data to process
			return;
		}
		if (tf->inputBGRA.empty()) {
			// No data to process
			return;
		}
		imageBGRA = tf->inputBGRA.clone();
	}

	cv::Mat frame;
	cv::cvtColor(imageBGRA, frame, cv::COLOR_BGRA2BGR);
	std::vector<edgeyolo_cpp::Object> objects;

	try {
		std::unique_lock<std::mutex> lock(tf->modelMutex);
		objects = tf->edgeyolo->inference(frame);
	} catch (const Ort::Exception &e) {
		obs_log(LOG_ERROR, "ONNXRuntime Exception: %s", e.what());
		// TODO: Fall back to CPU if it makes sense
	} catch (const std::exception &e) {
		obs_log(LOG_ERROR, "%s", e.what());
	}

	if (objects.size() == 0) {
		return;
	}

	cv::Mat out_frame = frame.clone();
	edgeyolo_cpp::utils::draw_objects(out_frame, objects);

	std::lock_guard<std::mutex> lock(tf->outputLock);
	cv::cvtColor(out_frame, tf->outputPreviewBGRA, cv::COLOR_BGR2BGRA);
}

void detect_filter_video_render(void *data, gs_effect_t *_effect)
{
	UNUSED_PARAMETER(_effect);

	struct detect_filter *tf = reinterpret_cast<detect_filter *>(data);

	if (tf->isDisabled) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	uint32_t width, height;
	if (!getRGBAFromStageSurface(tf, width, height)) {
		if (tf->source) {
			obs_source_skip_video_filter(tf->source);
		}
		return;
	}

	// if preview is enabled, render the image
	if (tf->preview) {
		gs_texture_t *tex = nullptr;
		{
			// lock the outputLock mutex
			std::lock_guard<std::mutex> lock(tf->outputLock);
			if (tf->outputPreviewBGRA.empty()) {
				obs_log(LOG_ERROR, "Preview image is empty");
				obs_source_skip_video_filter(tf->source);
				return;
			}
			if ((uint32_t)tf->outputPreviewBGRA.cols != width ||
			    (uint32_t)tf->outputPreviewBGRA.rows != height) {
				obs_source_skip_video_filter(tf->source);
				return;
			}

			tex = gs_texture_create(
				width, height, GS_BGRA, 1,
				(const uint8_t **)&tf->outputPreviewBGRA.data,
				0);
		}

		gs_eparam_t *imageParam =
			gs_effect_get_param_by_name(tf->effect, "image");
		gs_effect_set_texture(imageParam, tex);

		gs_blend_state_push();
		gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

		while (gs_effect_loop(tf->effect, "Draw")) {
			gs_draw_sprite(tex, 0, 0, 0);
		}

		gs_blend_state_pop();
		gs_texture_destroy(tex);
	} else {
		obs_source_skip_video_filter(tf->source);
	}
	return;
}
