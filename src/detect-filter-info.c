#include "detect-filter.h"

struct obs_source_info detect_filter_info = {
	.id = "detect-filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO,
	.get_name = detect_filter_getname,
	.create = detect_filter_create,
	.destroy = detect_filter_destroy,
	.get_defaults = detect_filter_defaults,
	.get_properties = detect_filter_properties,
	.update = detect_filter_update,
	.activate = detect_filter_activate,
	.deactivate = detect_filter_deactivate,
	.video_tick = detect_filter_video_tick,
	.video_render = detect_filter_video_render,
};
