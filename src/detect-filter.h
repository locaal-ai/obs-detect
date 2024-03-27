#include <obs-module.h>

#ifdef __cplusplus
extern "C" {
#endif

const char *detect_filter_getname(void *unused);
void *detect_filter_create(obs_data_t *settings, obs_source_t *source);
void detect_filter_destroy(void *data);
void detect_filter_defaults(obs_data_t *settings);
obs_properties_t *detect_filter_properties(void *data);
void detect_filter_update(void *data, obs_data_t *settings);
void detect_filter_activate(void *data);
void detect_filter_deactivate(void *data);
void detect_filter_video_tick(void *data, float seconds);
void detect_filter_video_render(void *data, gs_effect_t *_effect);

#ifdef __cplusplus
}
#endif
