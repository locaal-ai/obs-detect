#ifndef OBS_UTILS_H
#define OBS_UTILS_H

#include "FilterData.h"

bool getRGBAFromStageSurface(filter_data *tf, uint32_t &width, uint32_t &height);

gs_texture_t *blur_image(struct filter_data *tf, uint32_t width, uint32_t height,
			 gs_texture_t *alphaTexture = nullptr);

gs_texture_t *pixelate_image(struct filter_data *tf, uint32_t width, uint32_t height,
			     gs_texture_t *alphaTexture, float pixelateRadius);

#endif /* OBS_UTILS_H */
