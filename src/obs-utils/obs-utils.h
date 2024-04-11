#ifndef OBS_UTILS_H
#define OBS_UTILS_H

#include "FilterData.h"

bool getRGBAFromStageSurface(filter_data *tf, uint32_t &width, uint32_t &height);

gs_texture_t *blur_image(struct filter_data *tf, uint32_t width, uint32_t height,
			 gs_texture_t *alphaTexture = nullptr);

#endif /* OBS_UTILS_H */
