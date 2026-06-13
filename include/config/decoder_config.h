#ifndef DECODER_CONFIG_H
#define DECODER_CONFIG_H

#include "decoder_framework.h"

void DecoderConfigInitDefaults(DecoderConfig *config);
int DecoderConfigLoadFromFile(DecoderConfig *config, const char *filePath);
void DecoderConfigApplyEnv(DecoderConfig *config);
int DecoderFeatureDimension(const DecoderConfig *config);
const char *DecoderFeatureName(const DecoderConfig *config, int localFeatureIndex);

#endif
