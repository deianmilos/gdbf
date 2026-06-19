#ifndef DATASET_WRITER_H
#define DATASET_WRITER_H

#include <stdint.h>
#include <stdio.h>

void SaveDatasetRow(FILE *datasetFile,
                    const int8_t *features,
                    int featureCount,
                    const int *labels,
                    int labelCount);

int CountPositiveLabels(const int *labels, int labelCount);

#endif
