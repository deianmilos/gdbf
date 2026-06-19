#include "dataset_writer.h"

void SaveDatasetRow(FILE *datasetFile,
                    const int8_t *features,
                    int featureCount,
                    const int *labels,
                    int labelCount)
{
  int i;
  int j;

  if (datasetFile == NULL) {
    return;
  }

  for (i = 0; i < featureCount; i++) {
    fprintf(datasetFile, "%d,", (int)features[i]);
  }

  for (j = 0; j < labelCount; j++) {
    fprintf(datasetFile, "%d%s", labels[j], (j < labelCount - 1) ? "," : "\n");
  }
}

int CountPositiveLabels(const int *labels, int labelCount)
{
  int i;
  int total = 0;

  if (labels == NULL) {
    return 0;
  }

  for (i = 0; i < labelCount; i++) {
    if (labels[i] > 0) {
      total++;
    }
  }

  return total;
}
