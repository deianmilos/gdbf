#include "common.h"
#include "matrix_io.h"
#include "as_enum.h"

#define MAX_AS_SIZE 6

/* Output file handle (global for callback simplicity) */
static FILE *gOutputFile = NULL;

static void PrintAbsorbingSet(const AbsorbingSetResult *as, void *userData)
{
    int i;
    (void)userData;

    if (gOutputFile) {
        fprintf(gOutputFile, "%d %d", as->size, as->unsatisfied);
        for (i = 0; i < as->size; i++) {
            fprintf(gOutputFile, " %d", as->nodes[i]);
        }
        fprintf(gOutputFile, "\n");
    }
}

int main(int argc, char *argv[])
{
    BaseMatrixData base;
    TannerGraph graph;
    int total;
    const char *outputPath;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <BaseMatrixPrefix> <OutputFile>\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "  BaseMatrixPrefix: path prefix for base matrix files\n");
        fprintf(stderr, "    (expects _mat and _size files)\n");
        fprintf(stderr, "  OutputFile: path to write results\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Example:\n");
        fprintf(stderr, "  %s input/wifin_r_1_2_Base results/absorbing_sets.txt\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "Output format (one AS per line):\n");
        fprintf(stderr, "  <size> <unsatisfied_checks> <v1> <v2> ... <vn>\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Enumerates all absorbing sets of size 3..%d.\n", MAX_AS_SIZE);
        return 1;
    }

    outputPath = argv[2];

    fprintf(stderr, "Loading base matrix from '%s'...\n", argv[1]);
    if (LoadBaseMatrixData(argv[1], &base) != 0) {
        fprintf(stderr, "Failed to load base matrix.\n");
        return 1;
    }
    fprintf(stderr, "  Base matrix: %d x %d, circulant size Z=%d\n",
            base.RowBlockCount, base.ColBlockCount, base.CirculantSize);
    fprintf(stderr, "  Expanded H: %d x %d (M=%d check nodes, N=%d variable nodes)\n",
            base.RowBlockCount * base.CirculantSize,
            base.ColBlockCount * base.CirculantSize,
            base.RowBlockCount * base.CirculantSize,
            base.ColBlockCount * base.CirculantSize);

    fprintf(stderr, "Building Tanner graph from base matrix...\n");
    if (BuildTannerGraphFromBase(&base, &graph) != 0) {
        fprintf(stderr, "Failed to build Tanner graph.\n");
        FreeBaseMatrixData(&base);
        return 1;
    }

    gOutputFile = fopen(outputPath, "w");
    if (!gOutputFile) {
        fprintf(stderr, "Failed to open output file '%s'\n", outputPath);
        FreeTannerGraph(&graph);
        FreeBaseMatrixData(&base);
        return 1;
    }

    /* Header */
    fprintf(gOutputFile, "# Absorbing sets for %s (Z=%d, M=%d, N=%d)\n",
            argv[1], base.CirculantSize, graph.M, graph.N);
    fprintf(gOutputFile, "# Format: size unsatisfied_checks v1 v2 ... vn\n");

    fprintf(stderr, "Enumerating absorbing sets (size 3..%d)...\n", MAX_AS_SIZE);
    total = EnumerateAbsorbingSets(&graph, MAX_AS_SIZE, PrintAbsorbingSet, NULL);
    fprintf(stderr, "\nDone. Total absorbing sets found: %d\n", total);

    fclose(gOutputFile);
    FreeTannerGraph(&graph);
    FreeBaseMatrixData(&base);
    return 0;
}
