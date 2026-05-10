#ifndef AS_ENUM_H
#define AS_ENUM_H

#include "common.h"

/* Tanner graph with both check-to-var and var-to-check adjacency */
typedef struct {
    int N;              /* number of variable nodes */
    int M;              /* number of check nodes */
    int *varDegree;     /* varDegree[v] = degree of variable node v */
    int **varToCheck;   /* varToCheck[v][i] = i-th check neighbor of v */
    int *checkDegree;   /* checkDegree[c] = degree of check node c */
    int **checkToVar;   /* checkToVar[c][i] = i-th variable neighbor of c */
} TannerGraph;

/* Result: one absorbing set */
typedef struct {
    int *nodes;         /* variable node indices (sorted) */
    int size;           /* |A| */
    int unsatisfied;    /* |O(A)| = number of odd-degree check nodes */
} AbsorbingSetResult;

/* Callback invoked for each found absorbing set */
typedef void (*ASFoundCallback)(const AbsorbingSetResult *as, void *userData);

/* Build Tanner graph from sparse parity-check matrix */
int BuildTannerGraph(const SparseMatrixData *H, TannerGraph *G);

/* Build Tanner graph directly from QC-LDPC base matrix */
int BuildTannerGraphFromBase(const BaseMatrixData *base, TannerGraph *G);

void FreeTannerGraph(TannerGraph *G);

/* Enumerate all absorbing sets of exact size nu.
 * Calls callback for each found AS. Returns total count. */
int EnumerateAbsorbingSetsOfSize(const TannerGraph *G, int nu,
                                  ASFoundCallback callback, void *userData);

/* Enumerate all absorbing sets of size 3..maxSize */
int EnumerateAbsorbingSets(const TannerGraph *G, int maxSize,
                           ASFoundCallback callback, void *userData);

#endif
