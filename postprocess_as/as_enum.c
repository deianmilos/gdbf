#include "as_enum.h"
#include "matrix_io.h"
#include <string.h>

/* ---------- Tanner graph construction ---------- */

int BuildTannerGraph(const SparseMatrixData *H, TannerGraph *G)
{
    int c, i, v;

    G->M = H->M;
    G->N = H->N;

    /* Check-to-variable adjacency (copy from sparse matrix) */
    G->checkDegree = (int *)calloc(G->M, sizeof(int));
    G->checkToVar = (int **)calloc(G->M, sizeof(int *));
    if (!G->checkDegree || !G->checkToVar) return 1;

    for (c = 0; c < G->M; c++) {
        G->checkDegree[c] = H->RowDegree[c];
        G->checkToVar[c] = (int *)malloc(G->checkDegree[c] * sizeof(int));
        if (!G->checkToVar[c]) return 1;
        memcpy(G->checkToVar[c], H->Mat[c], G->checkDegree[c] * sizeof(int));
    }

    /* Variable-to-check adjacency (transpose) */
    G->varDegree = (int *)calloc(G->N, sizeof(int));
    if (!G->varDegree) return 1;

    for (c = 0; c < G->M; c++) {
        for (i = 0; i < G->checkDegree[c]; i++) {
            G->varDegree[H->Mat[c][i]]++;
        }
    }

    G->varToCheck = (int **)calloc(G->N, sizeof(int *));
    if (!G->varToCheck) return 1;

    for (v = 0; v < G->N; v++) {
        G->varToCheck[v] = (int *)malloc(G->varDegree[v] * sizeof(int));
        if (!G->varToCheck[v]) return 1;
    }

    /* Fill variable-to-check using temporary counters */
    {
        int *count = (int *)calloc(G->N, sizeof(int));
        if (!count) return 1;
        for (c = 0; c < G->M; c++) {
            for (i = 0; i < G->checkDegree[c]; i++) {
                v = H->Mat[c][i];
                G->varToCheck[v][count[v]++] = c;
            }
        }
        free(count);
    }

    return 0;
}

/* Build Tanner graph from QC-LDPC base matrix by expanding circulants */
int BuildTannerGraphFromBase(const BaseMatrixData *base, TannerGraph *G)
{
    int Z = base->CirculantSize;
    int mb = base->RowBlockCount;
    int nb = base->ColBlockCount;
    int r, c, i, v, chk;
    int *varCount;

    G->M = mb * Z;
    G->N = nb * Z;

    /* Count degrees: each non-(-1) entry contributes Z edges */
    G->checkDegree = (int *)calloc(G->M, sizeof(int));
    G->varDegree = (int *)calloc(G->N, sizeof(int));
    if (!G->checkDegree || !G->varDegree) return 1;

    for (r = 0; r < mb; r++) {
        for (c = 0; c < nb; c++) {
            if (base->ShiftMatrix[r][c] >= 0) {
                /* Each row in this block gets one edge */
                for (i = 0; i < Z; i++) {
                    G->checkDegree[r * Z + i]++;
                    v = c * Z + (i + base->ShiftMatrix[r][c]) % Z;
                    G->varDegree[v]++;
                }
            }
        }
    }

    /* Allocate adjacency lists */
    G->checkToVar = (int **)calloc(G->M, sizeof(int *));
    G->varToCheck = (int **)calloc(G->N, sizeof(int *));
    if (!G->checkToVar || !G->varToCheck) return 1;

    for (chk = 0; chk < G->M; chk++) {
        G->checkToVar[chk] = (int *)malloc(G->checkDegree[chk] * sizeof(int));
        if (!G->checkToVar[chk]) return 1;
        G->checkDegree[chk] = 0; /* reset for filling */
    }
    for (v = 0; v < G->N; v++) {
        G->varToCheck[v] = (int *)malloc(G->varDegree[v] * sizeof(int));
        if (!G->varToCheck[v]) return 1;
        G->varDegree[v] = 0; /* reset for filling */
    }

    /* Fill adjacency lists */
    for (r = 0; r < mb; r++) {
        for (c = 0; c < nb; c++) {
            int s = base->ShiftMatrix[r][c];
            if (s >= 0) {
                for (i = 0; i < Z; i++) {
                    chk = r * Z + i;
                    v = c * Z + (i + s) % Z;
                    G->checkToVar[chk][G->checkDegree[chk]++] = v;
                    G->varToCheck[v][G->varDegree[v]++] = chk;
                }
            }
        }
    }

    return 0;
}

void FreeTannerGraph(TannerGraph *G)
{
    int i;
    if (!G) return;

    if (G->checkToVar) {
        for (i = 0; i < G->M; i++) free(G->checkToVar[i]);
        free(G->checkToVar);
    }
    if (G->varToCheck) {
        for (i = 0; i < G->N; i++) free(G->varToCheck[i]);
        free(G->varToCheck);
    }
    free(G->checkDegree);
    free(G->varDegree);
    memset(G, 0, sizeof(*G));
}

/* ---------- BFS layering ---------- */

typedef struct {
    int *varLayer;      /* varLayer[v] = BFS layer index (or -1) */
    int *checkLayer;    /* checkLayer[c] = BFS layer index (or -1) */
    int maxLayer;       /* maximum variable layer index */
    int **layerVars;    /* layerVars[l] = array of variable nodes in V_l */
    int *layerVarCount; /* layerVarCount[l] = |V_l| */
    int **layerChecks;  /* layerChecks[l] = array of check nodes in C_l */
    int *layerCheckCount;
} BFSLayers;

static void FreeBFSLayers(BFSLayers *bfs, int N, int M)
{
    int l;
    (void)N; (void)M;
    if (bfs->varLayer) free(bfs->varLayer);
    if (bfs->checkLayer) free(bfs->checkLayer);
    if (bfs->layerVars) {
        for (l = 0; l <= bfs->maxLayer; l++) free(bfs->layerVars[l]);
        free(bfs->layerVars);
    }
    if (bfs->layerChecks) {
        for (l = 0; l <= bfs->maxLayer; l++) free(bfs->layerChecks[l]);
        free(bfs->layerChecks);
    }
    if (bfs->layerVarCount) free(bfs->layerVarCount);
    if (bfs->layerCheckCount) free(bfs->layerCheckCount);
}

static int BuildBFSLayers(const TannerGraph *G, int root, BFSLayers *bfs)
{
    int *varQueue, *checkQueue;
    int vqHead, vqTail, cqHead, cqTail;
    int l, v, c, i, j;
    int N = G->N, M = G->M;

    memset(bfs, 0, sizeof(*bfs));

    bfs->varLayer = (int *)malloc(N * sizeof(int));
    bfs->checkLayer = (int *)malloc(M * sizeof(int));
    if (!bfs->varLayer || !bfs->checkLayer) return 1;

    memset(bfs->varLayer, -1, N * sizeof(int));
    memset(bfs->checkLayer, -1, M * sizeof(int));

    varQueue = (int *)malloc(N * sizeof(int));
    checkQueue = (int *)malloc(M * sizeof(int));
    if (!varQueue || !checkQueue) { free(varQueue); free(checkQueue); return 1; }

    /* BFS starting from root */
    bfs->varLayer[root] = 0;
    varQueue[0] = root;
    vqHead = 0; vqTail = 1;
    bfs->maxLayer = 0;

    for (l = 0; ; l++) {
        /* Expand variable nodes at layer l → check nodes at layer l */
        cqHead = cqTail = 0;
        for (i = vqHead; i < vqTail; i++) {
            v = varQueue[i];
            if (bfs->varLayer[v] != l) continue;
            for (j = 0; j < G->varDegree[v]; j++) {
                c = G->varToCheck[v][j];
                if (bfs->checkLayer[c] == -1) {
                    bfs->checkLayer[c] = l;
                    checkQueue[cqTail++] = c;
                }
            }
        }

        if (cqTail == 0) break; /* no more check nodes to expand */

        /* Expand check nodes at layer l → variable nodes at layer l+1 */
        vqHead = vqTail;
        for (i = cqHead; i < cqTail; i++) {
            c = checkQueue[i];
            for (j = 0; j < G->checkDegree[c]; j++) {
                v = G->checkToVar[c][j];
                if (bfs->varLayer[v] == -1) {
                    bfs->varLayer[v] = l + 1;
                    varQueue[vqTail++] = v;
                    if (l + 1 > bfs->maxLayer) bfs->maxLayer = l + 1;
                }
            }
        }
    }

    /* Build layer arrays */
    bfs->layerVars = (int **)calloc(bfs->maxLayer + 1, sizeof(int *));
    bfs->layerVarCount = (int *)calloc(bfs->maxLayer + 1, sizeof(int));
    bfs->layerChecks = (int **)calloc(bfs->maxLayer + 1, sizeof(int *));
    bfs->layerCheckCount = (int *)calloc(bfs->maxLayer + 1, sizeof(int));
    if (!bfs->layerVars || !bfs->layerVarCount || !bfs->layerChecks || !bfs->layerCheckCount) {
        free(varQueue); free(checkQueue); return 1;
    }

    /* Count nodes per layer */
    for (v = 0; v < N; v++) {
        if (bfs->varLayer[v] >= 0)
            bfs->layerVarCount[bfs->varLayer[v]]++;
    }
    for (c = 0; c < M; c++) {
        if (bfs->checkLayer[c] >= 0)
            bfs->layerCheckCount[bfs->checkLayer[c]]++;
    }

    /* Allocate and fill */
    for (l = 0; l <= bfs->maxLayer; l++) {
        bfs->layerVars[l] = (int *)malloc(bfs->layerVarCount[l] * sizeof(int));
        bfs->layerChecks[l] = (int *)malloc(bfs->layerCheckCount[l] * sizeof(int));
        bfs->layerVarCount[l] = 0;
        bfs->layerCheckCount[l] = 0;
    }
    for (v = 0; v < N; v++) {
        if (bfs->varLayer[v] >= 0) {
            l = bfs->varLayer[v];
            bfs->layerVars[l][bfs->layerVarCount[l]++] = v;
        }
    }
    for (c = 0; c < M; c++) {
        if (bfs->checkLayer[c] >= 0) {
            l = bfs->checkLayer[c];
            bfs->layerChecks[l][bfs->layerCheckCount[l]++] = c;
        }
    }

    free(varQueue);
    free(checkQueue);
    return 0;
}

/* ---------- AS condition check ---------- */

/*
 * Check absorbing set condition for variable nodes in A at layer prevLayer.
 * checkDegreeInA[c] must be maintained externally (counts |N(c) ∩ A|).
 * Returns 1 if all variable nodes in the specified layer satisfy AS condition.
 */
static int ASCheckLayer(const TannerGraph *G, const int *layerNodes, int layerNodeCount,
                        const int *inA, const int *checkDegreeInA)
{
    int i, j, v, c, satisfied, unsatisfied;

    for (i = 0; i < layerNodeCount; i++) {
        v = layerNodes[i];
        if (!inA[v]) continue;
        satisfied = 0;
        unsatisfied = 0;
        for (j = 0; j < G->varDegree[v]; j++) {
            c = G->varToCheck[v][j];
            if (checkDegreeInA[c] == 0) continue; /* not connected to A */
            if (checkDegreeInA[c] % 2 == 0)
                satisfied++;
            else
                unsatisfied++;
        }
        if (satisfied <= unsatisfied)
            return 0;
    }
    return 1;
}

/* ---------- Subset enumeration ---------- */

/*
 * Context for the DFS enumeration.
 */
typedef struct {
    const TannerGraph *G;
    const BFSLayers *bfs;
    int root;
    int nu;             /* target absorbing set size */
    int *inA;           /* inA[v] = 1 if v is currently in A */
    int *checkDegreeInA;/* checkDegreeInA[c] = |N(c) ∩ A| */
    int *currentSet;    /* current absorbing set nodes (up to nu) */
    int currentSize;    /* current |A| */
    ASFoundCallback callback;
    void *userData;
    int foundCount;
} DFSContext;

/* Add variable node v to the current candidate set A */
static void AddVarNode(DFSContext *ctx, int v)
{
    int j, c;
    ctx->inA[v] = 1;
    ctx->currentSet[ctx->currentSize++] = v;
    for (j = 0; j < ctx->G->varDegree[v]; j++) {
        c = ctx->G->varToCheck[v][j];
        ctx->checkDegreeInA[c]++;
    }
}

/* Remove variable node v from the current candidate set A */
static void RemoveVarNode(DFSContext *ctx, int v)
{
    int j, c;
    ctx->inA[v] = 0;
    ctx->currentSize--;
    for (j = 0; j < ctx->G->varDegree[v]; j++) {
        c = ctx->G->varToCheck[v][j];
        ctx->checkDegreeInA[c]--;
    }
}

/*
 * Find descendants of current A_l nodes in layer l+1.
 * Returns nodes in V_{l+1} connected to A_l through C_l, with index >= root.
 */
static int FindDescendants(const DFSContext *ctx, int currentLayer,
                           int *descendants, int *descCount)
{
    const TannerGraph *G = ctx->G;
    const BFSLayers *bfs = ctx->bfs;
    int i, j, v, c;
    int nextLayer = currentLayer + 1;
    int *visited;

    *descCount = 0;

    if (nextLayer > bfs->maxLayer) return 0;

    visited = (int *)calloc(G->N, sizeof(int));
    if (!visited) return 1;

    /* For each check node in C_{currentLayer} that connects to A */
    for (i = 0; i < bfs->layerCheckCount[currentLayer]; i++) {
        c = bfs->layerChecks[currentLayer][i];
        /* Check if this check node connects to any node in A at currentLayer */
        {
            int connectsToA = 0;
            for (j = 0; j < G->checkDegree[c]; j++) {
                v = G->checkToVar[c][j];
                if (ctx->inA[v] && bfs->varLayer[v] == currentLayer) {
                    connectsToA = 1;
                    break;
                }
            }
            if (!connectsToA) continue;
        }
        /* Add variable neighbors in V_{l+1} with index >= root */
        for (j = 0; j < G->checkDegree[c]; j++) {
            v = G->checkToVar[c][j];
            if (bfs->varLayer[v] == nextLayer && v >= ctx->root && !visited[v]) {
                visited[v] = 1;
                descendants[(*descCount)++] = v;
            }
        }
    }

    free(visited);
    return 0;
}

/*
 * Recursive DFS: enumerate subsets at layer (currentLayer+1).
 * subsetCandidates[0..candidateCount-1] are the available nodes.
 * We enumerate all subsets of size 1..remaining using combination enumeration.
 */
static void ASDFS(DFSContext *ctx, int currentLayer);

/* Enumerate subsets of candidates[] of size exactly k, recurse for each */
static void EnumerateSubsets(DFSContext *ctx, int currentLayer,
                             int *candidates, int candidateCount,
                             int startIdx, int remaining, int chosen)
{
    int i;

    if (chosen == remaining) {
        /* We've chosen a complete subset of size 'remaining' at layer currentLayer+1 */
        ASDFS(ctx, currentLayer + 1);
        return;
    }

    for (i = startIdx; i < candidateCount - (remaining - chosen - 1); i++) {
        AddVarNode(ctx, candidates[i]);
        EnumerateSubsets(ctx, currentLayer, candidates, candidateCount,
                         i + 1, remaining, chosen + 1);
        RemoveVarNode(ctx, candidates[i]);
    }
}

static void ASDFS(DFSContext *ctx, int currentLayer)
{
    const TannerGraph *G = ctx->G;
    const BFSLayers *bfs = ctx->bfs;
    int remaining;
    int *descendants;
    int descCount = 0;
    int k, maxSubsetSize;

    /* Check AS condition for previous layer (if ℓ > 0) */
    if (currentLayer > 0) {
        int prevLayer = currentLayer - 1;
        if (!ASCheckLayer(G, bfs->layerVars[prevLayer], bfs->layerVarCount[prevLayer],
                          ctx->inA, ctx->checkDegreeInA)) {
            return; /* backtrack: AS condition violated */
        }
    }

    /* Check if target size reached */
    if (ctx->currentSize == ctx->nu) {
        /* Check AS condition for the last layer */
        if (ASCheckLayer(G, bfs->layerVars[currentLayer], bfs->layerVarCount[currentLayer],
                         ctx->inA, ctx->checkDegreeInA)) {
            /* Found a valid absorbing set! */
            AbsorbingSetResult result;
            int i, c, unsatisfied = 0;
            result.nodes = ctx->currentSet;
            result.size = ctx->nu;
            /* Count unsatisfied check nodes */
            for (c = 0; c < G->M; c++) {
                if (ctx->checkDegreeInA[c] > 0 && ctx->checkDegreeInA[c] % 2 == 1)
                    unsatisfied++;
            }
            result.unsatisfied = unsatisfied;
            ctx->foundCount++;
            if (ctx->callback) {
                ctx->callback(&result, ctx->userData);
            }
            (void)i;
        }
        return; /* backtrack */
    }

    /* Can't extend beyond max BFS layer */
    if (currentLayer >= bfs->maxLayer) return;

    /* Find descendants at next layer */
    descendants = (int *)malloc(G->N * sizeof(int));
    if (!descendants) return;

    if (FindDescendants(ctx, currentLayer, descendants, &descCount) != 0) {
        free(descendants);
        return;
    }

    if (descCount == 0) {
        free(descendants);
        return; /* no candidates available, can't complete */
    }

    /* Enumerate all non-empty subsets of descendants with size ≤ remaining */
    remaining = ctx->nu - ctx->currentSize;
    maxSubsetSize = (descCount < remaining) ? descCount : remaining;

    for (k = 1; k <= maxSubsetSize; k++) {
        EnumerateSubsets(ctx, currentLayer, descendants, descCount, 0, k, 0);
    }

    free(descendants);
}

/* ---------- Public API ---------- */

int EnumerateAbsorbingSetsOfSize(const TannerGraph *G, int nu,
                                  ASFoundCallback callback, void *userData)
{
    int root, totalFound = 0;
    DFSContext ctx;
    BFSLayers bfs;

    if (nu < 1 || nu > G->N) return 0;

    ctx.G = G;
    ctx.nu = nu;
    ctx.callback = callback;
    ctx.userData = userData;
    ctx.inA = (int *)calloc(G->N, sizeof(int));
    ctx.checkDegreeInA = (int *)calloc(G->M, sizeof(int));
    ctx.currentSet = (int *)malloc(nu * sizeof(int));
    if (!ctx.inA || !ctx.checkDegreeInA || !ctx.currentSet) {
        free(ctx.inA);
        free(ctx.checkDegreeInA);
        free(ctx.currentSet);
        return 0;
    }

    for (root = 0; root < G->N; root++) {
        /* Build BFS layers from this root */
        if (BuildBFSLayers(G, root, &bfs) != 0) continue;

        ctx.bfs = &bfs;
        ctx.root = root;
        ctx.currentSize = 0;
        ctx.foundCount = 0;

        /* A_0 = {root} */
        AddVarNode(&ctx, root);

        /* Start DFS at layer 0 */
        ASDFS(&ctx, 0);

        /* Clean up */
        RemoveVarNode(&ctx, root);
        totalFound += ctx.foundCount;

        FreeBFSLayers(&bfs, G->N, G->M);

        if (root % 100 == 0 && root > 0) {
            fprintf(stderr, "\r  Root %d/%d, found %d AS(size=%d) so far...",
                    root, G->N, totalFound, nu);
        }
    }

    fprintf(stderr, "\r  AS(size=%d): %d absorbing sets found (all roots).\n", nu, totalFound);

    free(ctx.inA);
    free(ctx.checkDegreeInA);
    free(ctx.currentSet);
    return totalFound;
}

int EnumerateAbsorbingSets(const TannerGraph *G, int maxSize,
                           ASFoundCallback callback, void *userData)
{
    int nu, total = 0;
    for (nu = 3; nu <= maxSize; nu++) {
        fprintf(stderr, "Enumerating absorbing sets of size %d...\n", nu);
        total += EnumerateAbsorbingSetsOfSize(G, nu, callback, userData);
    }
    return total;
}
