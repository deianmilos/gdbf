#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

#define arrondi(x) ((ceil(x)-x)<(x-floor(x))?(int)ceil(x):(int)floor(x))
#define min(x,y) ((x)<(y)?(x):(y))
#define	max(x,y) ((x)<(y)?(y):(x))
#define SQR(A) ((A)*(A))
#define BPSK(x) (1-2*(x))
#define PI 3.1415926536

#define GDBF 1
#define PGDBF 0
#define RAND_SEED    95732   // random seed


typedef struct SparseMatrixData
{
  int M;
  int N;
  int *ColumnDegree;
  int *RowDegree;
  int **Mat;
} SparseMatrixData;

static void FreeSparseMatrixData(SparseMatrixData *data)
{
  int m;
  if (data == NULL) {
    return;
  }

  if (data->Mat != NULL) {
    for (m = 0; m < data->M; m++) {
      free(data->Mat[m]);
    }
    free(data->Mat);
  }
  free(data->RowDegree);
  free(data->ColumnDegree);

  data->Mat = NULL;
  data->RowDegree = NULL;
  data->ColumnDegree = NULL;
  data->M = 0;
  data->N = 0;
}

static int LoadSparseMatrixData(const char *matrixFile, SparseMatrixData *data)
{
  FILE *f = NULL;
  char fileName[512];
  int m, k;

  if (matrixFile == NULL || data == NULL) {
    fprintf(stderr, "Invalid input to LoadSparseMatrixData.\n");
    return 1;
  }

  memset(data, 0, sizeof(*data));

  snprintf(fileName, sizeof(fileName), "%s_size", matrixFile);
  f = fopen(fileName, "r");
  if (f == NULL) {
    fprintf(stderr, "Unable to open %s\n", fileName);
    return 1;
  }

  if (fscanf(f, "%d", &data->M) != 1 || fscanf(f, "%d", &data->N) != 1) {
    fprintf(stderr, "Invalid matrix size file format: %s\n", fileName);
    fclose(f);
    return 1;
  }
  fclose(f);
  f = NULL;

  data->ColumnDegree = (int *)calloc(data->N, sizeof(int));
  data->RowDegree = (int *)calloc(data->M, sizeof(int));
  if (data->ColumnDegree == NULL || data->RowDegree == NULL) {
    fprintf(stderr, "Memory allocation failed for degree arrays.\n");
    goto fail;
  }

  snprintf(fileName, sizeof(fileName), "%s_RowDegree", matrixFile);
  f = fopen(fileName, "r");
  if (f == NULL) {
    fprintf(stderr, "Unable to open %s\n", fileName);
    goto fail;
  }
  for (m = 0; m < data->M; m++) {
    if (fscanf(f, "%d", &data->RowDegree[m]) != 1) {
      fprintf(stderr, "Invalid row degree data in %s at row %d\n", fileName, m);
      fclose(f);
      f = NULL;
      goto fail;
    }
  }
  fclose(f);
  f = NULL;

  data->Mat = (int **)calloc(data->M, sizeof(int *));
  if (data->Mat == NULL) {
    fprintf(stderr, "Memory allocation failed for matrix pointers.\n");
    goto fail;
  }
  for (m = 0; m < data->M; m++) {
    data->Mat[m] = (int *)calloc(data->RowDegree[m], sizeof(int));
    if (data->Mat[m] == NULL) {
      fprintf(stderr, "Memory allocation failed for matrix row %d.\n", m);
      goto fail;
    }
  }

  snprintf(fileName, sizeof(fileName), "%s", matrixFile);
  f = fopen(fileName, "r");
  if (f == NULL) {
    fprintf(stderr, "Unable to open %s\n", fileName);
    goto fail;
  }
  for (m = 0; m < data->M; m++) {
    for (k = 0; k < data->RowDegree[m]; k++) {
      if (fscanf(f, "%d", &data->Mat[m][k]) != 1) {
        fprintf(stderr, "Invalid matrix entry in %s at row %d, index %d\n", fileName, m, k);
        fclose(f);
        f = NULL;
        goto fail;
      }
    }
  }
  fclose(f);
  f = NULL;

  for (m = 0; m < data->M; m++) {
    for (k = 0; k < data->RowDegree[m]; k++) {
      if (data->Mat[m][k] < 0 || data->Mat[m][k] >= data->N) {
        fprintf(stderr, "Column index out of range at row %d, index %d: %d\n", m, k, data->Mat[m][k]);
        goto fail;
      }
      data->ColumnDegree[data->Mat[m][k]]++;
    }
  }

  return 0;

fail:
  if (f != NULL) {
    fclose(f);
  }
  FreeSparseMatrixData(data);
  return 1;
}

typedef struct
{
  int RowBlockCount;
  int ColBlockCount;
  int CirculantSize;
  int **ShiftMatrix;
} BaseMatrixData;

static void FreeBaseMatrixData(BaseMatrixData *base)
{
  int row;
  if (base == NULL) {
    return;
  }

  if (base->ShiftMatrix != NULL) {
    for (row = 0; row < base->RowBlockCount; row++) {
      free(base->ShiftMatrix[row]);
    }
    free(base->ShiftMatrix);
  }

  base->ShiftMatrix = NULL;
  base->RowBlockCount = 0;
  base->ColBlockCount = 0;
  base->CirculantSize = 0;
}

static int LoadBaseMatrixData(const char *baseMatrixPrefix, BaseMatrixData *base)
{
  FILE *f = NULL;
  char fileName[512];
  int row, col;

  if (baseMatrixPrefix == NULL || base == NULL) {
    fprintf(stderr, "Invalid input to LoadBaseMatrixData.\n");
    return 1;
  }

  memset(base, 0, sizeof(*base));

  snprintf(fileName, sizeof(fileName), "%s_size", baseMatrixPrefix);
  f = fopen(fileName, "r");
  if (f == NULL) {
    fprintf(stderr, "Unable to open %s\n", fileName);
    return 1;
  }
  if (fscanf(f, "%d", &base->RowBlockCount) != 1 ||
      fscanf(f, "%d", &base->ColBlockCount) != 1 ||
      fscanf(f, "%d", &base->CirculantSize) != 1) {
    fprintf(stderr, "Invalid base matrix size file format: %s\n", fileName);
    fclose(f);
    return 1;
  }
  fclose(f);
  f = NULL;

  base->ShiftMatrix = (int **)calloc(base->RowBlockCount, sizeof(int *));
  if (base->ShiftMatrix == NULL) {
    fprintf(stderr, "Memory allocation failed for base matrix row pointers.\n");
    goto fail;
  }
  for (row = 0; row < base->RowBlockCount; row++) {
    base->ShiftMatrix[row] = (int *)calloc(base->ColBlockCount, sizeof(int));
    if (base->ShiftMatrix[row] == NULL) {
      fprintf(stderr, "Memory allocation failed for base matrix row %d.\n", row);
      goto fail;
    }
  }

  snprintf(fileName, sizeof(fileName), "%s_mat", baseMatrixPrefix);
  f = fopen(fileName, "r");
  if (f == NULL) {
    fprintf(stderr, "Unable to open %s\n", fileName);
    goto fail;
  }
  for (row = 0; row < base->RowBlockCount; row++) {
    for (col = 0; col < base->ColBlockCount; col++) {
      if (fscanf(f, "%d", &base->ShiftMatrix[row][col]) != 1) {
        fprintf(stderr, "Invalid base matrix entry in %s at row %d, col %d\n", fileName, row, col);
        fclose(f);
        f = NULL;
        goto fail;
      }
    }
  }

  fclose(f);
  return 0;

fail:
  if (f != NULL) {
    fclose(f);
  }
  FreeBaseMatrixData(base);
  return 1;
}

static int ComputeMrbSystematicForm(int *columnPermutation, int **outputMatrix, int **workingMatrix, int rowCount, int colCount)
{
  int pivotRow, pivotSearchRow, eliminationRow, columnIndex, swapBuffer;
  int currentColumn = 0;
  int deferredColumnCount = 0;
  int dependentRowCount = 0;
  int rank;
  int *deferredColumns = (int *)calloc(colCount, sizeof(int));

  if (deferredColumns == NULL) {
    return 0;
  }

  for (pivotRow = 0; pivotRow < rowCount; pivotRow++)
  {
    if (currentColumn == colCount) {
      dependentRowCount = rowCount - pivotRow;
      break;
    }

    for (pivotSearchRow = pivotRow; pivotSearchRow < rowCount; pivotSearchRow++) {
      if (workingMatrix[pivotSearchRow][currentColumn] != 0) {
        break;
      }
    }

    if (pivotSearchRow < rowCount)
    {
      for (columnIndex = currentColumn; columnIndex < colCount; columnIndex++) {
        swapBuffer = workingMatrix[pivotRow][columnIndex];
        workingMatrix[pivotRow][columnIndex] = workingMatrix[pivotSearchRow][columnIndex];
        workingMatrix[pivotSearchRow][columnIndex] = swapBuffer;
      }

      for (eliminationRow = pivotRow + 1; eliminationRow < rowCount; eliminationRow++) {
        if (workingMatrix[eliminationRow][currentColumn] == 1) {
          for (columnIndex = currentColumn; columnIndex < colCount; columnIndex++) {
            workingMatrix[eliminationRow][columnIndex] ^= workingMatrix[pivotRow][columnIndex];
          }
        }
      }

      columnPermutation[pivotRow] = currentColumn;
    }
    else
    {
      deferredColumns[deferredColumnCount++] = currentColumn;
      pivotRow--;
    }

    currentColumn++;
  }

  rank = rowCount - dependentRowCount;

  for (columnIndex = 0; columnIndex < deferredColumnCount; columnIndex++) {
    columnPermutation[rank + columnIndex] = deferredColumns[columnIndex];
  }

  for (pivotRow = 0; pivotRow < rowCount; pivotRow++) {
    for (columnIndex = 0; columnIndex < colCount; columnIndex++) {
      outputMatrix[pivotRow][columnIndex] = workingMatrix[pivotRow][columnPermutation[columnIndex]];
    }
  }

  for (pivotRow = 0; pivotRow < rank - 1; pivotRow++)
  {
    for (eliminationRow = pivotRow + 1; eliminationRow < rank; eliminationRow++)
    {
      if (outputMatrix[pivotRow][eliminationRow] == 1) {
        for (columnIndex = eliminationRow; columnIndex < colCount; columnIndex++) {
          outputMatrix[pivotRow][columnIndex] ^= outputMatrix[eliminationRow][columnIndex];
        }
      }
    }
  }

  free(deferredColumns);
  return rank;
}

int GaussianElimination_MRB(int *Perm,int **MatOut,int **Mat,int M,int N)
{
  return ComputeMrbSystematicForm(Perm, MatOut, Mat, M, N);
}

static int **AllocateZeroIntMatrix(int rowCount, int colCount)
{
  int rowIndex;
  int **matrix = (int **)calloc(rowCount, sizeof(int *));
  if (matrix == NULL) {
    return NULL;
  }

  for (rowIndex = 0; rowIndex < rowCount; rowIndex++) {
    matrix[rowIndex] = (int *)calloc(colCount, sizeof(int));
    if (matrix[rowIndex] == NULL) {
      while (rowIndex-- > 0) {
        free(matrix[rowIndex]);
      }
      free(matrix);
      return NULL;
    }
  }

  return matrix;
}

static void FreeIntMatrix(int **matrix, int rowCount)
{
  int rowIndex;
  if (matrix == NULL) {
    return;
  }

  for (rowIndex = 0; rowIndex < rowCount; rowIndex++) {
    free(matrix[rowIndex]);
  }
  free(matrix);
}

typedef struct
{
  int RowCount;
  int ColCount;
  int Rank;
  int *ColumnPermutation;
  int **SystematicMatrix;
} EncodingTransformData;

static void FreeEncodingTransformData(EncodingTransformData *encoding)
{
  if (encoding == NULL) {
    return;
  }

  FreeIntMatrix(encoding->SystematicMatrix, encoding->RowCount);
  free(encoding->ColumnPermutation);
  encoding->SystematicMatrix = NULL;
  encoding->ColumnPermutation = NULL;
  encoding->RowCount = 0;
  encoding->ColCount = 0;
  encoding->Rank = 0;
}

static int BuildEncodingTransformFromSparseMatrix(const struct SparseMatrixData *matrix, EncodingTransformData *encoding)
{
  int rowIndex, rowEntryIndex, columnIndex;
  int **denseParityCheck = NULL;

  if (matrix == NULL || encoding == NULL) {
    fprintf(stderr, "Invalid input to BuildEncodingTransformFromSparseMatrix.\n");
    return 1;
  }

  memset(encoding, 0, sizeof(*encoding));
  encoding->RowCount = matrix->M;
  encoding->ColCount = matrix->N;
  encoding->ColumnPermutation = (int *)calloc(matrix->N, sizeof(int));
  encoding->SystematicMatrix = AllocateZeroIntMatrix(matrix->M, matrix->N);
  denseParityCheck = AllocateZeroIntMatrix(matrix->M, matrix->N);
  if (encoding->ColumnPermutation == NULL || encoding->SystematicMatrix == NULL || denseParityCheck == NULL) {
    fprintf(stderr, "Memory allocation failed for encoding transform.\n");
    goto fail;
  }

  for (columnIndex = 0; columnIndex < matrix->N; columnIndex++) {
    encoding->ColumnPermutation[columnIndex] = columnIndex;
  }

  for (rowIndex = 0; rowIndex < matrix->M; rowIndex++) {
    for (rowEntryIndex = 0; rowEntryIndex < matrix->RowDegree[rowIndex]; rowEntryIndex++) {
      denseParityCheck[rowIndex][matrix->Mat[rowIndex][rowEntryIndex]] = 1;
    }
  }
 
  for (columnIndex = 0; columnIndex < matrix->N; columnIndex++) {
    printf("%d ", encoding->ColumnPermutation[columnIndex]);
  }
  printf("\n");

  encoding->Rank = GaussianElimination_MRB(
    encoding->ColumnPermutation,
    encoding->SystematicMatrix,
    denseParityCheck,
    matrix->M,
    matrix->N);
  printf("Encoding Rank: %d\n", encoding->Rank);
  FreeIntMatrix(denseParityCheck, matrix->M);
  return 0;

fail:
  FreeIntMatrix(denseParityCheck, matrix->M);
  FreeEncodingTransformData(encoding);
  return 1;
}

/*
 * Generates one random valid LDPC codeword using a systematic-form parity-check matrix.
 *
 * Parameters:
 *   rank              Number of independent parity equations (pivot columns).
 *   N                 Full codeword length (number of variables / columns).
 *   systematicMatrix  Rank x N matrix after Gaussian elimination; each row defines
 *                     one parity relation used for back-substitution.
 *   columnPermutation Maps systematic-domain indices back to original variable order.
 *   workVector        Length-N temporary vector in permuted/systematic domain;
 *                     filled in-place with parity bits (0..rank-1) and random info
 *                     bits (rank..N-1).
 *   codeword          Length-N output vector in original column order.
 */
static void EncodeRandomCodeword(int rank, int N, int **systematicMatrix, const int *columnPermutation, int *workVector, int *codeword)
{
  int k;
  int l;

  for (k = 0; k < rank; k++) {
    workVector[k] = 0;
  }

  for (k = rank; k < N; k++) {
    workVector[k] = (int)floor(((double)(rand()) / RAND_MAX) * 2);
  }

  for (k = rank - 1; k >= 0; k--) {
    for (l = k + 1; l < N; l++) {
      workVector[k] = workVector[k] ^ (systematicMatrix[k][l] * workVector[l]);
    }
  }

  for (k = 0; k < N; k++) {
    codeword[columnPermutation[k]] = workVector[k];
  }
}

static void PrintCodeword(const int *codeword, int length, int index)
{
  int i;
  printf("Codeword[%d]: ", index);
  for (i = 0; i < length; i++) {
    printf("%d", codeword[i]);
  }
  printf("\n");
}

/*
 * Applies Binary Symmetric Channel (BSC) noise with crossover probability alpha.
 * Each bit is flipped independently with probability alpha.
 */
static void AddBscNoise(const int *codeword, int *receivedword, int length, float alpha)
{
  int i;
  for (i = 0; i < length; i++) {
    if (((float)rand() / (float)RAND_MAX) < alpha) {
      receivedword[i] = 1 - codeword[i];
    } else {
      receivedword[i] = codeword[i];
    }
  }
}

/*
 * Initializes decoder hard decisions from the channel output.
 * Equivalent to old code: for (k=0; k<N; k++) Decide[k] = Receivedword[k];
 */
static void InitializeDecoderBitsFromReceived(const int *receivedword, int *decodedBits, int length)
{
  int i;
  for (i = 0; i < length; i++) {
    decodedBits[i] = receivedword[i];
  }
}

/*
 * Initializes GDBF bit energy from current hard decisions and received bits.
 * Old equivalent:
 *   if (Decide[n] ^ Receivedword[n]) energy[n] = 1; else energy[n] = -1;
 */
static void InitializeBitEnergyFromDecisions(const int *decodedBits, const int *receivedword, int *bitEnergy, int length)
{
  int n;
  for (n = 0; n < length; n++) {
    bitEnergy[n] = (decodedBits[n] ^ receivedword[n]) ? 1 : -1;
  }
}

/*
 * Resets layer-local check-node parity accumulator.
 * Old equivalent: for (i=0; i<Circulant; i++) CNvalue[i] = 0;
 */
static void ResetLayerCheckNodeSyndrome(int *checkNodeSyndrome, int circulantSize)
{
  int i;
  for (i = 0; i < circulantSize; i++) {
    checkNodeSyndrome[i] = 0;
  }
}

/*
 * Applies a right circulant shift to a VN layer buffer.
 * Old equivalent:
 *   for(i=0,j=shift;i<Circulant;i++){ VNvalue2[i]=VNvalue1[j]; j++; if(j==Circulant) j=0; }
 */
static void ShiftLayerVariableBuffer(const int *inputBuffer, int *shiftedBuffer, int circulantSize, int shiftOffset)
{
  int i;
  int sourceIndex = shiftOffset;

  for (i = 0; i < circulantSize; i++) {
    shiftedBuffer[i] = inputBuffer[sourceIndex];
    sourceIndex++;
    if (sourceIndex == circulantSize) {
      sourceIndex = 0;
    }
  }
}

/*
 * XOR-accumulates one shifted VN block into the layer check-node syndrome.
 * Old equivalent: for(i=0;i<Circulant;i++) CNvalue[i] ^= VNvalue2[i];
 */
static void AccumulateCheckNodeSyndrome(int *checkNodeSyndrome, const int *shiftedLayerVariableBuffer, int circulantSize)
{
  int i;
  for (i = 0; i < circulantSize; i++) {
    checkNodeSyndrome[i] ^= shiftedLayerVariableBuffer[i];
  }
}

/*
 * Back-projects layer syndrome values to VN ordering for one column block.
 * Old equivalent: for(i=0,j=shift;i<Circulant;i++){ VNvalue1[j]=CNvalue[i]; j++; if(j==Circulant) j=0; }
 */
static void BackProjectSyndromeToLayerBuffer(
  const int *checkNodeSyndrome,
  int *layerVariableBuffer,
  int circulantSize,
  int shiftOffset)
{
  int i;
  int targetIndex = shiftOffset;

  for (i = 0; i < circulantSize; i++) {
    layerVariableBuffer[targetIndex] = checkNodeSyndrome[i];
    targetIndex++;
    if (targetIndex == circulantSize) {
      targetIndex = 0;
    }
  }
}

/*
 * Adds one layer contribution to per-bit energy for the selected column block.
 * Old equivalent: if(VNvalue1[i]==0) energy[j+i]+=-1; else energy[j+i]+=1;
 */
static void AccumulateLayerContributionToBitEnergy(
  const int *layerVariableBuffer,
  int *bitEnergy,
  int blockStart,
  int circulantSize)
{
  int i;
  for (i = 0; i < circulantSize; i++) {
    bitEnergy[blockStart + i] += (layerVariableBuffer[i] == 0) ? -1 : 1;
  }
}

/*
 * Returns 1 if any parity check in the current layer is unsatisfied.
 * Old equivalent: scan CNvalue and set Syndrome=1 on first non-zero entry.
 */
static int LayerHasParityViolation(const int *checkNodeSyndrome, int circulantSize)
{
  int i;
  for (i = 0; i < circulantSize; i++) {
    if (checkNodeSyndrome[i] != 0) {
      return 1;
    }
  }
  return 0;
}

/*
 * Returns the maximum bit-energy value over all variable nodes.
 * Old equivalent: for(n=0,EnergyMax=-255;n<N;n++) if(energy[n]>EnergyMax) EnergyMax=energy[n];
 */
static int FindMaxBitEnergy(const int *bitEnergy, int length)
{
  int n;
  int maxEnergy = -255;
  for (n = 0; n < length; n++) {
    if (bitEnergy[n] > maxEnergy) {
      maxEnergy = bitEnergy[n];
    }
  }
  return maxEnergy;
}

/*
 * GDBF bit-flip rule: flip every bit that reaches the global max energy.
 * Old equivalent: for(n=0;n<N;n++) if(energy[n]==EnergyMax) Decide[n]=1-Decide[n];
 */
static void FlipBitsAtMaxEnergy(int *decodedBits, const int *bitEnergy, int length, int maxEnergy)
{
  int n;
  for (n = 0; n < length; n++) {
    if (bitEnergy[n] == maxEnergy) {
      decodedBits[n] = 1 - decodedBits[n];
    }
  }
}

/*
 * Counts bit mismatches between decoded bits and transmitted codeword.
 * Old equivalent: NbError=0; for(k=0;k<N;k++) if(Decide[k]!=Codeword[k]) NbError++;
 */
static int CountBitErrors(const int *decodedBits, const int *codeword, int length)
{
  int n;
  int errorCount = 0;
  for (n = 0; n < length; n++) {
    if (decodedBits[n] != codeword[n]) {
      errorCount++;
    }
  }
  return errorCount;
}

int main(int argc, char *argv[])
{
  SparseMatrixData matrix;          
  BaseMatrixData base;              
  EncodingTransformData encoding;   
  FILE *fout = NULL;                
  char resultFileName[512];         
  int NbMonteCarlo;                 
  int maxDecoderIterations;         
  int NBframes;                     
  int nb;                           
  float alpha;                      
  float alpha_max;                  
  float alpha_min;                  
  float alpha_step;                 
  int nbtestedframes = 0;           

  // statistics section 
  int NiterMoy = 0;                 
  int NiterMax = 0;                 
  int Dmin = 100000;                
  int NbTotalErrors = 0;            
  int NbBitError = 0;               
  int NbUnDetectedErrors = 0;       
  int *codeword = NULL;             
  int *receivedword = NULL;         
  int *workVector = NULL;           

  // becoder buffers 
  int *decodedBits = NULL;          // Decoder hard decisions, initialized from receivedword (length N).
  int *bitEnergy = NULL;            // GDBF energy/metric per variable node to select flips (length N).
  int *checkNodeSyndrome = NULL;    // Layer-local check-node parity values CNvalue (length Circulant).
  int *layerVariableBuffer = NULL;  // Layer-local VN scratch buffer before shift (old VNvalue1).
  int *shiftedLayerVariableBuffer = NULL; // Layer-local VN scratch after circulant shift (old VNvalue2).

  if (argc < 7) {
    fprintf(stderr, "Usage: %s <NbMonteCarlo> <NbIter> <MatrixFile> <BaseMatrixPrefix> <ResultFile> <alpha> [other args...]\n", argv[0]);
    return 1;
  }

  NbMonteCarlo = atoi(argv[1]);
  maxDecoderIterations = atoi(argv[2]);
  NBframes = (argc > 7) ? atoi(argv[7]) : 0;
  alpha = (float)atof(argv[6]);
  alpha_max = (argc > 9) ? (float)atof(argv[9]) : alpha;
  alpha_min = (argc > 10) ? (float)atof(argv[10]) : (alpha - 1.0f);
  alpha_step = (argc > 11) ? (float)atof(argv[11]) : 1.0f;
  if (NbMonteCarlo <= 0) {
    fprintf(stderr, "NbMonteCarlo must be > 0\n");
    return 1;
  }
  if (maxDecoderIterations <= 0) {
    fprintf(stderr, "NbIter must be > 0\n");
    return 1;
  }
  if (alpha < 0.0f || alpha > 1.0f) {
    fprintf(stderr, "alpha must be in [0, 1]\n");
    return 1;
  }
  if (alpha_step <= 0.0f) {
    fprintf(stderr, "alpha_step must be > 0\n");
    return 1;
  }

  if (LoadSparseMatrixData(argv[3], &matrix) != 0) {
    return 1;
  }

  if (LoadBaseMatrixData(argv[4], &base) != 0) {
    FreeSparseMatrixData(&matrix);
    return 1;
  }

  if (BuildEncodingTransformFromSparseMatrix(&matrix, &encoding) != 0) {
    FreeBaseMatrixData(&base);
    FreeSparseMatrixData(&matrix);
    return 1;
  }

  printf("Matrix Loaded\n");
  printf("Graph Built\n");
  printf("Base Matrix Loaded\n");
  printf("Encoding Transform Built\n");
  printf("BaseRows=%d, BaseCols=%d, Circulant=%d\n", base.RowBlockCount, base.ColBlockCount, base.CirculantSize);
  printf("EncodingRank=%d\n", encoding.Rank);
  printf("\n");
  printf("Base Matrix:\n");
  for (int m=0;m<base.RowBlockCount;m++) { 
    for (int k=0;k<base.ColBlockCount;k++) 
    printf("%d  ",base.ShiftMatrix[m][k]); printf("\n");
  }

  #if GDBF
  printf("-------------------------La-GDBF--------------------------------------------------\n");
  #endif

  #if PGDBF
  printf("-------------------------La-P-GDBF--------------------------------------------------\n");
  #endif

  snprintf(resultFileName, sizeof(resultFileName), "%s.res", argv[5]);
  fout = fopen(resultFileName, "a+");
  if (fout == NULL) {
    fprintf(stderr, "Output file %s error!.. Abort\n", resultFileName);
    FreeEncodingTransformData(&encoding);
    FreeBaseMatrixData(&base);
    FreeSparseMatrixData(&matrix);
    return 1;
  }

  printf("alpha\t\t\tNbEr(BER)\t\t\tNbFer(FER)\t\t\tNbtested\t\tIterAver(Itermax)\tNbUndec(Dmin)\n");
  fprintf(fout, "alpha\t\t\tNbEr(BER)\t\t\tNbFer(FER)\t\t\tNbtested\t\tIterAver(Itermax)\tNbUndec(Dmin)\n");

  workVector = (int *)calloc(matrix.N, sizeof(int));
  codeword = (int *)calloc(matrix.N, sizeof(int));
  receivedword = (int *)calloc(matrix.N, sizeof(int));
  decodedBits = (int *)calloc(matrix.N, sizeof(int));
  bitEnergy = (int *)calloc(matrix.N, sizeof(int));
  checkNodeSyndrome = (int *)calloc(base.CirculantSize, sizeof(int));
  layerVariableBuffer = (int *)calloc(base.CirculantSize, sizeof(int));
  shiftedLayerVariableBuffer = (int *)calloc(base.CirculantSize, sizeof(int));
  if (workVector == NULL ||
      codeword == NULL ||
      receivedword == NULL ||
      decodedBits == NULL ||
      bitEnergy == NULL ||
      checkNodeSyndrome == NULL ||
      layerVariableBuffer == NULL ||
      shiftedLayerVariableBuffer == NULL) {
    fprintf(stderr, "Memory allocation failed for encoding/decoder buffers.\n");
    free(workVector);
    free(codeword);
    free(receivedword);
    free(decodedBits);
    free(bitEnergy);
    free(checkNodeSyndrome);
    free(layerVariableBuffer);
    free(shiftedLayerVariableBuffer);
    FreeEncodingTransformData(&encoding);
    FreeBaseMatrixData(&base);
    FreeSparseMatrixData(&matrix);
    return 1;
  }

  srand((unsigned int)(time(NULL) + RAND_SEED));

  for (alpha = alpha_max; alpha > alpha_min; alpha -= alpha_step) {
    nbtestedframes = 0;
    NiterMoy = 0;
    NiterMax = 0;
    Dmin = 100000;
    NbTotalErrors = 0;
    NbBitError = 0;
    NbUnDetectedErrors = 0;

    for (nb = 0; nb < NbMonteCarlo; nb++) {
      EncodeRandomCodeword(
        encoding.Rank,
        matrix.N,
        encoding.SystematicMatrix,
        encoding.ColumnPermutation,
        workVector,
        codeword);

      AddBscNoise(codeword, receivedword, matrix.N, alpha);

      InitializeDecoderBitsFromReceived(receivedword, decodedBits, matrix.N);

      int isCodeword = 0;
      int usedIterations = maxDecoderIterations;

      for (int decoderIteration = 0; decoderIteration < maxDecoderIterations; decoderIteration++) {
        int syndromeFlag = 0; 

        InitializeBitEnergyFromDecisions(decodedBits, receivedword, bitEnergy, matrix.N);

        for (int layerIndex = 0; layerIndex < base.RowBlockCount; layerIndex++) {

          ResetLayerCheckNodeSyndrome(checkNodeSyndrome, base.CirculantSize);

          for (int columnBlockIndex = 0; columnBlockIndex < base.ColBlockCount; columnBlockIndex++) {
            int shiftOffset = base.ShiftMatrix[layerIndex][columnBlockIndex];
            if (shiftOffset != -1) {
              int blockStart = columnBlockIndex * base.CirculantSize;

              for (int circulantIndex = 0; circulantIndex < base.CirculantSize; circulantIndex++) {
                layerVariableBuffer[circulantIndex] = decodedBits[blockStart + circulantIndex];
              }

              ShiftLayerVariableBuffer(
                layerVariableBuffer,
                shiftedLayerVariableBuffer,
                base.CirculantSize,
                shiftOffset);

              AccumulateCheckNodeSyndrome(
                checkNodeSyndrome,
                shiftedLayerVariableBuffer,
                base.CirculantSize);
            }
          }

          for (int columnBlockIndex = 0; columnBlockIndex < base.ColBlockCount; columnBlockIndex++) {
            int shiftOffset = base.ShiftMatrix[layerIndex][columnBlockIndex];
            if (shiftOffset != -1) {
              int blockStart = columnBlockIndex * base.CirculantSize;

              BackProjectSyndromeToLayerBuffer(
                checkNodeSyndrome,
                layerVariableBuffer,
                base.CirculantSize,
                shiftOffset);

              AccumulateLayerContributionToBitEnergy(
                layerVariableBuffer,
                bitEnergy,
                blockStart,
                base.CirculantSize);
            }
          }

          if (syndromeFlag == 0 && LayerHasParityViolation(checkNodeSyndrome, base.CirculantSize)) {
            syndromeFlag = 1;
          }
        }

        if (syndromeFlag == 0) {
          isCodeword = 1;
          usedIterations = decoderIteration + 1;
          break;
        }

        // energy max search
        int maxEnergy = FindMaxBitEnergy(bitEnergy, matrix.N);

        // GDBF bit-flip update at max energy
        FlipBitsAtMaxEnergy(decodedBits, bitEnergy, matrix.N, maxEnergy);
      }

      // frame-level statistics after decoder iterations (same logic as old main)
      int frameBitErrors = CountBitErrors(decodedBits, codeword, matrix.N);
      nbtestedframes++;
      NbBitError += frameBitErrors;

      if (!isCodeword) {
        NiterMoy += maxDecoderIterations;
        NbTotalErrors++;
      } else if (frameBitErrors == 0) {
        NiterMax = max(NiterMax, usedIterations);
        NiterMoy += usedIterations;
      } else {
        NiterMax = max(NiterMax, usedIterations);
        NiterMoy += usedIterations;
        NbTotalErrors++;
        NbUnDetectedErrors++;
        Dmin = min(Dmin, frameBitErrors);
      }

      if (nbtestedframes % 1000000 == 0) {
        printf("%1.5f\t\t", alpha);
        printf("%10d (%1.15f)\t\t", NbBitError, (float)NbBitError / matrix.N / nbtestedframes);
        printf("%4d (%1.15f)\t\t", NbTotalErrors, (float)NbTotalErrors / nbtestedframes);
        printf("%10d", nbtestedframes);
        printf("%1.2f(%d)\t\t", (float)NiterMoy / nbtestedframes, NiterMax);
        printf("%d(%d)\n", NbUnDetectedErrors, Dmin);
        fflush(stdout);

        fprintf(fout, "%1.5f\t\t", alpha);
        fprintf(fout, "%10d (%1.15f)\t\t", NbBitError, (float)NbBitError / matrix.N / nbtestedframes);
        fprintf(fout, "%4d (%1.15f)\t\t", NbTotalErrors, (float)NbTotalErrors / nbtestedframes);
        fprintf(fout, "%10d", nbtestedframes);
        fprintf(fout, "%1.2f(%d)\t\t", (float)NiterMoy / nbtestedframes, NiterMax);
        fprintf(fout, "%d(%d)\n", NbUnDetectedErrors, Dmin);
        fflush(fout);
      }

      if (NBframes > 0 && NbTotalErrors == NBframes) {
        break;
      }
    }

    if (nbtestedframes > 0) {
      printf("%1.5f\t\t", alpha);
      printf("%10d (%1.15f)\t\t", NbBitError, (float)NbBitError / matrix.N / nbtestedframes);
      printf("%4d (%1.15f)\t\t", NbTotalErrors, (float)NbTotalErrors / nbtestedframes);
      printf("%10d\t", nbtestedframes);
      printf("%1.2f(%d)\t\t", (float)NiterMoy / nbtestedframes, NiterMax);
      printf("%d(%d)\n", NbUnDetectedErrors, Dmin);
      fflush(stdout);

      fprintf(fout, "%1.5f\t\t", alpha);
      fprintf(fout, "%10d (%1.15f)\t\t", NbBitError, (float)NbBitError / matrix.N / nbtestedframes);
      fprintf(fout, "%4d (%1.15f)\t\t", NbTotalErrors, (float)NbTotalErrors / nbtestedframes);
      fprintf(fout, "%10d\t", nbtestedframes);
      fprintf(fout, "%1.2f(%d)\t\t", (float)NiterMoy / nbtestedframes, NiterMax);
      fprintf(fout, "%d(%d)\n", NbUnDetectedErrors, Dmin);
      fflush(fout);
    }
  }


  free(codeword);
  free(receivedword);
  free(workVector);
  free(decodedBits);
  free(bitEnergy);
  free(checkNodeSyndrome);
  free(layerVariableBuffer);
  free(shiftedLayerVariableBuffer);

  if (fout != NULL) {
    fclose(fout);
  }

  FreeEncodingTransformData(&encoding);
  FreeBaseMatrixData(&base);
  FreeSparseMatrixData(&matrix);
  return 0;
}
