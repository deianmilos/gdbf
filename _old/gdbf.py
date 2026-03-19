# gdbf.py — Python mirror of the refactored C code (no numpy required)
import sys
import math
import time
import argparse

# ---------------------------
# MSVC-compatible rand()/srand()
# ---------------------------
class MsvcRand:
    # MSVCRT rand: seed = seed*214013 + 2531011; return (seed >> 16) & 0x7fff
    def __init__(self, seed=1):
        self.seed = seed & 0xffffffff

    def srand(self, seed):
        self.seed = seed & 0xffffffff

    def rand(self):
        self.seed = (self.seed * 214013 + 2531011) & 0xffffffff
        return (self.seed >> 16) & 0x7fff

    def random(self):
        # return in [0,1)
        return self.rand() / 32767.0

# ---------------------------
# File loaders
# ---------------------------
def load_matrix_sparse(base_path):
    # <base_path>_size: M N
    # <base_path>_RowDegree: M ints
    # <base_path>: M rows of RowDegree[m] ints (column indices)
    with open(f"{base_path}_size", "r") as f:
        parts = f.read().strip().split()
        M, N = int(parts[0]), int(parts[1])

    RowDegree = []
    with open(f"{base_path}_RowDegree", "r") as f:
        RowDegree = [int(x) for x in f.read().strip().split()]
        assert len(RowDegree) == M

    Rows = []
    with open(base_path, "r") as f:
        data = [int(x) for x in f.read().strip().split()]
    idx = 0
    for m in range(M):
        row = data[idx: idx + RowDegree[m]]
        idx += RowDegree[m]
        Rows.append(row)

    ColDegree = [0]*N
    for m in range(M):
        for c in Rows[m]:
            if c < 0 or c >= N:
                raise ValueError(f"Column index {c} out of range 0..{N-1}")
            ColDegree[c] += 1

    print(f"Matrix Loaded (M={M}, N={N})")
    return M, N, Rows, RowDegree, ColDegree

def load_base_matrix(base_path):
    # <base_path>_size: R C Z
    # <base_path>_mat: R*C ints
    with open(f"{base_path}_size", "r") as f:
        parts = f.read().strip().split()
        R, C, Z = int(parts[0]), int(parts[1]), int(parts[2])

    mat = []
    with open(f"{base_path}_mat", "r") as f:
        data = [int(x) for x in f.read().strip().split()]
    idx = 0
    for r in range(R):
        mat.append(data[idx: idx + C])
        idx += C

    print(f"Base Matrix loaded: NbCol={C}, NbRow={R}, Circulant={Z}")
    return R, C, Z, mat

# ---------------------------
# Gaussian elimination over GF(2) - Python version
# ---------------------------
def gaussian_elimination_mrb(Mat):
    """
    Mat: MxN list of lists, 0/1 ints.
    Returns: (Rank, Perm, MatOut)
      Perm: pivot columns then free columns
      MatOut: Mat with columns permuted by Perm, pivot block diagonalized
    """
    M = len(Mat)
    N = len(Mat[0]) if M > 0 else 0

    # deep copy Mat for in-place ops
    A = [row[:] for row in Mat]
    Perm = list(range(N))
    MatOut = [[0]*N for _ in range(M)]
    Index = [0]*N
    nb = 0
    dep = 0

    indColumn = 0
    # We'll also track Perm pivot columns in Perm[0..Rank-1]
    for m in range(M):
        if indColumn == N:
            dep = M - m
            break
        ind = m
        while ind < M and A[ind][indColumn] == 0:
            ind += 1
        if ind < M:
            # swap rows m and ind from indColumn..N-1
            for n in range(indColumn, N):
                A[m][n], A[ind][n] = A[ind][n], A[m][n]
            # zero below
            for m1 in range(m+1, M):
                if A[m1][indColumn] == 1:
                    for n in range(indColumn, N):
                        A[m1][n] ^= A[m][n]
            Perm[m] = indColumn
        else:
            Index[nb] = indColumn
            nb += 1
            m -= 1  # not meaningful in Python's for; we simulate by using extra bookkeeping
        indColumn += 1

    Rank = M - dep
    for n in range(nb):
        Perm[Rank + n] = Index[n]

    # MatOut: columns permuted by Perm
    for m in range(M):
        for n in range(N):
            MatOut[m][n] = A[m][Perm[n]]

    # Diagonalization above diagonal
    for m in range(0, Rank-1):
        for n in range(m+1, Rank):
            if MatOut[m][n] == 1:
                for k in range(n, N):
                    MatOut[m][k] ^= MatOut[n][k]

    return Rank, Perm[:], MatOut

# ---------------------------
# Build full binary matrix from sparse rows
# ---------------------------
def build_full_matrix_binary(M, N, Rows, RowDegree):
    Full = [[0]*N for _ in range(M)]
    for m in range(M):
        for c in Rows[m]:
            Full[m][c] = 1
    return Full

# ---------------------------
# Encoding using GE result
# ---------------------------
def encode_from_GE(N, rank, MatG, PermG, rng):
    U = [0]*N
    for k in range(rank, N):
        U[k] = int(math.floor(rng.random() * 2.0))
    for k in range(rank-1, -1, -1):
        acc = 0
        row = MatG[k]
        for l in range(k+1, N):
            acc ^= (row[l] & U[l])
        U[k] = acc
    Codeword = [0]*N
    for k in range(N):
        Codeword[PermG[k]] = U[k]
    return Codeword

def add_bsc_noise(codeword, alpha, rng):
    rcv = [0]*len(codeword)
    for i, b in enumerate(codeword):
        if rng.random() < alpha:
            rcv[i] = 1 - b
        else:
            rcv[i] = b
    return rcv

# ---------------------------
# Layered GDBF/PGDBF/PGDBF1
# ---------------------------
def decode_layered_gdbf(N, NbIter, Nbrow, Nbcol, Z, Basemat, Received, mode="GDBF", PoPGDBF=0.7, rng=None):
    Decide = Received[:]  # initial
    energy = [0]*N
    CNvalue = [0]*Z
    VNvalue1 = [0]*Z
    VNvalue2 = [0]*Z
    oldBatch = -1

    for it in range(NbIter):
        # energy from channel agreement
        for n in range(N):
            energy[n] = 1 if (Decide[n] ^ Received[n]) else -1

        Syndrome = 0
        # layered updates
        for lay in range(Nbrow):
            for i in range(Z):
                CNvalue[i] = 0

            for col in range(Nbcol):
                if Basemat[lay][col] != -1:
                    base = col * Z
                    for i in range(Z):
                        VNvalue1[i] = Decide[base + i]
                    # shift right by Basemat[lay][col]
                    j = Basemat[lay][col]
                    for i in range(Z):
                        VNvalue2[i] = VNvalue1[j]
                        j += 1
                        if j == Z: j = 0
                    for i in range(Z):
                        CNvalue[i] ^= VNvalue2[i]

            # accumulate into energy
            for col in range(Nbcol):
                if Basemat[lay][col] != -1:
                    j = Basemat[lay][col]
                    for i in range(Z):
                        VNvalue1[j] = CNvalue[i]
                        j += 1
                        if j == Z: j = 0
                    base = col * Z
                    for i in range(Z):
                        energy[base+i] += -1 if VNvalue1[i] == 0 else 1

            if Syndrome != 1:
                for i in range(Z):
                    if CNvalue[i] != 0:
                        Syndrome = 1
                        break

        if Syndrome == 0:
            return True, Decide, it

        # flip with max energy
        EnergyMax = max(energy)

        if mode == "GDBF":
            for n in range(N):
                if energy[n] == EnergyMax:
                    Decide[n] = 1 - Decide[n]
        elif mode == "PGDBF":
            if rng is None:
                raise ValueError("PGDBF requires rng")
            for n in range(N):
                if energy[n] == EnergyMax and rng.random() < PoPGDBF:
                    Decide[n] = 1 - Decide[n]
        elif mode == "PGDBF1":
            if it <= 1:
                for n in range(N):
                    if energy[n] == EnergyMax:
                        Decide[n] = 1 - Decide[n]
            else:
                batchStart = -1
                p = 1
                cnt = 0
                for batch in range(0, N, Z*p):
                    isMax = any(energy[i] == EnergyMax for i in range(batch, min(N, batch+Z*p)))
                    if isMax and batch != oldBatch:
                        if (batchStart == -1) or (rng.random() < 0.5):
                            cnt += 1
                            batchStart = batch
                            for n in range(batchStart, min(N, batchStart+Z*p)):
                                if energy[n] == EnergyMax:
                                    Decide[n] = 1 - Decide[n]
                            oldBatch = batchStart
                    if cnt == 12:
                        break
                if batchStart < 0:
                    batchStart = oldBatch
                    for n in range(batchStart, min(N, batchStart+Z*p)):
                        if energy[n] == EnergyMax:
                            Decide[n] = 1 - Decide[n]
        else:
            raise ValueError("Unknown mode")

    return False, Decide, NbIter-1

# ---------------------------
# Main simulation
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--NbMonteCarlo", type=int, required=True)
    ap.add_argument("--NbIter", type=int, required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--alpha", type=float, default=0.0)  # not used; loop uses alpha_max..min..step
    ap.add_argument("--NBframes", type=int, required=True)
    ap.add_argument("--Graine", type=int, required=True)
    ap.add_argument("--alpha-max", type=float, required=True)
    ap.add_argument("--alpha-min", type=float, required=True)
    ap.add_argument("--alpha-step", type=float, required=True)
    ap.add_argument("--mode", choices=["GDBF", "PGDBF", "PGDBF1"], default="GDBF")
    ap.add_argument("--PoPGDBF", type=float, default=0.7)
    args = ap.parse_args()

    # Seed like C: srand(time(0)+Graine*31+113)
    seed = int(time.time()) + args.Graine * 31 + 113
    rng = MsvcRand(seed)

    M, N, Rows, RowDegree, ColDegree = load_matrix_sparse(args.matrix)
    R, C, Z, Basemat = load_base_matrix(args.base)

    # Build full matrix, GE
    MatFull = build_full_matrix_binary(M, N, Rows, RowDegree)
    rank, PermG, MatG = gaussian_elimination_mrb(MatFull)

    print(f"Rank={rank}")

    # Stats headers
    print("alpha\t\tNbEr(BER)\t\tNbFer(FER)\t\tNbtested\t\tIterAver(Itermax)\t\tNbUndec(Dmin)")

    a = args.alpha_max
    while a > args.alpha_min + 1e-15:
        NiterMoy = 0
        NiterMax = 0
        Dmin = 10**9
        NbTotalErrors = 0
        NbBitError = 0
        NbUnDetectedErrors = 0
        nbtested = 0

        for nb in range(args.NbMonteCarlo):
            Codeword = encode_from_GE(N, rank, MatG, PermG, rng)
            Received = add_bsc_noise(Codeword, a, rng)

            ok, Decide, it = decode_layered_gdbf(N, args.NbIter, R, C, Z, Basemat,
                                                 Received, mode=args.mode,
                                                 PoPGDBF=args.PoPGDBF, rng=rng)
            nbtested += 1
            nbErr = sum(1 for i in range(N) if Decide[i] != Codeword[i])
            NbBitError += nbErr

            if not ok:
                NiterMoy += args.NbIter
                NbTotalErrors += 1
            elif ok and nbErr == 0:
                NiterMax = max(NiterMax, it + 1)
                NiterMoy += (it + 1)
            elif ok and nbErr != 0:
                NiterMax = max(NiterMax, it + 1)
                NiterMoy += (it + 1)
                NbTotalErrors += 1
                NbUnDetectedErrors += 1
                Dmin = min(Dmin, nbErr)

            if NbTotalErrors == args.NBframes:
                break

            if nbtested % 1000000 == 0:
                ber = NbBitError / (N * nbtested)
                fer = NbTotalErrors / nbtested
                print(f"{a:1.5f}\t\t{NbBitError:10d} ({ber:1.15f})\t\t{NbTotalErrors:4d} ({fer:1.15f})\t\t"
                      f"{nbtested:10d}\t\t{(NiterMoy/nbtested):1.2f}({NiterMax})\t\t{NbUnDetectedErrors}({Dmin})")
        # final line per alpha
        if nbtested > 0:
            ber = NbBitError / (N * nbtested)
            fer = NbTotalErrors / nbtested
            print(f"{a:1.5f}\t\t{NbBitError:10d} ({ber:1.15f})\t\t{NbTotalErrors:4d} ({fer:1.15f})\t\t"
                  f"{nbtested:10d}\t\t{(NiterMoy/nbtested):1.2f}({NiterMax})\t\t{NbUnDetectedErrors}({Dmin})")
        a -= args.alpha_step

if __name__ == "__main__":
    main()