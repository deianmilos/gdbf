import os
import sys

def read_base_matrix(path):
    matrix = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                row = list(map(int, line.strip().split()))
                matrix.append(row)
    return matrix

def read_size(path):
    with open(path, 'r') as f:
        parts = list(map(int, f.read().split()))
        return parts[0], parts[1], parts[2]  # m, n, Z

def generate_dform(base, m, n, Z):
    M = m * Z
    N = n * Z

    dform = []

    for i in range(m):
        for z in range(Z):
            row_connections = []

            for j in range(n):
                shift = base[i][j]

                if shift >= 0:
                    col = j * Z + ((z + shift) % Z)
                    row_connections.append(col)

            dform.append(sorted(row_connections))

    return dform, M, N

def write_outputs(base_path, dform, M, N, Z):
    folder = os.path.dirname(base_path)
    base_name = os.path.basename(base_path).replace("_Base_mat", "")

    dform_path = os.path.join(folder, base_name + "_Dform")
    degree_path = os.path.join(folder, base_name + "_Dform_RowDegree")
    size_path = os.path.join(folder, base_name + "_Dform_size")

    # Write Dform
    with open(dform_path, 'w') as f:
        for row in dform:
            f.write(" ".join(map(str, row)) + "\n")

    # Write Row Degree
    with open(degree_path, 'w') as f:
        for row in dform:
            f.write(str(len(row)) + "\n")

    # Write size
    with open(size_path, 'w') as f:
        f.write(f"{M}\t{N}\t{Z}\n")

    print("Generated:")
    print(dform_path)
    print(degree_path)
    print(size_path)


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <base_matrix> <base_size>")
        sys.exit(1)

    base_matrix_path = sys.argv[1]
    base_size_path = sys.argv[2]

    base = read_base_matrix(base_matrix_path)
    m, n, Z = read_size(base_size_path)

    dform, M, N = generate_dform(base, m, n, Z)
    write_outputs(base_matrix_path, dform, M, N, Z)


if __name__ == "__main__":
    main()