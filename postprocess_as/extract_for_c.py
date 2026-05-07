Z = 27

unique_sets = set()

absorbing_sets = []

with open("C:\\Users\\dmilos\\git\\gdbf\\results\\absorbing_sets_6_validated.txt") as f:
    for line in f:

        if "VALID (6,2)" in line or "VALID (6,3)" in line:

            parts = line.split()

            v = list(map(int, parts[2:8]))

            # normalize by QC
            v_mod = sorted([x % Z for x in v])

            key = tuple(v_mod)

            if key not in unique_sets:
                unique_sets.add(key)
                absorbing_sets.append(v)

# Print C array
print("int absorbingSets[][6] = {")
for s in absorbing_sets:
    print("  {%d,%d,%d,%d,%d,%d}," % tuple(s))
print("};")

print("\nint nbAbsSets = %d;" % len(absorbing_sets))
