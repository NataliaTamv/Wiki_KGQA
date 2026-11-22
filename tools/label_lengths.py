
import os
import sys


def main(path, n=5000):
    size = os.path.getsize(path)
    print(f"Αρχείο: {size / 1e9:.2f} GB, δείγμα {n} γραμμών ανά σημείο")
    print(f"{'θέση':>6} {'μέσο μήκος (χαρακτήρες)':>26} {'% με >50 χαρ.':>15} {'% με >80 χαρ.':>15}")
    with open(path, "rb") as f:
        for frac in (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99):
            f.seek(int(size * frac))
            if frac > 0:
                f.readline()                                   # τελειώνει η μισή γραμμή
            lengths = []
            for _ in range(n):
                line = f.readline()
                if not line:
                    break
                _, _, label = line.decode("utf-8", "replace").rstrip("\n").partition("\t")
                lengths.append(len(label))
            if not lengths:
                continue
            avg = sum(lengths) / len(lengths)
            p50 = 100 * sum(1 for x in lengths if x > 50) / len(lengths)
            p80 = 100 * sum(1 for x in lengths if x > 80) / len(lengths)
            print(f"{frac:>6.0%} {avg:>26.1f} {p50:>14.1f}% {p80:>14.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Χρήση: python label_lengths.py labels.tsv")
    main(sys.argv[1])