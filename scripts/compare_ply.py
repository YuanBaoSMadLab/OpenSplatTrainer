"""对比 C++ 与 Python 输出的统计特征（RNG 随机性导致逐点不同，看分布）。"""
import numpy as np


def read_ply_vertices(path):
    with open(path, "rb") as f:
        data = f.read()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii")
    n = 0
    for line in header.splitlines():
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
    raw = np.frombuffer(data[header_end:], dtype=np.float32)
    return raw.reshape(n, 17)


for name, path in [("C++", "test_data/test_out.ply"), ("Python", "test_data/ref.ply")]:
    a = read_ply_vertices(path)
    print(f"=== {name} ===")
    print("  xyz range: min", a[:, :3].min(0), "max", a[:, :3].max(0))
    print("  xyz std:", a[:, :3].std(0))
    print("  f_dc range:", a[:, 6:9].min(), a[:, 6:9].max())
    print("  opacity (logit) range:", a[:, 9].min(), a[:, 9].max())
    print("  scale (log) range:", a[:, 10:13].min(), a[:, 10:13].max())
    print("  rot range:", a[:, 13:17].min(), a[:, 13:17].max())
    norms = np.linalg.norm(a[:, 13:17], axis=1)
    print("  rot norm: mean", norms.mean(), "min", norms.min(), "max", norms.max())
