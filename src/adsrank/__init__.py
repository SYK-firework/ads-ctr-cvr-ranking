"""CTR / CVR prediction and ad ranking on Criteo public datasets."""
import os

# PyTorch CPU wheels call Intel MKL for some vector math (e.g. the sqrt inside Adam). MKL picks an
# AVX-512 code path when the CPU reports AVX-512; on some WSL / virtualized laptops that path crashes
# with "Illegal instruction". Default to AVX2. This runs when the package is imported, i.e. before
# any module (python -m adsrank.<module>) imports torch. Set the variable yourself to override.
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX2")

NUM = [f"I{i}" for i in range(1, 14)]          # 13 integer features
CAT = [f"C{i}" for i in range(1, 27)]          # 26 hashed categorical features
