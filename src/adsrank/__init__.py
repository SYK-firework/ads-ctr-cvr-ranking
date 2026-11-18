"""CTR / CVR prediction and ad ranking on Criteo public datasets."""

NUM = [f"I{i}" for i in range(1, 14)]          # 13 integer features
CAT = [f"C{i}" for i in range(1, 27)]          # 26 hashed categorical features
