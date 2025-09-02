import numpy as np
import time


def compute_bound_test(size=10_000_000, loops=100):
    """演算負荷型テスト（CPUの生計算性能を見る）"""
    arr = np.random.rand(size)
    start = time.time()
    for _ in range(loops):
        arr = np.sin(arr) + np.sqrt(arr)
    end = time.time()
    return end - start


def memory_bound_test(size=500_000_000):
    """メモリ負荷型テスト（メモリ帯域依存度を見る）"""
    arr = np.random.rand(size)  # 約4GB（float64）
    start = time.time()
    arr = arr + 1.0  # 全要素アクセス
    end = time.time()
    return end - start


if __name__ == "__main__":
    for i in range(5):
        print(str(i+1)+"回目")
        print("=== Compute-bound Test ===")
        t_compute = compute_bound_test()
        print(f"計算処理時間: {t_compute:.2f} 秒")

        print("\n=== Memory-bound Test ===")
        t_memory = memory_bound_test()
        print(f"メモリ処理時間: {t_memory:.2f} 秒")
