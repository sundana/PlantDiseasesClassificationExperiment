import numpy as np, time
from sklearn.svm import SVC

np.random.seed(42)
n_samples = 329
X = np.random.randn(n_samples, 120)
y = np.random.randint(0, 4, n_samples)

SVC(kernel='rbf', random_state=42).fit(X[:10], y[:10])  # warm-up

times = []
for _ in range(20):
    t0 = time.perf_counter()
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X, y)
    svm.predict(X)
    elapsed = time.perf_counter() - t0
    times.append(elapsed)

mean_t = np.mean(times)
print(f"SVM time: {mean_t*1000:.2f} ms (n={n_samples}, d=120)")
print(f"In seconds at 3 decimal places: {mean_t:.3f}s")
print(f"In seconds at 2 decimal places: {mean_t:.2f}s")
