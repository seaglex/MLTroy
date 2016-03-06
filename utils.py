import numpy as np

def update_inv(invA, b, c, d):
    dim = invA.shape[0]
    b = b.reshape(dim, 1)
    c = c.reshape(1, dim)
    M = 1.0 / (d - np.dot(np.dot(c, invA), b))[0, 0]
    inv = np.zeros((dim+1, dim+1))
    inv[-1, -1] = M
    inv[[-1], :-1] = -M * np.dot(c, invA)
    inv[:-1, [-1]] = -M * np.dot(invA, b)
    inv[:-1, :-1] = invA + M * np.dot(np.dot(invA, b), np.dot(c, invA))
    return inv

if __name__ == "__main__":
    N = 4
    A = np.random.randn(N, N)
    inv = update_inv(np.linalg.inv(A[:N-1, :N-1]), A[:N-1, N-1], A[N-1, :N-1], A[N-1, N-1])
    print(np.dot(A, inv))
