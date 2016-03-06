import numpy as np
import utils
import pdb

class LARS(object):
    def __init__(self):
        self.weights_histories = []
    def fit(self, X, y):
        num, dim = X.shape
        weights = np.zeros(dim)
        # initialization
        cors = np.dot(X.T, y)
        signs = np.ones(dim)
        signs[cors < 0] = -1
        cors[cors<0] *= -1
        I = np.argmax(cors)
        actives = []
        # step-wise select new direction
        iCXX = np.zeros(())
        for i in range(dim):
            if i == 0:
                iCXX = 1.0 / np.dot(X[:, I].reshape(1, num), X[:, I].reshape(num, 1))
                actives.append(I)
            else:
                z = X[:, I] * signs[I]
                czz = np.dot(z, z)
                czX = np.dot(z.reshape(1, num), X[:, actives]*signs[actives])
                iCXX = utils.update_inv(iCXX, czX.T, czX, czz)
                actives.append(I)
            C = cors[I]
            a = 1/np.sqrt(np.sum(iCXX))
            omega = a*np.sum(iCXX, 1)
            direction = np.dot(X[:, actives]*signs[actives], omega)
            # tooptimize: cor_desc[actives) == a, no need to compute
            cor_descs = np.dot((X*signs).T, direction)
            # select the next dim to active set
            selection = (I, cors[I]/cor_descs[I])
            for n in range(dim):
                # tooptimize: inefficient search operation
                if n in actives:
                    continue
                gamma1 = (C-cors[n]) / (a-cor_descs[n])
                gamma2 = (C+cors[n]) / (a+cor_descs[n])
                if gamma1 >= 0 and gamma1 < selection[1]:
                    selection = (n, gamma1)
                if gamma2 >= 0 and gamma2 < selection[1]:
                    selection = (n, gamma2)
            # update x'(y-mu)
            I = selection[0]
            cors -= cor_descs*selection[1]
            # tooptimize: only update the inactive
            if i < dim - 1:
                # in the last iteration, cors ~== 0, avoid numeric problem
                signs[cors < 0] *= -1
                cors[cors < 0] *= -1
            weights = weights.copy()
            weights[actives] += selection[1] * omega * signs[actives]
            self.weights_histories.append(weights)
        assert np.allclose(cors, 0)

def get_error(X, y, weights):
    e = np.dot(X, weights) - y
    return np.dot(e, e)

def get_testcase1():
    X = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, -1, 0]
    ], dtype=np.float64).T
    X[:, 2] *= 1.0 / np.sqrt(3)
    y = np.array([1, 0.6, 0.2, 0.1])
    return X, y

def get_testcase2(N, D):
    X = np.random.randn(N, D)
    y = np.random.randn(N)
    return X, y
if __name__ == "__main__":
    pdb.set_trace()
    X, y = get_testcase2(10, 5)
    lars = LARS()
    lars.fit(X, y)
    last_weights = lars.weights_histories[-1]
    ols_weights = np.dot(
        np.linalg.pinv(np.dot(X.T, X)),
        np.dot(X.T, y)
    )
    print("LAR", get_error(X, y, last_weights))
    print(np.vstack(lars.weights_histories))
    print("OLS", get_error(X, y, ols_weights))
    print(ols_weights)
