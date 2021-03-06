class LinRegression:
    def __init__(self):
        self.about = "Linear Regression by Sergei Bernadsky"
        self.W = []  # model's weights
        self.fscaling = False  # is feature scaling used

    def cost(self, y_real, y_pred):
        # cost function for gradient descent algorithm
        return np.sum((y_pred - y_real) ** 2) / (len(y_real))

    def gradient_descent_step(self, learning_rate, dy, m, n, X_tr):
        # one gradient descent step
        s = (np.dot(dy.T, X_tr)).reshape(n, 1)
        dW = 2 * (learning_rate * s / m).reshape(n, 1)
        return self.W - dW

    def normalize(self, X):
        # normilize X table
        for j in range(X.shape[1]):
            X[:, j] = X[:, j] / np.max(X[:, j])
        return X

    def fit(self, X, y, learning_rate=0.99, nsteps=3000, e=0.000000001,
            weight_low=0, weight_high=1,
            fscaling=False, kweigths=1, random_state=0):
        # train our Linear Regression model

        np.random.seed(random_state)
        X = X.astype(float)

        # Normilize process
        if fscaling == True:
            X = self.normalize(X)
            self.fscaling = True
        m = X.shape[0]
        # add one's column to X
        X = np.hstack((np.ones(m).reshape(m, 1), X))
        n = X.shape[1]

        # Weights: random initialization
        self.W = np.random.randint(low=weight_low, high=weight_high, size=(n, 1))

        y_pred = np.dot(X, self.W)
        cost0 = self.cost(y, y_pred)
        y = y.reshape(m, 1)
        k = 0

        ########## Gradient descent's steps #########
        while True:
            dy = y_pred - y
            W_tmp = self.W
            self.W = self.gradient_descent_step(learning_rate, dy, m, n, X)
            y_pred = np.dot(X, self.W)
            cost1 = self.cost(y, y_pred)
            k += 1
            if (cost1 > cost0):
                self.W = W_tmp
                break

            if ((cost0 - cost1) < e) or (k == nsteps):
                break

            cost0 = cost1
        #############################################
        return self.W  # return model's weights

    def predict(self, X):
        m = X.shape[0]
        if self.fscaling == False:
            return np.dot(np.hstack((np.ones(m).reshape(m, 1),
                                     X.astype(float))),
                          self.W)
        else:
            return np.dot(np.hstack((np.ones(m).reshape(m, 1),
                                     self.normalize(X.astype(float)))),
                          self.W);

