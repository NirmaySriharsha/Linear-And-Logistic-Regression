import numpy as np

# Template code for gradient descent, to make the implementation
# a bit more straightforward.
# Please fill in the missing pieces, and then feel free to
# call it as needed in the functions below.
def gradient_descent(X, y, lr, num_iters):
    losses = []
    n, d = X.shape
    w = np.zeros((d,1))
    for i in range(num_iters):
        grad = 2*(X.T@(X@w - y))
        w = w - lr * grad
        loss = ((X@w - y).T)@(X@w - y)
        #print(loss)
        losses.append(loss)
    return losses, w

# Part 3.
def part_3():
    X = np.array([[1,1],[2,3],[3,3]])
    Y = np.array([[1],[3],[3]])    
    X_transpose = X.T
    Id_1 = np.array([[1,0,0], [0,1,0], [0,0,1]])
    Id_2 = np.array([[1,0], [0,1]])
    l = 0 #l = lambda
    #closed form part (a) 
    #w_1 = (np.linalg.inv(X_transpose@X + l*Id_2))@X_transpose@Y
    #print(w_1)
    #closed form part (b)
    #w_2 = X_transpose@np.linalg.inv(X@X_transpose + l*Id_1)@Y
    #print(w_2)
    #Regularization
    #l = 1
    #l = 1e-3
    #l = 1e-5
    #w_1 = (np.linalg.inv(X_transpose@X + l*Id_2))@X_transpose@Y
    #print(w_1)
    #w_2 = X_transpose@(np.linalg.inv(X@X_transpose + l*Id_1))@Y
    #print(w_2)
    num_iter = 10000
    losses, w = gradient_descent(X, Y, 0.01, num_iter)
    print(w)


    ##### ADD YOUR CODE FOR ALL PARTS OF 3 HERE

# Part 4.
def part_4():
    X = np.array([[1,1],[2,3],[3,3]]).T
    X_1 = np.array([[1, 2, 3], [1, 3, 3]])
    Y = np.array([[1],[3]])
    X_transpose = X.T
    Id_1 = np.array([[1, 0], [0,1]])
    Id_2 = np.array([[1,0, 0], [0, 1, 0],[0,0, 1]])
    l = 0 #l = lambda
    #closed form part (a) 
    #w_1 = np.linalg.inv(X_transpose@X + l*Id_2)@X_transpose@Y
    #print(w_1)
    #closed form part (b)
    #w_2 = X_transpose@np.linalg.inv(X@X_transpose + l*Id_1)@Y
    #print(w_2)

    ##### ADD YOUR CODE FOR BOTH PARTS OF 4 HERE

# Part 5.
def part_5():
    X1 = np.array([[100,0.001],[200,0.001],[-200,0.0005]])
    X2 = np.array([[100,100],[200,100],[-200,100]])
    Y = np.array([[1],[1],[1]])
    #losses_1, w_1 = gradient_descent(X1, Y, 0.01, 50)
    #print(losses_1[-1])
    #print(w_1)
    #losses_2, w_2 = gradient_descent(X2, Y, 0.01, 50)
    #print(losses_2[-1])
    #print(w_2)
    #losses_1, w_1 = gradient_descent(X1, Y, 0.5e-5, 50)
    #print(losses_1[0])
    #print(losses_1[-1])
    #print(w_1)
    #losses_2, w_2 = gradient_descent(X2, Y, 0.5e-5, 50)
    #print(losses_2[0])
    #print(losses_2[-1])
    #print(w_2)
    #eigenvalues_1, eigenvectors_1 = np.linalg.eig(2*X1.T@X1)
    #print(eigenvalues_1)
    #eigenvalues_2, eigenvectors_2 = np.linalg.eig(2*X2.T@X2)
    #print(eigenvalues_2)

    ##### ADD YOUR CODE FOR ALL PARTS OF 5 HERE

#part_5()


