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
        grad = 2*(X.T@X@w - X.T@y)
        w = w - lr * grad
        loss = ((X@w-y).T)@(X@w-y) ##### ADD YOUR CODE HERE: f(w) = ...? (with lambda = 0)
        losses.append(loss)
    return losses, w

# Part 3.
def part_3():
    X = np.array([[1,1],[2,3],[3,3]])
    Y = np.array([[1],[3],[3]])    

    ##### ADD YOUR CODE FOR ALL PARTS OF 3 HERE

# Part 4.
def part_4():
    X = np.array([[1,1],[2,3],[3,3]]).T
    Y = np.array([[1],[3]])

    ##### ADD YOUR CODE FOR BOTH PARTS OF 4 HERE

# Part 5.
def part_5():
    X1 = np.array([[100,0.001],[200,0.001],[-200,0.0005]])
    X2 = np.array([[100,100],[200,100],[-200,100]])
    Y = np.array([[1],[1],[1]])
    losses, w = gradient_descent(X1, Y, 0.5e-06, 50)
    print(losses)
    print(w)

    ##### ADD YOUR CODE FOR ALL PARTS OF 5 HERE
part_5()


