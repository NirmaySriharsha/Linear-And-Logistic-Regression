import numpy as np
import matplotlib.pyplot as plt
import pickle

# NOTE: The things to complete are marked by "##### ADD YOUR CODE HERE".

""" PLEASE use this completed function for for plot generation,
    for consistency among submitted plots. """
def plot(train_losses, train_accs, test_losses, test_accs, prefix):
    num_epochs = len(train_losses)
    plt.plot(range(num_epochs),train_losses,label='train loss', marker='o',linestyle='dashed',linewidth=1,markersize=2)
    plt.plot(range(num_epochs),test_losses,label='test loss', marker='o',linestyle='dashed',linewidth=1,markersize=2)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('log loss')
    plt.savefig('./{}_loss.pdf'.format(prefix))
    plt.clf()
    plt.plot(range(num_epochs),train_accs,label='train acc',marker='o',linestyle='dashed',linewidth=1,markersize=2)
    plt.plot(range(num_epochs),test_accs,label='test acc',marker='o',linestyle='dashed',linewidth=1,markersize=2)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('./{}_acc.pdf'.format(prefix))

""" This function should add repeated features,
    as needed for Part 3b."""
def augment(xs, vocab_idx, num_repeats):
    augment_col = xs[:,[vocab_idx]]
    augment_col = np.tile(augment_col, (1,num_repeats))
    xs_new = np.concatenate([xs, augment_col], axis = 1)
    return xs_new
    ##### ADD YOUR CODE HERE

""" This function computes the logistic loss on (xs, ys).
    Please compute the MEAN over the dataset vs. the SUM,
    the latter of which was used in the preceding theory
    section."""
def loss(params, xs, ys):
    ##### ADD YOUR CODE HERE
    w = params[0]
    b = params[1]
    z = xs@w + b
    loss = (np.squeeze(np.sum(np.log(1+np.exp(z))) - z.T@ys))/len(ys)
    return loss


""" This function computes the accuracy on (xs, ys)."""
def acc(params, xs, ys):
    w = params[0]
    b = params[1]
    z = xs@w + b
    predictions = []
    for i in range(len(z)):
        if z[i] >= 0:
            predictions.append(1)
        else:
            predictions.append(0)
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == ys[i]:
            count = count + 1
    accuracy = count/len(ys)
    return accuracy
    ##### ADD YOUR CODE HERE

""" This function should compute the gradient of the
    logistic loss with respect to the parameters.
    The loss should be a MEAN over the dataset vs. the SUM,
    the latter of which was used in the preceding theory
    section.
    To ensure your code runs in a reasonable amount of time,
    it's important to vectorize this computation. If you're
    not sure what this means, please come to office hours. """
def get_gradients(params, xs, ys):
    w = params[0]
    b = params[1]
    z = xs@w +b
    A = (1/(1+np.exp(-z))) - ys
    grad_1 = (A.T@xs)/len(ys)
    grad_2 = np.sum(A)/len(ys)
    return grad_1.T, grad_2

    ##### ADD YOUR CODE HERE

""" This function applies a gradient descent update
    step to params, using the provided computed gradients
    and learning_rate, and returns the new params."""
def apply_gradients(params, gradients, learning_rate):
    params_new_w = params[0] - learning_rate * gradients[0]
    params_new_b = params[1] - learning_rate * gradients[1]
    return params_new_w, params_new_b
    ##### ADD YOUR CODE HERE

""" Run this function for Parts 1 & 2.
    This function runs gradient descent on (train_xs, train_ys)
    for num_epochs, using the specified learning_rate.
    The inputs train_xs, train_ys, test_xs, test_ys are obtained
    directly from the HW 1/HW 2 dataset dictionary.
    The labels are converted to {0,1} at the start of the function.
    The parameters are initialized at 0.
    The final metrics, along with parameters, are returned.
    YOU DO NOT NEED TO MODIFY ANY PARTS OF THIS FUNCTION. """
def train(train_xs, train_ys, test_xs, test_ys, num_epochs, learning_rate):
    # Convert y to {0,1}. Map 1-->1 and 2-->0. Recall that original labels are in {1,2}.
    for i in range(len(train_ys)): train_ys[i,0] = 1 if train_ys[i,0] == 1 else 0
    for i in range(len(test_ys)): test_ys[i,0] = 1 if test_ys[i,0] == 1 else 0
    
    # Initialize w and b at 0.
    d = train_xs.shape[1]
    w = np.zeros((d,1))
    b = 0.0
    
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    for epoch in range(num_epochs):
        # Perform an update step.
        #print("Epoch Number")
        #print(epoch)
        gradients = get_gradients((w,b), train_xs, train_ys)
        w,b = apply_gradients((w,b), gradients, learning_rate)
    
        # Compute and store train and test metrics.
        train_losses.append(loss((w,b), train_xs, train_ys))
        train_accs.append(acc((w,b), train_xs, train_ys))
        test_losses.append(loss((w,b), test_xs, test_ys))
        test_accs.append(acc((w,b), test_xs, test_ys))

    return train_losses, test_losses, train_accs, test_accs, w, b

""" Run this function for Part 3a - passing in the appropriate arguments.
    You will first need to complete the missing pieces above.
    YOU DO NOT NEED TO MODIFY ANY PARTS OF THIS FUNCTION. """
def q3a(train_xs, train_ys, test_xs, test_ys, num_epochs, learning_rate, num_features):
    train_xs = train_xs[:,:num_features]
    test_xs = test_xs[:,:num_features]
    return train(train_xs, train_ys, test_xs, test_ys, num_epochs, learning_rate)

""" Run this function for Part 3b - passing in the appropriate arguments.
    You will first need to complete the missing pieces above.
    YOU DO NOT NEED TO MODIFY ANY PARTS OF THIS FUNCTION. """
def q3b(train_xs, train_ys, test_xs, test_ys, num_epochs, learning_rate, num_features, repeat_idx, num_repeats):
    train_xs = train_xs[:,:num_features]
    test_xs = test_xs[:,:num_features]
    train_xs = augment(train_xs, repeat_idx, num_repeats)
    test_xs = augment(test_xs, repeat_idx, num_repeats)
    return train(train_xs, train_ys, test_xs, test_ys, num_epochs, learning_rate)


#with open("hw2data_TA.pkl", "rb") as f:
#    data = pickle.load(f)

#train_xs = data["XTrain"]
#train_ys = data["yTrain"]
#test_xs = data["XTest"]
#test_ys = data["yTest"]

#4.2.1
#train_losses, test_losses, train_accs, test_accs, w, b = train(train_xs, train_ys, test_xs, test_ys, 100, 0.1)
#print(test_accs[-1])
#plot(train_losses, train_accs, test_losses, test_accs, "Q4part1")

#4.2.2
#train_losses, test_losses, train_accs, test_accs, w, b = train(train_xs, train_ys, test_xs, test_ys, 3000, 0.1)
#print(test_accs[-1])
#plot(train_losses, train_accs, test_losses, test_accs, "Q4part2")

#4.2.3(a)
#train_losses, test_losses, train_accs, test_accs, w, b = q3a(train_xs, train_ys, test_xs, test_ys, 5000, 0.1, 10)
#print(train_accs[-1])
#print(test_accs[-1])

#4.2.3(b)
#train_losses, test_losses, train_accs, test_accs, w, b = q3b(train_xs, train_ys, test_xs, test_ys, 5000, 0.1, 10, 4, 500)
#print(train_accs[-1])
#print(test_accs[-1])