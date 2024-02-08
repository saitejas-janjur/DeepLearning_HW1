import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "D:\TAMU\Spring 2024\CSCE 636 Deep Learning\HW1\data"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE

    plt.scatter(X[y == 1, 0], X[y == 1, 1], alpha=0.5, label='Class 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], alpha=0.5, label='Class -1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Training Features Visualization')
    plt.savefig('train_features.png')
    plt.show()
    ### END YOUR CODE

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training. 

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 1 or -1.
		W: An array of shape [n_features,].
	
	Returns:
		No return. Save the plot to 'train_result_sigmoid.*' and include it
		in submission.
	'''
	### YOUR CODE HERE
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                           np.arange(x2_min, x2_max, 0.1))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    probs = logistic_regression.predict_proba(np.hstack((np.ones((grid.shape[0], 1)), grid)))
    probs = probs.reshape(xx1.shape)
    plt.contourf(xx1, xx2, probs, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='b')
    plt.title('Sigmoid Model Results')
    plt.savefig('train_result_sigmoid.png')
    plt.show()
	### END YOUR CODE

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training. 

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 0,1,2.
		W: An array of shape [n_features, 3].
	
	Returns:
		No return. Save the plot to 'train_result_softmax.*' and include it
		in submission.
	'''
	### YOUR CODE HERE
    import numpy as np
import matplotlib.pyplot as plt

def visualize_result_multi(X, y, model):
    # # Create a mesh to plot the decision boundaries
    # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
    #                      np.arange(y_min, y_max, 0.02))

    # # Predict class using model and make grid predictions
    # Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)

    # # Choose a color palette with seaborn.
    # n_classes = len(np.unique(y))
    # palette = np.array(sns.color_palette("hsv", n_classes))

    # # Plot the prediction contours
    # plt.figure(figsize=(8, 6))
    # plt.contourf(xx, yy, Z, alpha=0.5, levels=np.arange(n_classes + 1) - 0.5, cmap=plt.cm.Paired)
    # plt.contour(xx, yy, Z, colors='k', linestyles=':', linewidths=0.5)

    # # Plot the training points
    # sc = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=35, edgecolors='k')

    # # Add a color bar with class labels
    # plt.colorbar(sc, ticks=np.arange(n_classes))

    # # Label the axes
    # plt.xlabel('Symentry')
    # plt.ylabel('Internsity')
    # plt.title('Multiclass Logistic Regression Model Decision Boundry')
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    # plt.xticks(())
    # plt.yticks(())

    # # Save the figure
    # plt.savefig('train_result_multi.png')
    plt.show()
    #testing
    #testting1

	### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0] 

#    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check BGD, SGD, miniBGD
   
   
    
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    print(f"Training Accuracy: {logisticR_classifier.score(train_X, train_y)}")
    print(f"Validation Accuracy: {logisticR_classifier.score(valid_X, valid_y)}")

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    learning = [0.01, 0.1, 0.5]
    epoch = [100, 500]

    results = []
    for lr in learning:
        for iter in epoch:
            logisticR_classifier_tuning = logistic_regression(learning_rate=lr, max_iter=iter)
            logisticR_classifier_tuning.fit_miniBGD(train_X, train_y, 10)
            results.append((lr, iter, logisticR_classifier_tuning.score(train_X, train_y)))

    for lr, iter, acc in results:
        print(lr, iter, acc)
   
    ### END YOUR CODE

	# Visualize the your 'best' model after training.
    # visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())

    ### YOUR CODE HERE

    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE

    ### END YOUR CODE

    
    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  miniBGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    print(f"Multiclass Training Accuracy: {logisticR_classifier_multiclass.score(train_X, train_y)}")
    print(f"Multiclass Validation Accuracy: {logisticR_classifier_multiclass.score(valid_X, valid_y)}")

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    results = []
    for lr in learning:
        for iter in epoch:
            logisticR_classifier_multiclass_tuning = logistic_regression_multiclass(learning_rate=lr, max_iter=iter, k= 3)
            logisticR_classifier_multiclass_tuning.fit_miniBGD(train_X, train_y, 10)
            results.append((lr, iter, logisticR_classifier_multiclass_tuning.score(train_X, train_y)))

    for lr, iter, acc in results:
        print(lr, iter, acc)
    ### END YOUR CODE

	


    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
    raw_test_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_X = prepare_X(raw_test_data)
    test_y, _ = prepare_y(test_labels)
    ### END YOUR CODE

    
    # Visualize the your 'best' model after training.
    visualize_result(test_X[:, 1:3], test_y, logisticR_classifier_multiclass.get_params())

    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE

    ### END YOUR CODE






    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1   

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE

    ### END YOUR CODE
    
    best_logisticR = logisticR_classifier

    # Assuming best_logisticR is your best binary logistic regression model after hyperparameter tuning
    binary_test_accuracy = best_logisticR.score(test_X, test_y)
    print(f"Binary Test Accuracy: {binary_test_accuracy}")

    best_logistic_multi_R = logisticR_classifier_multiclass

    # Assuming best_logistic_multi_R is your best multiclass logistic regression model after hyperparameter tuning
    multiclass_test_accuracy = best_logistic_multi_R.score(test_X, test_y)
    print(f"Multiclass Test Accuracy: {multiclass_test_accuracy}")

    ################Compare and report the observations/prediction accuracy


'''
Explore the training of these two classifiers and monitor the graidents/weights for each step. 
Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
'''
    ### YOUR CODE HERE

    ### END YOUR CODE

    # ------------End------------
    

if __name__ == '__main__':
	main()