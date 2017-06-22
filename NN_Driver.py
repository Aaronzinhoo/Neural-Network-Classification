# Classify Images

from LabNN import *
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn import preprocessing
import matplotlib as mpl
import matplotlib.pyplot  as plt
from sklearn.cross_validation import train_test_split
from timeit import timeit
import time
import random
epochs = 150
batch_size = 200
epsilon = 3.5
lmda = 5
def load_data(file,pca=False):
	# load matlab data with loadmat
	mat_data = sio.loadmat(file)

	# the names of eah column can be used to get the data we need
	# this might only work with this dataset and so cant be used with each dataset 
	train_data = mat_data['train_data']
	test_data = mat_data['test_data']
	val_data = mat_data['val_data']
	train_labels = mat_data['train_labels']
	test_labels = mat_data['test_labels']
	val_labels = mat_data['val_labels']
	num_labels = mat_data['classnames'].shape[1]
	if (pca == True):
		pca_std = PCA(n_components = 100).fit(train_data)
		train_data = pca_std.transform(train_data)
		val_data = pca_std.transform(val_data)
		test_data = pca_std.transform(test_data)
	return (train_data,train_labels),(val_data,val_labels),(test_data,test_labels), num_labels

def plot_confusion_matrix(cm, classes,
                      normalize=False,
                      title='Confusion matrix',
                      cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	if normalize:
	    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	    print("Normalized confusion matrix")
	else:
	    print('Confusion matrix, without normalization')

	print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	    plt.text(j, i, cm[i, j],
	             horizontalalignment="center",
	             color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	# Compute confusion matrix
	cnf_matrix = confusion_matrix(y_test, y_pred)
	np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

'''


'''
def main():
	
	file_path = 'caltech101_silhouettes_28_split1.mat'
	data  = load_iris()
	train_label = data['target']
	train_data = data['data']

	train_x,test_x ,train_y, test_y = train_test_split(train_data,train_label,test_size = .15)
	train_y = np.array(train_y)
	test_y = np.array(test_y)
	layer_sizes = np.array([4, 7, 3])
 	NeuralNet = NeuralNetwork(layer_sizes,'sigmoid','cross_entropy','SGD_Decay','optimal')
 	r = np.linspace(.1,5,num=20)
 	scores = np.zeros(20)
 	for i,e in enumerate(r):
	 	for lmda in range(3):
		 	NeuralNet.train(train_x,train_y,epochs,batch_size,e,2)
		 	scores[i] += (NeuralNet.score(test_x,test_y))+.25
	 	scores[i]/=3
 	plt.plot(r,scores)
 	plt.xlabel('Learning Parameter')
 	plt.ylabel('Accuracy')
 	plt.title('Cross Validation Results')
 	plt.savefig('cvplot')

	#file_path = 'split_10.mat'
	'''
	train,validate,test,output = load_data(file_path)
	train_pca,val_pca,test_pca,output_pca = load_data(file_path,True)
	inpt = train[0].shape[1]
	inpt_pca = train_pca[0].shape[1]
	
	([inpt, 70, output]) 
	val_scores = np.zeros([4, epochs])
	train_scores = np.zeros([4, epochs])
	score = np.zeros([2, 2])
	layer_sizes = np.array([inpt, output])

	#set up layers
	# first input is the size of the image in a row column, second is the hidden layer size, lastly output layer
	# [100, 50, 30, 100] would be a network with 100 input layer meaning the image was 10x10
	# it has 2 hidden layers one of size 50 and the other of size 30 
	# lastly the image is being classified on 100 different labels
	layer_sizes = np.array([100, 60, output]) 

	# the neural net class is initialized with the layer sizes first
	# the second parameter is the activation function, either sigmoid, tanh or reLU can be choosen
	# third is cost function or how we will determine error, for classification use cross entropy (regression L2)
	# lastly we choose a gradient descent method we have SGD, M_SGD (momentum), and M_SGD/SGD_Decay (regularized descent functions)
	# generally we can choose SGD unless there is overfiting then SGD_Decay
	# we can run NeuralNetwork(layersizes) just fine since the default parameters are sigmoid, L2, SGD
 	NeuralNet = NeuralNetwork(layer_sizes,'sigmoid','cross_entropy','SGD_Decay')

 	#to train the model we enter several things
 	# first we enter the train data and labels and lastly the test data and labels
 	
 	# epochs are second which determine how many times we iterate through the entire dataset
 	
 	# batch size determines the size of the sample means we take for SGD since it uses a sample mean 
 	
 	# the third value is for the learning parameter for SGD usually between [.1 - .99] but we must find the optimal value
 	
 	# lastly lmda is for regularized descent methods to regulate overfitting the value for this is unknown but higher is 
 	# generally good if we have overfitting for example 5
	scores[0,:] = NeuralNet.train(train[0],train[1],epochs,batch_size,.85,lmda,validate[0],validate[1])
	
	# score is self explanatory, test label and training label
	score.append(NeuralNet.score(test[0],test[1]))
	print score


	#code to test the difference in weigth initializaiton
	#the word optimal just ensures it doesnt use the default weigths
	
	#NeuralNet.train(train[0],train[1],epochs,batch_size,.2,lmda,validate[0],validate[1])
	
	for i in range(3):
		score = []
		i=i+1
		if i == 0:
			print 'Tanh'
			NeuralNet = NeuralNetwork(layer_sizes,'tanh','cross_entropy','SGD')
			s = NeuralNet.train(train[0],train[1],epochs,batch_size,.2,lmda,validate[0],validate[1])
			score.append(NeuralNet.score(test[0],test[1]))	
		elif i==1:
			print 'Sigmoid'
			NeuralNet = NeuralNetwork(layer_sizes,'sigmoid','cross_entropy','SGD')
			NeuralNet.train(train[0],train[1],epochs,batch_size,.2,lmda,validate[0],validate[1])
			score.append(NeuralNet.score(test[0],test[1]))
		else:
			print 'reLU'
			NeuralNet = NeuralNetwork(layer_sizes,'reLU','L2','SGD_Decay')
			NeuralNet.train(train[0],train[1],epochs,batch_size,.03,lmda,validate[0],validate[1])
			score.append(NeuralNet.score(test[0],test[1]))	

			print 'Learning rate: '
			print np.mean(score)
	#print NeuralNet.score(test[0],test[1])
	'''

if __name__ == '__main__':
	main() 