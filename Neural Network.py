#Neural Network

import numpy as np
import random
mu = .5

class NeuralNetwork(object):
	# layer sizes should be a vector that contains the sizes of each layer
	# the first and last layer are the input and output layer resp.
	# note: init isnt needed since we will use the same layout for each sample......
	def __init__(self, layer_sizes,activation ='sigmoid',cost='L2',optimize = 'SGD', weight = 'normal'):
		# the input and output sizes should be preallocated
		# weights are a 3-D array with the w[0] containing the weight matrix from 1st to 2nd layer
		# bias is a similar 2-D array with b[0] being the bias of the 2nd layer since the first is composed of input
		self.optimize_function = optimize
		self.cost = cost
		self.activation_function = activation
		self.size = len(layer_sizes)
		self.sizes = layer_sizes
		if weight =='normal':
			self.weights = [np.random.randn(nxt_layer,prev_layer) for nxt_layer,prev_layer \
					   in zip(layer_sizes[1:], layer_sizes[:-1])] #/np.sqrt(prev_layer
		else:
	   		self.weights = [np.random.randn(nxt_layer,prev_layer)/np.sqrt(prev_layer) for nxt_layer,prev_layer \
					   in zip(layer_sizes[1:], layer_sizes[:-1])] 
		self.bias = [np.random.randn(layer_size) for layer_size in layer_sizes[1:]]
		self.activation = [np.zeros(x) for x in layer_sizes[:]]
	# set the bias and weights to an gaussian dist. value so we can have something to optimize in G.D.
	# func to calc the activation energy of each layer
	
	def predict(self,data,labels):
		predictions = np.array([np.argmax(self.feed_forward2(sample))+1 for sample in data])
		#return np.sum(predictions==labels)/np.float32(n)
		return predictions
	def score(self,data,labels):
		'''
			This will be used to score the testing and validation sets.
			Need to prepare the data in some way to be able to read it this way.
			data: NxD where N is the sample size and D is 784 or 28^2 
			labels: Nx1 vector holding a value coresspond to an image type

			Note: For each run the activation will change
		'''
		n = data.shape[0]
		labels = labels.reshape(labels.shape[0])
		predictions = self.predict(data,labels)
		return np.sum(predictions==labels)/np.float32(n)


	# Section of Cost functions
	def cross_entropy(self,output,label):
		return np.sum(np.nan_to_num(-label*np.log(output)-(1-label)*np.log(1-output)))
	
	def cross_entropy_prime(self,z,output,label):
		return (output-label)

	def L2(self,output,label):
		return .5*np.array((output-label))**2
	
	def L2_prime(self, sigmoid_prime, output,label):
		return (output - label)*sigmoid_prime
	
	# Section of Activation Functions
	def tanh(self,z):
		return np.tanh(z)
	
	def tanh_prime(self,z):
		return 1 - (np.tanh(z))*(np.tanh(z))
	
	def reLU(self,z):
		z[z<0] = 0
		return z
	
	def reLU_prime(self,z):
		z[z<0] = 0
		z[z>0] = 1
		return z
	
	def sigmoid_func(self,z):
		'''
			Used to calc. the sigmoid that determines the activation energy of the previous step
		'''
		return 1.0/(1.0+np.exp(-z))
	
	def sigmoid_prime(self,z):
		'''
		 	Working through the derivative can see the solution can be rewritten in this form for 
		 	faster computation and reuses the prev function
		'''
		return self.sigmoid_func(z)*(1.0-self.sigmoid_func(z))
	
	def softmax(self,z):
		return np.exp(z)/np.sum(np.exp(z))
	
	def softmax_prime(self,z):
		return self.softmax(z) * (1 - self.softmax(z))
	
	def train(self,train_data, train_labels, epochs, batch_size, epsilon,lmda,test_data=None,test_labels=None):
		'''
			data: NxD array with N samples each in D dimension
			labels: Nx1 vector w/ labels for each of the samples in data
			epochs: how many times SGD performs over the entire data set
			batch_size: parameter to be adjusted
			epsilon: parameter to be adjusted in SGD

			Note: may want to split this function up but serves intuitive purpose to understanding
		'''
		data = zip(train_data, train_labels)
		n = len(data)
		scores = []
		epoch = []
		train_error = []
		# this will give us intuition onto how the NN is improving when classifying
		# since we are displaying the test scores after each epoch
		for i in xrange(epochs):
			random.shuffle(data)
			batches = [data[k:k+batch_size] for k in xrange(0,n,batch_size)]
			for batch in batches:
				self.update_batch(batch,epsilon,lmda,n)
			if test_data!=None and test_labels!=None:
				score = self.score(test_data,test_labels)
				of = self.score(train_data,train_labels)
				print "Epoch {0} Validation {1}".format(i, score)
				epoch.append(score)
				train_error.append(of)
			else:
				print "Epoch {0} complete".format(i)
		print self.score(train_data,train_labels)
		return epoch, train_error

	def update_batch(self,batch,eps,lmda,n):
		'''
			Use SGD to calc the update (once) on the weight and bias to try to get optimal values
			Psuedo-Code:
				1) For each data sample and label run back_prop
				2) Add up the gradient of the cost in terms of the batch
				3) Perform one iteration of grad descent with these values
		'''
		if self.optimize_function=='M_SGD' or self.optimize_function == 'M_SGD_Decay':
			vb = [np.zeros(b.shape) for b in self.bias]
			vw = [np.zeros(w.shape) for w in self.weights]
		
		sum_b = [np.zeros(b.shape) for b in self.bias]
		sum_w = [np.zeros(w.shape) for w in self.weights]
		size_batch = np.float32(len(batch))
		for data,label in batch:
			# 1) / 2)
			delta_w,delta_b = self.back_prop(data,label)
			sum_b = [nb+dnb for nb, dnb in zip(sum_b, delta_b)]
			sum_w = [nw+dnw for nw, dnw in zip(sum_w, delta_w)]
		# 3) Perform one iter of GD we use LC since each 
		if self.optimize_function=='SGD':
			self.weights = [w-(eps/size_batch)*nw for w, nw in zip(self.weights, sum_w)]
			self.bias = [b-(eps/size_batch)*nb for b, nb in zip(self.bias, sum_b)]
		
		if self.optimize_function=='SGD_Decay':
			self.weights = [(1 - eps*lmda/n)*w-(eps/size_batch)*nw for w, nw in zip(self.weights, sum_w)]
			self.bias = [b-(eps/size_batch)*nb for b, nb in zip(self.bias, sum_b)]
		
		if self.optimize_function=='M_SGD':
			vb = [mu*v - (eps/size_batch)*nb  for v,nb in zip(vb,sum_b)]
			vw = [mu*v - (eps/size_batch)*nw for v,nw in zip(vw,sum_w)]
			self.weights = [w + v for w, v in zip(self.weights, vw)]
			self.bias = [b + v for b, v in zip(self.bias,vb)]
		
		if self.optimize_function=='M_SGD_Decay':
			vb = [mu*v - (eps/size_batch)*nb  for v,nb in zip(vb,sum_b)]
			vw = [mu*v - (eps/size_batch)*nw for v,nw in zip(vw,sum_w)]
			self.weights = [(1 - eps*lmda/n)*w + v for w, v in zip(self.weights, vw)]
			self.bias = [b + v for b, v in zip(self.bias,vb)]
	def back_prop(self, data,label):
		'''	
			Goal: To calculate how much each weight and bias affects the resulting activation of each node
			So for each weight and bias going to calc the derivative which can be done by calculating the 
			activation from the last layer to the second to first.
			1) forward propagate to the output
			2) Calc the error of the last layer and get grad(C)*sigma'(z) 
				- while trying to get the error it helps to calc the z term in f.f. for each node since its used in b.p.
			3) Calc delta for each layer and subsequently each bias and weight affect on activation
		'''
		if self.cost == 'L2':
			cost_function = self.L2_prime
		if self.cost == 'cross_entropy':
			cost_function = self.cross_entropy_prime
		# 1)
		z = True
		sig_prime = self.feed_forward2(data,z)
		#print sig_prime[-1].shape # messed up here so feed forward wrong
		# sig_prime has vectors corresponding to levels other than the input layer
		label_vector = np.zeros(self.activation[-1].shape[0])
		label_vector[label-1] = 1
		# 2) 
		# init the delta matrices
		# zeros_like will not copy correctly so use list def to copy
		delta_b = [np.zeros(b.shape) for b in self.bias]
		delta_w = [np.zeros(w.shape) for w in self.weights]
		#row vector of size = output layer
		#print 'activation shape:', self.activation[-1].shape
		#print 'label vec shape:', label_vector.shape
		#grad_c = np.array(self.activation[-1] - label_vector)
		# delta is just a vector with no second size
		delta = cost_function(sig_prime[-1],self.activation[-1],label_vector)
		delta_b[-1] = delta
		delta = delta.reshape(delta.shape[0],1)
		activation = self.activation[-2].reshape(1,self.activation[-2].shape[0])
		delta_w[-1] = np.dot(delta,activation)
		# 3)
		'''
			Its important to be wary of the indicies since we are using multiple levels of arrays and vectors
		'''
		layer = range(self.size)[1:]
		layer = np.array(layer[-2::-1])-1
		#print 'LAYER',layer -> [1,0]
		for l in layer:
			wd = np.dot(self.weights[l+1].T,delta).reshape(self.weights[l+1].T.shape[0])
			delta = wd*sig_prime[l]
			delta_b[l] = delta
			delta = delta.reshape(delta.shape[0],1)
			activation = self.activation[l].reshape(1,self.activation[l].shape[0])
			delta_w[l] = np.dot(delta,activation)
		return delta_w,delta_b

	def feed_forward(self,sample,store_z=False):
		'''
			Completes the feed forward portion of the alg. Since we have activation as a matrix we set it's [0] value to that
			of the data to get the next layers activation. We dont need to return the activations since they are already stored.
			We use the store_z var to differentiate from the back_prop use of the function and the eval function use of it.

			Note: activaiton_prime[0] corresponds to the 2nd layer \sigma'(z) since the input layer has none
		'''
		a_func = None
		a_prime = None
		if self.activation_function == 'tanh':
			a_func = self.tanh
			a_prime = self.tanh_prime
		if self.activation_function == 'sigmoid':
			a_func = self.sigmoid_func
			a_prime = self.sigmoid_prime
		if self.activation_function == 'reLU':
			a_func = self.reLU
			a_prime = self.reLU_prime

		self.activation[0] = sample
		if store_z == False:
			for i in range(self.size-1):
				self.activation[i+1] = a_func(np.dot(self.weights[i],self.activation[i]) + self.bias[i])
			return self.activation[-1]
		else:
			activation_prime = [np.zeros(z) for z in self.sizes[1:]]
			for i in range(self.size-1):
				#print 'IN FF'
				z =  np.dot(self.weights[i],self.activation[i]) + self.bias[i]
				activation_prime[i] = a_prime(z)
				self.activation[i+1] = a_func(z)
			return activation_prime

	def feed_forward2(self,sample,store_z=False):
		'''
			Now we have added soft max output layer and initial back prop layer
			Completes the feed forward portion of the alg. Since we have activation as a matrix we set it's [0] value to that
			of the data to get the next layers activation. We dont need to return the activations since they are already stored.
			We use the store_z var to differentiate from the back_prop use of the function and the eval function use of it.

			Note: activaiton_prime[0] corresponds to the 2nd layer \sigma'(z) since the input layer has none
		'''
		a_func =None
		a_prime= None
		if self.activation_function == 'tanh':
			a_func = self.tanh
			a_prime = self.tanh_prime
		if self.activation_function == 'sigmoid':
			a_func = self.sigmoid_func
			a_prime = self.sigmoid_prime
		if self.activation_function == 'reLU':
			a_func = self.reLU
			a_prime = self.reLU_prime

		activation_prime = [np.zeros(z) for z in self.sizes[1:]]
		self.activation[0] = sample
		if store_z == False:
			for i in range(self.size-2):
				self.activation[i+1] = a_func(np.dot(self.weights[i],self.activation[i]) + self.bias[i])
			z = np.dot(self.weights[-1], self.activation[-2])+self.bias[-1]
			self.activation[-1] = self.softmax(z)
			return self.activation[-1]
		else:
			for i in range(self.size-2):
				#print 'IN FF'
				z =  np.dot(self.weights[i],self.activation[i]) + self.bias[i]
				activation_prime[i] = a_prime(z)
				self.activation[i+1] = a_func(z)
			z = np.dot(self.weights[-1], self.activation[-2])+self.bias[-1]
			self.activation[-1] = self.softmax(z)
			activation_prime[-1] = self.softmax_prime(z) 
			return activation_prime
