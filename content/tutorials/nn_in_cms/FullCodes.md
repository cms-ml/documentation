---
Full codes
---

The purpose of this part is to give an example of full and working script, for each type of classification (binary and multi-class). These scripts are meant to be illustrative and are naturally improvable. They are commented quite extensively as to guide explain the different aspects of the code. Each code has two type of outputs:

* Plots which provide information about the training (eg. loss, accuracy versus epoch)
* Files which are usable for further investigation of the NN (eg. performance).

Both scripts are based on python, and use Keras. Both run on csv files where each line represents the entries for an event, and where:

* the N first columns are the N input variables: discriminating features, which can either be continuous or discrete variables. Examples: $p_T$ and spatial distribution of various reconstructed particles, invariant masses, charge and identification and isolation variables of leptons, jet- and b-tag multiplicities, b-tagging discriminant distributions, ...
* followed by a column containing the event weight:
* and then by a column containing the string specifying the physical process of the event; the way that the script expects this string is in the NameProcess.root (eg. ttbar.root) form.

This order is important as the numpy manipulation the various elements of the csv file in the script naturally takes it into account. In a certain measure, and for secondary aspects, the two codes have different functionalities, as to illustrate various possible outcomes.

### Binary

The script for the binary NN uses numpy for the manipulation of vectors. It is meant to turn on csv files which provide N=12 input variables. It is integrating all snippets mentioned in previous sections. Its outcomes are:

* 3 plots: TrnVal_loss.png, TrnVal_accuracy.png, TrnVal_roc.png, respectively showing the evolution of the loss, accuracy, and roc curves as a function of the epoch, these for the training and validation samples.
* A file for weights: best_weights.h5 which has the weights of the NN corresponding to the epoch where the validation loss is the lowest, ie. the best weights one can get from the NN training. This file is useful downstream calculations (eg. for calculating the performance of the NN).
* 2 files val_output.csv and test_output.csv, where the outputs of the NN are saved for the validation and test samples, and which are necessary for downstream calculations.

??? example "Full Code Binary"
	```python
	import os
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	from tensorflow.python.keras import *
	from tensorflow.python.keras.optimizer_v2.adam import Adam
	from tensorflow.python.keras.models import Sequential
	from tensorflow.python.keras.layers import Dense, Dropout, AlphaDropout
	from tensorflow.python.keras.regularizers import l2
	from tensorflow.keras.mixed_precision import experimental as mixed_precision

	#from tensorflow.python.keras.utils import metrics_utils

	import numpy
	import time
	import pandas
	from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_curve, auc
	import matplotlib.pyplot as plt
	import pickle
	import numpy as np
	import pandas as pd

	def assure_path_exists(path):
		if not os.path.exists(path):
			os.mkdir(path)

	def load_data(path, tag):
		""" Read data from CSV
		Args:
		path (str): path to the data
		tag (int): 1 if signal, 0 if background
		Returns:
		np.c_[data, tag] (np.array): array with data and tag
		label (np.array): array with the name of event process
		"""
		full_data = np.loadtxt(path, delimiter=",", dtype="str")
		tag = np.ones(len(full_data))*tag
		label = full_data[:, -1]
		data = np.array(full_data[:,:-1], dtype="float")
		return np.c_[data, tag], label

	def mae(y_true, y_pred):
		n = len(y_true)
		error = np.abs(y_true - y_pred)
		return error / n

	if __name__ == "__main__":
		import argparse
		import sys

		# Uses the "weight" variable of roottuples: provided as (N(input vars)+1)th variable in csv files
		# Passed on weightTrn & weightVal for appropriate samples
		# Weighting each event by these vectors: effectively weighting samples of various XS & Nevts, thus
		# no need of a single "Background.root"
		
		# Input arguments from command line
		parser = argparse.ArgumentParser(description='Process the command line options')
		parser.add_argument('-v', '--verbose', action='store_true', help='Whether to print verbose output')
		parser.add_argument('-y', '--year', type=str, required=True, help='Year of data being trained')
		parser.add_argument('-c', '--channel', type=str, required=True, help='Final state of data being trained')
		parser.add_argument('-n', '--name', type=str, required=True, help='model name')

		args = parser.parse_args()

		### Read input arguments
		year = args.year
		channel=args.channel
		name=args.name
		verbose = 0
		if args.verbose:
			verbose = 1
		###

		### NN parameters
		activ = "relu"
		learning_rate = 5.e-3
		decay_rate = 0.
		# Each number (here 50) is the number of nodes (Nnode) in each layer.
		# The number of different Nnode (here 2) is the number of layers.
		NodeLayer = "50 50"
		architecture=NodeLayer.split()
		ini = "he_normal" # Function for initialising the random weights of the layer
		n_epochs = 2000
		batch_size = 5000
		dropout_rate = 0.
		###
		
		# Model's compilation arguments, training parameters and optimizer
		compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}
		trainParams = {'epochs': n_epochs, 'batch_size': batch_size, 'verbose': verbose}
		print(Adam)
		# Define optimizer which has the learning- & decay-rates
		myOpt = Adam(lr=learning_rate, decay=decay_rate)
		# Protect from rounding errors: Number by which the loss-function is multiplied to protect from rounding errors
		#                               dynamically scales the loss to prevent underflow
	#    myOpt = mixed_precision.LossScaleOptimizer(myOpt, 1e5)
		compileArgs['optimizer'] = myOpt

		# Creating the directory where the files will be stored
		dirname = year+"_"+channel+"_"+name
		filepath = "models/{}/".format(dirname)
		if not os.path.exists(filepath):
			os.mkdir(filepath)

		# Printing info and starting time
		if args.verbose:
			print("Dir "+filepath+" created.")
			print("Starting the training")
			start = time.time()

		##### Build the model
		model = Sequential()

		# 1st hidden layer: it has as many nodes as provided by architecture[0]: 12 b/c that much input variables
		model.add(Dense(int(architecture[0]), input_dim=12, activation=activ, kernel_initializer=ini))

		i=1
		while i < len(architecture) :
			model.add(Dense(int(architecture[i]), activation=activ, kernel_initializer=ini))
	#        model.add(kernel_regularizer=l2(1e-5))
	#        model.add(Dropout(dropout_rate))
			i=i+1
		model.add(Dense(1, activation='sigmoid')) # Output layer: 1 node, with sigmoid

		model.compile(**compileArgs)
		model.summary()
		#####
		
		# Read input data
		print("LOADING DATA")
		# Do the labeling for S & B to 1 & 0 for train, val and test samples
		# For train sample: avoid writing the value of label with "_", 
		train_sig, _ = load_data("data/train_sig.csv", 1)
		train_bkg, _ = load_data("data/train_bkg.csv", 0)
		# validation sample
		val_sig, label_val_sig = load_data("data/val_sig.csv", 1)
		val_bkg, label_val_bkg = load_data("data/val_bkg.csv", 0)
		val_label = np.concatenate((label_val_sig, label_val_bkg))
		val_label = np.array([l[:-5] for l in val_label]) # Keep all file/process name except ".root" to identify processes
		# test sample
		test_sig, label_test_sig = load_data("data/test_sig.csv", 1)
		test_bkg, label_test_bkg = load_data("data/test_bkg.csv", 0)
		test_label = np.concatenate((label_test_sig, label_test_bkg))
		test_label = np.array([l[:-5] for l in test_label]) #  Keep all file/process name except ".root" to identify processes
		# Concatenation of variables and weights only
		# NB: could have been done in the general form (concatenate the total object, as above)
		test_data = np.concatenate((test_sig[:, :-2], test_bkg[:, :-2]))
		test_weight = np.concatenate((test_sig[:, -2], test_bkg[:, -2]))

		# Weight normalisation, event balancing:
		# Render the sum of S events equal to sum of B events
		# The totality of weights should be equal, even though the relative weights
		# among B or S signal are correct by weighting each event by the "weight"
		# variable in the roottuples
	#    train_sig[:, -2]*=np.sum(train_bkg[:, -2])/np.sum(train_sig[:, -2])
		# AND
		# each sample should be numerically not too small: normalize by the N(B),
		# both for S & B, in both training & validation sample
		train_bkg[:,-2] *=  train_bkg.shape[0] / np.sum(train_bkg[:,-2])
		train_sig[:,-2] *=  train_bkg.shape[0] / np.sum(train_sig[:,-2])
		# Save unweighted values of validation weights, to be written in val_output.csv
		full_val_not_scaled = np.concatenate((val_sig, val_bkg))
		weightVal_not_scaled = full_val_not_scaled[:, -2]
		val_bkg[:,-2] *=  val_bkg.shape[0] / np.sum(val_bkg[:,-2])
		val_sig[:,-2] *=  val_bkg.shape[0] / np.sum(val_sig[:,-2])

		# Concatenate signal and background for train and validation
		full_train = np.concatenate((train_sig, train_bkg))
		full_val = np.concatenate((val_sig, val_bkg))

		# Variables normalisation to [-1,+1]
		# Looping over number of variables:
		#   shape = Dimension of the full_train (variables * Nevts) array
		#   shape[0] gives lines: the dimension is Nevts
		#   shape[1] gives columns: the dimension here is 12 (for input vars.) + 1 (weight) + 1 (tag=0,1)
		for var in range(full_train.shape[1] - 2):
		top = np.max(full_train[:, var]) # checks all lines by value of (variable in) column var
		bot = np.min(full_train[:, var])
		full_train[:, var] = (2*full_train[:, var] - top - bot)/(top - bot)
		full_val[:, var] = (2*full_val[:, var] - top - bot)/(top - bot)
		test_data[:, var] = (2*test_data[:, var] - top - bot)/(top - bot)

		np.random.shuffle(full_train) # Without this line: NN first exposed to S for many events: b/c fullt_train = np.concatenate((train_sig, train_bkg))
		xTrn = full_train[:, :-2] # Input of NN (kinematic variables)
		yTrn = full_train[:, -1] # Target of NN (1 for signal, 0 for background)
		weightTrn = full_train[:, -2]
		xVal = full_val[:, :-2]
		yVal = full_val[:, -1]
		weightVal = full_val[:, -2]

		# Define criterium for best epoch saving
		# Returns the weights as they are at the minimum, b/c of argument of mode
		checkpoint = callbacks.ModelCheckpoint(
		filepath=filepath+"best_weights.h5",
		verbose=1,
		save_weights_only=True,
		monitor="val_loss",
		mode="min",
		save_best_only=True)

		### Train model ###
		history = model.fit(xTrn, yTrn, validation_data=(xVal,yVal,weightVal), sample_weight=weightTrn,shuffle=True, callbacks=[checkpoint], **trainParams)

		loss = history.history['loss']
		val_loss = history.history['val_loss']
		acc = history.history["accuracy"]
		val_acc = history.history['val_accuracy']

		# Saving accuracy and loss values in a pickle file for later plotting
		pickle.dump(acc, open(filepath+"acc.pickle", "wb"))
		pickle.dump(loss, open(filepath+"loss.pickle", "wb"))
		pickle.dump(val_acc, open(filepath+"val_acc.pickle", "wb"))
		pickle.dump(val_loss, open(filepath+"val_loss.pickle", "wb"))

		# Getting the roc curve
		y_pred_Trn = model.predict(xTrn).ravel()
		fpr_Trn, tpr_Trn, thresholds_Trn = roc_curve(yTrn, y_pred_Trn)
		auc_Trn = auc(fpr_Trn, tpr_Trn)
		y_pred_Val = model.predict(xVal).ravel()
		fpr_Val, tpr_Val, thresholds_Val = roc_curve(yVal, y_pred_Val)
		auc_Val = auc(fpr_Val, tpr_Val)

		# Getting errors
	#    mae_Trn = mae(yTrn, y_pred_Trn)
	#    mae_Val = mae(yVal, y_pred_Val)

		# Plot loss curves
		plt.figure(1)
		plt.plot(loss, label="Training loss")
		plt.plot(val_loss, label="Validation loss")
		plt.legend()    
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
	#    plt.yscale("log")
		plt.savefig(filepath+"TrnVal_loss.png")
		# Plot accuracy curves
		plt.figure(2)
		plt.plot(acc, label="Training accuracy")
		plt.plot(val_acc, label="Validation accuracy")
		plt.legend()
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
	#    plt.yscale("log")
		plt.savefig(filepath+"TrnVal_accuracy.png")
		# Plot ROC curves
		plt.figure(3)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.plot(fpr_Trn, tpr_Trn, label='Training ROC (area = {:.2f})'.format(auc_Trn))
		plt.plot(fpr_Val, tpr_Val, label='Validation ROC (area = {:.2f})'.format(auc_Val))
		plt.legend()
		plt.xlabel('False positive rate')
		plt.ylabel('True positive rate')
		plt.savefig(filepath+"TrnVal_roc.png")
	#    plt.figure(4)
	##    plt.plot([0, 100], 'k--')
	#    plt.plot(mae_Trn, label="Training MAE")
	#    plt.plot(mae_Val, label="Validation MAE")
	#    plt.legend()
	#    plt.xlabel('N(events)')
	#    plt.ylabel('MAE')
	#    plt.savefig(filepath+"TrnVal_mae.png")

		# Time of the training
		if args.verbose:
			print("Training took ", (time.time()-start)//60, " minutes")

		# Getting predictions
		if args.verbose:
			print("Getting predictions")

		model.load_weights(filepath + "best_weights.h5") # Loading best epoch weights

		# Compute and save prediction for best epoch
		trnPredict = model.predict(xTrn)
		valPredict = model.predict(xVal)
		testPredict = model.predict(test_data)
	#    np.savetxt(filepath+"val_output.csv", np.c_[valPredict, weightVal, val_label], delimiter=",", fmt="%s")
		np.savetxt(filepath+"val_output.csv", np.c_[valPredict, weightVal_not_scaled, val_label], delimiter=",", fmt="%s")
		np.savetxt(filepath+"test_output.csv", np.c_[testPredict, test_weight, test_label], delimiter=",", fmt="%s")

		# Outputs
	#    plt.savefig(filepath+"loss.png")
		val_file = open(filepath + "val_loss.pickle", "rb")
		val = pickle.load(val_file)
		print("Epoch of the best training: ",np.argmin(val))
	```

### Multiclass

The script for the multi-class NN is classifying 5 different processes (Signal, Wjets, TTbar, Z2nu, Other), and has thus 5 different classes. It uses panda for the manipulation of vectors, and is performing this manipulation slightly differently than in the binary script, but with similar outcomes. It is meant to turn on csv files which provide N=17 input variables. This script is performing 2 loops, the first being nested in the second:

* It is simply running 10 times, letting all random-based processes (initialization, shuffling, seeding) produce 10 different results, which are close in performance.
* It is scanning the decay-rate from $10^{-5}$ to $10^{-3}$.

It goes without saying that these loops can be taken out, or modified. This script has performance calculation capacity embedded in it, and doesn't need a downstream script to be run. The outcomes of the script are:

* A file 17var_fom.csv necessary for possible downstream performance calculations.
* 10 different subdirectories, corresponding to the 10 runs, with each containing:
   * 2 plots: TrnVal_loss.png, TrnVal_accuracy.png, as in the binary case.
   * A file for weights best_weights.h5, as in the binary case.
   * A subdirectory FOM_figures_runs/ which has the performance plots of all 10 runs in it.

As for the event balancing used in the case of multi-class NN, we do an event weighting similar yet different than in the "Event weighting & balancing" section. Interested users are invited to look at slides 10-13 and 23 of [this presentation](https://indico.cern.ch/event/1503118/contributions/6327106/subcontributions/539906/attachments/3037743/5366723/PBargassa_ML3.pdf); referring to these slides, the balancing scheme used in this script is the "SQRT".

??? example "Full Code Multiclass"
	```python
	import os
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	from tensorflow.python.keras import *
	from tensorflow.python.keras import backend as K
	from tensorflow.python.keras.optimizer_v2.adam import Adam
	from tensorflow.python.keras.models import Sequential
	from tensorflow.python.keras.layers import Dense, Dropout
	from tensorflow.python.keras.regularizers import l2
	#from tensorflow.keras.mixed_precision import experimental as mixed_precision

	import tensorflow as tf

	import numpy
	import time
	import pandas as pd
	from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_curve, auc
	import seaborn as sns
	import matplotlib.pyplot as plt
	import pickle
	import numpy as np
	import pandas as pd

	def assure_path_exists(path):
		if not os.path.exists(path):
			os.mkdir(path)

	# Takes strings of processes from XYZ.root and returns abbreviated name XY as name of the process
	def rename_rows(label):
		filenames = set(label)
		names = []
		for i in filenames:
			tag = i[0] + i[1]
			if tag=="Wj":
				names += [i, "Wjets"]
			elif tag=="TT":
				names += [i, "TTbar"]
			elif tag=="ZJ":
				names += [i, "Z2nu"]
			else:
				names += [i, "Other"]
		N = len(names)
		dictionary = {names[i] : names[i+1] for i in range(0,N,2)}
		return dictionary


	# Maps a 1 for a specific process and 0's elsewhere, in an array of (here 5) elements
	def Tag(dataframe):
		data = dataframe.copy() # Makes a copy -> another address in memory
		# Mapping (input mapping) of various process strings to 1, and 0's for other processes
		data["SigTag"] = data.index.map({"Signal":1}).fillna(0)
		data["WjTag"] = data.index.map({"Wjets":1}).fillna(0)
		data["TTTag"] = data.index.map({"TTbar":1}).fillna(0)
		data["Z2nuTag"] = data.index.map({"Z2nu":1}).fillna(0)
		data["OtherTag"] = data.index.map({"Other":1}).fillna(0)
		# Makes the arry of (here 5) elements made of 0/1
		return pd.concat([data["SigTag"],data["WjTag"],data["TTTag"],data["Z2nuTag"], data["OtherTag"]], axis=1)

	N_runs=10

	print("Loading data...")

	# PREPARING DATA #######################################

	### Reading data
	names = ["LepChg", "LepPt", "LepEta", "Dxy", "Dz", "RelIso", "Met", "mt", "Njet",
			"Jet1Pt", "Jet2Pt", "HT", "NbLoose", "JetHBpt", "JetHBCSV", "JetB2pt", "DrJetHBLep", "Weights", "Tag"]

	# index_col defines last column as index (row names) of DataFrame. Resulting DataFrame has xyz.root as its last column (indexing starts @ 0)
	# When interpreting it as index column it plays the same part as do numbers 0,1,2,3... in numpy array (eg. a[i] is i-th element of array a).
	# It disappears from the list of values you are "interpreting as data"
	train_sig = pd.read_csv("17var/data/train_sig.csv", header=None, names=names, index_col=18)
	train_bkg= pd.read_csv("17var/data/train_bkg.csv", header=None, names=names, index_col=18)

	val_sig = pd.read_csv("17var/data/val_sig.csv", header=None, names=names, index_col=18)
	val_bkg = pd.read_csv("17var/data/val_bkg.csv", header=None, names=names, index_col=18)

	test_sig = pd.read_csv("17var/data/test_sig.csv", header=None, names=names, index_col=18)
	test_bkg = pd.read_csv("17var/data/test_bkg.csv", header=None, names=names, index_col=18)

	### Get S=(550,520) lines
	val_sig_S = val_sig.loc['T2DegStop_550_520.root'] # Filter DataFrame rows (only 1 argument) by name
	val_sig_S.index = ['Signal'] * len(val_sig_S)

	train_sig_S = train_sig.loc['T2DegStop_550_520.root']
	train_sig_S.index = ['Signal'] * len(train_sig_S)

	### Index = category of the process ("Signal", "Wjets", "TTbar", "Z2nu", "Other")
	train_sig.index = ['Signal'] * len(train_sig) # 
	names = rename_rows(train_bkg.index) # train_bkg.index holds the name of the process from index_col=18
	train_bkg = train_bkg.rename(index=names) # Rename (here) index labels

	val_sig.index = ['Signal'] * len(val_sig)
	names = rename_rows(val_bkg.index)
	val_bkg = val_bkg.rename(index=names)

	test_sig.index = ['Signal'] * len(test_sig)
	names = rename_rows(test_bkg.index)
	test_bkg = test_bkg.rename(index=names)

	### Separate processes
	train_wj = train_bkg.loc[["Wjets"]]
	train_tt = train_bkg.loc[["TTbar"]]
	train_z2nu = train_bkg.loc[["Z2nu"]]
	train_other = train_bkg.loc[["Other"]]

	val_wj = val_bkg.loc[["Wjets"]]
	val_tt = val_bkg.loc[["TTbar"]]
	val_z2nu = val_bkg.loc[["Z2nu"]]
	val_other = val_bkg.loc[["Other"]]

	for dlta in range(0,1): # ?

		### Weight normalisation

		delta = dlta # Useless

		N_sig = train_sig.shape[0]

		S = train_sig["Weights"].sum()
		W = train_wj["Weights"].sum()
		T = train_tt["Weights"].sum()
		Z = train_z2nu["Weights"].sum()
		O = train_other["Weights"].sum()

		sig = 1.
		wj = np.sqrt(W/S)
		tt = np.sqrt(T/S)
		z2nu = np.sqrt(Z/S)
		other = np.sqrt(O/S)

		weightTrn_sig = train_sig["Weights"]*N_sig/S*sig
		weightTrn_wj = train_wj["Weights"]*N_sig/W*wj
		weightTrn_tt = train_tt["Weights"]*N_sig/T*tt
		weightTrn_z2nu = train_z2nu["Weights"]*N_sig/Z*z2nu
		weightTrn_other = train_other["Weights"]*N_sig/O*other

		N_sig = val_sig.shape[0]

		S = val_sig["Weights"].sum()
		W = val_wj["Weights"].sum()
		T = val_tt["Weights"].sum()
		Z = val_z2nu["Weights"].sum()
		O = val_other["Weights"].sum()

		sig = 1.
		wj = np.sqrt(W/S)
		tt = np.sqrt(T/S)
		z2nu = np.sqrt(Z/S)
		other = np.sqrt(O/S)

		weightVal_sig = val_sig["Weights"]*N_sig/S*sig
		weightVal_wj = val_wj["Weights"]*N_sig/W*wj
		weightVal_tt = val_tt["Weights"]*N_sig/T*tt
		weightVal_z2nu = val_z2nu["Weights"]*N_sig/Z*z2nu
		weightVal_other = val_other["Weights"]*N_sig/O*other
			
			###
			
		# Full train
		full_train = pd.concat([train_sig, train_wj, train_tt, train_z2nu, train_other])
		full_train_weights = pd.concat([weightTrn_sig, weightTrn_wj, weightTrn_tt, weightTrn_z2nu, weightTrn_other])
		weights_not_scaled_train = full_train["Weights"]
		### Only S=(550,520) for plotting
		full_train_S = pd.concat([train_sig_S, train_wj, train_tt, train_z2nu, train_other])
		weights_not_scaled_train_S = full_train_S["Weights"]

		# Full validation
		full_val = pd.concat([val_sig, val_wj, val_tt, val_z2nu, val_other])
		full_val_weights = pd.concat([weightVal_sig, weightVal_wj, weightVal_tt, weightVal_z2nu, weightVal_other])
		weights_not_scaled_val = full_val["Weights"]
		### Only S=(550,520) for plotting
		full_val_S = pd.concat([val_sig_S, val_wj, val_tt, val_z2nu, val_other])
		weights_not_scaled_val_S = full_val_S["Weights"]

		# Full test
		full_test = pd.concat([test_sig, test_bkg])
		full_test_weights = full_test["Weights"]
		weights_not_scaled_test = full_test["Weights"]

		### Variable normalisation to [-1,1]
		top = full_train.max() ; bot = full_train.min()
		full_val = (2*full_val-top-bot)/(top-bot)
		full_val_S = (2*full_val_S-top-bot)/(top-bot)
		full_test = (2*full_test-top-bot)/(top-bot)
		full_train = (2*full_train-top-bot)/(top-bot)
		full_train_S = (2*full_train_S-top-bot)/(top-bot)

			# Replace original weights with balanced weights
		full_val = pd.concat([full_val.drop("Weights", axis=1), full_val_weights], axis=1) # Along columns (axis=1): drops the column "Weights" & replaces
		full_val_S = pd.concat([full_val_S.drop("Weights", axis=1), weights_not_scaled_val_S], axis=1)
		full_train_S = pd.concat([full_train_S.drop("Weights", axis=1), weights_not_scaled_train_S], axis=1)
		full_test = pd.concat([full_test.drop("Weights", axis=1), full_test_weights], axis=1)
		full_train = pd.concat([full_train.drop("Weights", axis=1), full_train_weights], axis=1)

		### INFO for plotting
		xTrn_p = full_train.drop("Weights", axis=1)
		yTrn_p = Tag(full_train)
		weightTrn_p = weights_not_scaled_train
		###### S=(550,520)
		xTrn_S = full_train_S.drop("Weights", axis=1)
		yTrn_S = Tag(full_train_S)
		weightTrn_S = weights_not_scaled_train_S

		xVal_p = full_val.drop("Weights", axis=1)
		yVal_p = Tag(full_val)
		weightVal_p = weights_not_scaled_val
		###### S=(550,520)
		xVal_S = full_val_S.drop("Weights", axis=1)
		yVal_S = Tag(full_val_S)
		weightVal_S = weights_not_scaled_val_S

		xTest_p = full_test.drop("Weights", axis=1)
		yTest_p = Tag(full_test)
		weightTest_p = weights_not_scaled_test

		### Shuffling
		full_train = full_train.sample(frac=1) # Returns a random sample of items from an axis of object
		full_val = full_val.sample(frac=1)     # with fraction of axis items to return
		full_test = full_test.sample(frac=1)

		### NN training
			# When dropping the weights, only input vars remain, the Tag column having been interpreted as index
		xTrn = full_train.drop("Weights", axis=1)
		yTrn = Tag(full_train)
		weightTrn = full_train["Weights"]

		xVal = full_val.drop("Weights", axis=1)
		yVal = Tag(full_val)
		weightVal = full_val["Weights"]

		xTest = full_test.drop("Weights", axis=1)
		yTest = Tag(full_test)
		weightTest = full_test["Weights"]

		for q in range(3, 5):
			learning_rate = 5.e-5
			decay_rate = 10**(-1.*q)
			NodeLayer = "100 100" # Means 2 layers of 100 nodes each
			architecture=NodeLayer.split()
			ini = "he_normal" # Function for initialising the random weights of the layer
			n_epochs = 1000
			batch_size = 5000
			dropout_rate = 0.

			verbose=0

			activ = "tanh"
			loss_function = "categorical_crossentropy"

			name="levels_00"#+str(dlta)
			lrdr="LR4e-1_DR5e-7"

			dirname ="2016_1l_Tanh_CatCrossEnt_100-100_L5e-5_D1e-"+str(q) 

			fom_data = pd.DataFrame(columns=["Scaling Function","NN Configuration","Activation Function", "Loss Function", "Learning Rate", "Decay Rate", "Max Val FOM", "at x_cut", "Test FOM", "Max Test FOM"])

			org_path = "17var_models_5C/"+dirname+"/"
			if not os.path.exists(org_path):
				os.mkdir(org_path)

			for k in range(N_runs):
				string = "17var_models_5C/{}/run"+str(k)+"/"
				filepath = string.format(dirname)
				if not os.path.exists(filepath):
					os.mkdir(filepath)
				# MULTICLASS NEURAL NETWORK TRAINING ###################
				
				# Model's compilation arguments, training parameters and optimizer #categorical_crossentropy
				compileArgs = {'loss':loss_function, 'optimizer': 'adam', 'metrics': ["accuracy"]}
				trainParams = {'epochs': n_epochs, 'batch_size': batch_size, 'verbose': verbose}
				print(Adam)
				# Define optimizer which has the learning- & decay-rates
				myOpt = Adam(lr=learning_rate, decay=decay_rate)
				compileArgs['optimizer'] = myOpt
				
				# Printing info and starting time
				print("Dir "+filepath+" created.")
				print("Starting the training")
				start = time.time()
				
				### Build the NN model
				model = Sequential()
				# 1st hidden layer: it has as many nodes as provided by architecture[0]
				# input_dim: as many as input variables
				model.add(Dense(int(architecture[0]), input_dim=17, activation=activ, kernel_initializer=ini))
				i=1
				while i < len(architecture) :
					model.add(Dense(int(architecture[i]), activation=activ, kernel_initializer=ini))
					i=i+1
				model.add(Dense(5, activation='softmax')) # Output layer: 5 nodes, with softmax b/c multi-class NN
				#        model.add(kernel_regularizer=l2(1e-5))
				#        model.add(Dropout(dropout_rate))
				model.compile(**compileArgs)
				model.summary()
				########
				# Define criterium for best epoch saving
				# Returns the weights as they are at the minimum, b/c of argument of mode
				checkpoint = callbacks.ModelCheckpoint(filepath=filepath+"best_weights.h5", verbose=1, save_weights_only=True, monitor="val_loss", mode="min", save_best_only=True)
				
				### Train model ###
				history = model.fit(xTrn, yTrn, validation_data=(xVal,yVal,weightVal), sample_weight=weightTrn,shuffle=True,  callbacks=[checkpoint], **trainParams)
				# Get weights for best epoch	
				model.load_weights(filepath + "best_weights.h5")
				### Time of the training
				print("Training took: ", (time.time()-start)//60, " minutes")
				
				########################################################


				########################################################

				# SAVING OUTPUT ########################################
				
				# LOSS #################################################
				loss = history.history['loss']
				val_loss = history.history['val_loss']

				plt.figure(figsize=(10, 6))
				plt.plot(loss, label="Training loss")
				plt.plot(val_loss, label="Validation loss")
				plt.ylabel("LOSS")
				plt.xlabel("EPOCHS")
				plt.legend()

				plt.savefig(filepath+"TrnVal_loss.png")
				########################################################

				# ACCURACY #############################################
				acc = history.history["accuracy"]
				val_acc = history.history["val_accuracy"]

				plt.figure(figsize=(10, 6))
				plt.plot(acc, label="Training accuracy")
				plt.plot(val_acc, label="Validation accuracy")
				plt.ylabel("ACCURACY")
				plt.xlabel("EPOCH")
				plt.yscale("log")
				plt.legend()

				plt.savefig(filepath+"TrnVal_accuracy.png")
				########################################################
				
				### Saving accuracy and loss values in a pickle file for later plotting
				pickle.dump(acc, open(filepath+"acc.pickle", "wb"))
				pickle.dump(loss, open(filepath+"loss.pickle", "wb"))
				pickle.dump(val_acc, open(filepath+"val_acc.pickle", "wb"))
				pickle.dump(val_loss, open(filepath+"val_loss.pickle", "wb"))

				### Test
				predicted_test = pd.DataFrame(model.predict(xTest_p)).rename(columns={0:"Signal",1:"Wjets",2:"TTbar",3:"Z2nu",4:"Other"})
				true_test = yTest_p

				predicted_test["Predicted"] = predicted_test.idxmax(axis=1)
				predicted_test.index = yTest_p.index
				data_test = pd.concat([predicted_test, weightTest_p],axis=1)

				#data_test.to_csv(filepath+"Test_Data.csv")

				### Train
				predicted_train = pd.DataFrame(model.predict(xTrn_p)).rename(columns={0:"Signal",1:"Wjets",2:"TTbar",3:"Z2nu",4:"Other"})
				true_train = yTrn_p

				predicted_train["Predicted"] = predicted_train.idxmax(axis=1)
				predicted_train.index = yTrn_p.index
				data_train = pd.concat([predicted_train, weightTrn_p],axis=1)
				### Train (550,520)
				predicted_train_S = pd.DataFrame(model.predict(xTrn_S)).rename(columns={0:"Signal",1:"Wjets",2:"TTbar",3:"Z2nu",4:"Other"})
				true_train_S = yTrn_S

				predicted_train_S["Predicted"] = predicted_train_S.idxmax(axis=1)
				predicted_train_S.index = yTrn_S.index
				data_train_S = pd.concat([predicted_train_S, weightTrn_S],axis=1)

				#data_train.to_csv(filepath+"Train_Data.csv")

				### Validation
				predicted_val = pd.DataFrame(model.predict(xVal_p)).rename(columns={0:"Signal",1:"Wjets",2:"TTbar",3:"Z2nu",4:"Other"})
				true_val = yVal_p

				predicted_val["Predicted"] = predicted_val.idxmax(axis=1)
				predicted_val.index = yVal_p.index
				data_val = pd.concat([predicted_val, weightVal_p],axis=1)

				#data_val.to_csv(filepath+"Validation_Data.csv")

				### Validation (550,520)
				predicted_val_S = pd.DataFrame(model.predict(xVal_S)).rename(columns={0:"Signal",1:"Wjets",2:"TTbar",3:"Z2nu",4:"Other"})
				true_val_S = yVal_S

				predicted_val_S["Predicted"] = predicted_val_S.idxmax(axis=1)
				predicted_val_S.index = yVal_S.index
				data_val_S = pd.concat([predicted_val_S, weightVal_S],axis=1)

				#data_val_S.to_csv(filepath+"Validation_550_520_Data.csv")
				
				#### FOM ###################################################################################
				
				### Split factors ###
				# For OPTION 5
				svs = 4. # Compensate for use of 25% of signal in validation
				svbb = 4. # Compensate for use of 25% of big background in validation
				svsb = 4. # Compensate for use of 25% of small background in validation
				sts = 2. # Compensate for use of 50% of signal in test
				stbb = 2. # Compensate for use of 50% of big background in test
				stsb = 2. # Compensate for use of 50% of small background in test
				strs = 4. # Compensate for use of 25% of signal in train
				strbb = 4. # Compensate for use of 25% of big background in train
				strsb = 4. # Compensate for use of 25% of small background in train
				#####################

				# Integrated luminosity: the one of 2016
				luminosity = 35866

				# Relative systematics for FOM
				f = 0.2
					
				# DataFrame -> index=(process category), Signal (output), Wjets (output), TTbar (output), Z2nu (output), Predicted, Weights
				data_val = data_val_S
				data_train = data_train_S
				# Rescale weights ###############################

				data_val.loc["Signal", "Weights"] = svs*luminosity*data_val.loc["Signal", "Weights"]
				data_val.loc["Wjets", "Weights"] = svbb*luminosity*data_val.loc["Wjets", "Weights"]
				data_val.loc["TTbar", "Weights"] = svbb*luminosity*data_val.loc["TTbar", "Weights"]
				data_val.loc["Z2nu", "Weights"] = svbb*luminosity*data_val.loc["Z2nu", "Weights"]
				data_val.loc["Other", "Weights"] = stbb*luminosity*data_val.loc["Other", "Weights"]

				data_test.loc["Signal", "Weights"] = sts*luminosity*data_test.loc["Signal", "Weights"]
				data_test.loc["Wjets", "Weights"] = stbb*luminosity*data_test.loc["Wjets", "Weights"]
				data_test.loc["TTbar", "Weights"] = stbb*luminosity*data_test.loc["TTbar", "Weights"]
				data_test.loc["Z2nu", "Weights"] = stbb*luminosity*data_test.loc["Z2nu", "Weights"]
				data_test.loc["Other", "Weights"] = stbb*luminosity*data_test.loc["Other", "Weights"]
				
				data_train.loc["Signal", "Weights"] = strs*luminosity*data_train.loc["Signal", "Weights"]
				data_train.loc["Wjets", "Weights"] = strbb*luminosity*data_train.loc["Wjets", "Weights"]
				data_train.loc["TTbar", "Weights"] = strbb*luminosity*data_train.loc["TTbar", "Weights"]
				data_train.loc["Z2nu", "Weights"] = strbb*luminosity*data_train.loc["Z2nu", "Weights"]
				data_train.loc["Other", "Weights"] = strbb*luminosity*data_train.loc["Other", "Weights"]

				
				# FIGURE OF MERIT CALCULATION

				def FOM(S,B):
					fom1 = (S+B)*np.log(((S+B)*(B+f*f*B*B))/(B*B+(S+B)*f*f*B*B))
					fom2 = (1/(f*f))*np.log(1+(f*f*B*B*S)/(B*(B+f*f*B*B)))
					if fom1>fom2:
						fom = np.sqrt(2*(fom1-fom2))
					else: fom = 0
					return fom


				# process -> which histogram is summed over  |  predicted_events -> on which plot |||||| node -> of NN output
				def SUM_OF_EVENTS(data, process, predicted_events, node, cut):
					lower = cut[0]
					upper = cut[1]
					new_data = data.loc[data["Predicted"]==predicted_events]
					if process=="Signal":
						new_data = new_data.loc["Signal"]
					else: new_data = new_data.drop(["Signal"])
					new_data = new_data.loc[(new_data[node]>lower)&(new_data[node]<upper)]
					SUM = new_data["Weights"].sum()
					return SUM
					
				# Total number of true events
				S_tot_test = data_test.loc["Signal", "Weights"].sum()
				B_tot_test = data_test.drop(index="Signal")["Weights"].sum()
				
				S_tot_val = data_val.loc["Signal", "Weights"].sum()
				B_tot_val = data_val.drop(index="Signal")["Weights"].sum()
				
				S_tot_train = data_train.loc["Signal", "Weights"].sum()
				B_tot_train = data_train.drop(index="Signal")["Weights"].sum()

				# FIRST VERSION ####################################################################################################

				# Steps in cuts -> epsilon
				N = 200
				epsilon0_lower = 0.7
				epsilon0_upper = 1.
				epsilon = (epsilon0_upper-epsilon0_lower)/N

				fom = np.zeros(N)
				eff_sig = np.zeros(N)
				eff_bkg = np.zeros(N)
				upper_cut = np.zeros(N) + epsilon0_upper
				lower_cut = np.zeros(N) + epsilon0_lower

				fom_test = np.zeros(N)
				eff_sig_test = np.zeros(N)
				eff_bkg_test = np.zeros(N)
				
				fom_train = np.zeros(N)
				eff_sig_train = np.zeros(N)
				eff_bkg_train = np.zeros(N)

				# Fill the values
				for i in range(N):
					upper_cut[i] += -i*epsilon
					lower_cut[i] += i*epsilon


				for i in range(N):
					S = SUM_OF_EVENTS(data_val, "Signal", "Signal", "Signal", [lower_cut[i],epsilon0_upper])
					B = SUM_OF_EVENTS(data_val, "Background", "Signal", "Signal", [lower_cut[i],epsilon0_upper])
					fom[i] = FOM(S,B)
					eff_sig[i] = S/S_tot_val
					eff_bkg[i] = B/B_tot_val
					
					S = SUM_OF_EVENTS(data_test, "Signal", "Signal", "Signal", [lower_cut[i],epsilon0_upper])
					B = SUM_OF_EVENTS(data_test, "Background", "Signal", "Signal", [lower_cut[i],epsilon0_upper])
					fom_test[i] = FOM(S,B)
					eff_sig_test[i] = S/S_tot_test
					eff_bkg_test[i] = B/B_tot_test
					
					S = SUM_OF_EVENTS(data_train, "Signal", "Signal", "Signal", [lower_cut[i],epsilon0_upper])
					B = SUM_OF_EVENTS(data_train, "Background", "Signal", "Signal", [lower_cut[i],epsilon0_upper])
					fom_train[i] = FOM(S,B)
					eff_sig_train[i] = S/S_tot_train
					eff_bkg_train[i] = B/B_tot_train
					
				fom_max_val = round(np.max(fom), 8)
				imax = np.argmax(fom)
				x_cut_val = round(epsilon0_lower + imax*epsilon,4)
				fom_max_test = round(np.max(fom_test), 8)
				jmax = np.argmax(fom_test)
				x_cut_test = round(epsilon0_lower + jmax*epsilon,4)
				fom_max_train = round(np.max(fom_train), 8)
				kmax = np.argmax(fom_train)
				x_cut_train = round(epsilon0_lower + kmax*epsilon,4)

				# Maximum FOM in test
				S = SUM_OF_EVENTS(data_test, "Signal", "Signal", "Signal", [lower_cut[imax],epsilon0_upper])
				B = SUM_OF_EVENTS(data_test, ["Wjets", "TTbar", "Z2nu", "Other"], "Signal", "Signal", [lower_cut[imax],epsilon0_upper])
				fom_max = round(FOM(S,B), 8)
				
				fom_data.loc[str(k)] = [name, "100-100", activ, loss_function, learning_rate, decay_rate, fom_max_val, x_cut_val, fom_max, fom_max_test]
				
				def SUM_OF_EVENTS(data, process, predicted_events, node, cut):
					lower = cut[0]
					upper = cut[1]
					new_data = data.loc[data["Predicted"]==predicted_events]
					if process=="Signal":
						new_data = new_data.loc["Signal"]
					else: 
						new_data = new_data.loc[process]
					new_data = new_data.loc[(new_data[node]>lower)&(new_data[node]<upper)]
					SUM = new_data["Weights"].sum()
					return SUM

				B_W = SUM_OF_EVENTS(data_test, "Wjets", "Signal", "Signal", [lower_cut[imax],epsilon0_upper])
				B_T = SUM_OF_EVENTS(data_test, "TTbar", "Signal", "Signal", [lower_cut[imax],epsilon0_upper])
				B_Z = SUM_OF_EVENTS(data_test, "Z2nu", "Signal", "Signal", [lower_cut[imax],epsilon0_upper])
				B_O = SUM_OF_EVENTS(data_test, "Other", "Signal", "Signal", [lower_cut[imax],epsilon0_upper])

				plt.figure(figsize=(20, 15))

				plt.subplot(2,1,1)
				plt.plot(lower_cut, fom, color="blue", linewidth=5, label="Validation Data")

				plt.plot(lower_cut, fom_test, ls="--", linewidth=5, label="Test Data")
				plt.plot(lower_cut, fom_train, ls="--", color="yellow", linewidth=5, label="Train Data")
				plt.ylabel("Figure Of Merit", size=30)
				plt.title("RUN "+ str(k), size=20)
				plt.xticks(fontsize=15)
				maximum = max([fom_max_val, fom_max_train, fom_max_test])
				ticks = np.arange(0,round(maximum, 1)+0.1,0.1)
				plt.yticks(ticks, fontsize=15)
				number = int(ticks[-1]/0.5)
				for num in range(number):
					plt.axhline(y=num*0.5+0.5, color="black", linewidth=1)
				plt.grid(True, color='black', linestyle='--', linewidth=0.5)
				plt.axvline(x=lower_cut[imax], label="Test FOM: " + str(round(fom_max,3)) +"\nMax Validation FOM: " + str(round(fom_max_val,4)), color="gray", linewidth=6, linestyle="--")
				plt.axvline(x=lower_cut[jmax], label="Max Test FOM: " + str(round(fom_max_test,4)), color="red", linewidth=3, linestyle="--")
				plt.axvline(x=lower_cut[kmax], label="Max Train FOM: " + str(round(fom_max_train,4)), color="green", linewidth=3, linestyle="--")
				plt.legend(fontsize=25, loc="lower left")

				plt.subplot(2,1,2)
				plt.plot(lower_cut, eff_sig, color="red", linewidth=5, label="Signal Efficiency")
				plt.plot(lower_cut, eff_bkg, color="purple", linewidth=5, label="Background Efficiency")
				plt.yscale("log")
				plt.ylabel("Efficiency", size=30)
				plt.axvline(x=lower_cut[imax], label="S: "+str(round(S,1)) + "  W: "+str(round(B_W,1))+"  T: "+str(round(B_T,1))+"  Z: "+str(round(B_Z,1))+"  O: "+str(round(B_O,1)), color="gray", linewidth=6, linestyle="--")

				S = SUM_OF_EVENTS(data_test, "Signal", "Signal", "Signal", [lower_cut[jmax],epsilon0_upper])
				B_W = SUM_OF_EVENTS(data_test, "Wjets", "Signal", "Signal", [lower_cut[jmax],epsilon0_upper])
				B_T = SUM_OF_EVENTS(data_test, "TTbar", "Signal", "Signal", [lower_cut[jmax],epsilon0_upper])
				B_Z = SUM_OF_EVENTS(data_test, "Z2nu", "Signal", "Signal", [lower_cut[jmax],epsilon0_upper])
				B_O = SUM_OF_EVENTS(data_test, "Other", "Signal", "Signal", [lower_cut[jmax],epsilon0_upper])

				plt.axvline(x=lower_cut[jmax], label="S: "+str(round(S,1)) + "  W: "+str(round(B_W,1))+"  T: "+str(round(B_T,1))+"  Z: "+str(round(B_Z,1))+"  O: "+str(round(B_O,1)), color="red", linewidth=3, linestyle="--")
				
				S = SUM_OF_EVENTS(data_test, "Signal", "Signal", "Signal", [lower_cut[kmax],epsilon0_upper])
				B_W = SUM_OF_EVENTS(data_test, "Wjets", "Signal", "Signal", [lower_cut[kmax],epsilon0_upper])
				B_T = SUM_OF_EVENTS(data_test, "TTbar", "Signal", "Signal", [lower_cut[kmax],epsilon0_upper])
				B_Z = SUM_OF_EVENTS(data_test, "Z2nu", "Signal", "Signal", [lower_cut[kmax],epsilon0_upper])
				B_O = SUM_OF_EVENTS(data_test, "Other", "Signal", "Signal", [lower_cut[kmax],epsilon0_upper])
				
				plt.axvline(x=lower_cut[kmax], label="S: "+str(round(S,1)) + "  W: "+str(round(B_W,1))+"  T: "+str(round(B_T,1))+"  Z: "+str(round(B_Z,1))+"  O: "+str(round(B_O,1)), color="green", linewidth=3, linestyle="--")
				
				plt.xticks(fontsize=15)
				plt.yticks(fontsize=15)
				plt.legend(fontsize=25, loc="lower left")
				plt.xlabel("Lower Cut On Signal Node Of NN Output", size=30)
				
				runs_path = org_path+"FOM_figures_runs/"
				if not os.path.exists(runs_path):
					os.mkdir(runs_path)

				plt.savefig(runs_path+"FOM_run"+str(k)+".png")
				
			fom_data.to_csv("17var_fom.csv", mode="a")
	```
