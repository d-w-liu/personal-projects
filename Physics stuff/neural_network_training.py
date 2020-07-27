from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import cross_validation
import pandas as pd
import awkward
import servicex
from servicex import ServiceXDataset
from func_adl_xAOD import ServiceXDatasetSource
import uproot_methods
from numpy import genfromtxt

# Eventually, what I want here is to pull some dataset and train the neural network to separate noise from signal for two different variables.
# But also, those two different variables need to remain uncorrelated.
data = retrieve_data("mc15_13TeV:mc15_13TeV.361106.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zee.merge.DAOD_STDM3.e3601_s2576_s2132_r6630_r6264_p2363_tid05630052_00")
separation_variables = retrieve_data("this is just a placeholder") # note to self, this should eventually have two sets of data

data_train, data_test, comparison_train, comparison_test = train_test_split(data, separation_variables)

scaler = StandardScaler()

scaler.fit(data_train)

data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

neural_network = MLPClassifier(hidden_layer_sizes=(50,50), activation='tanh', max_iter=500)

neural_network.fit(data_train, comparison_train)

predicted_x = neural_network.predict(data_test)

# this will eventually be phased out, just need it to see how well the neural network is working
print(confusion_matrix(comparison_x_test, predicted_x))
print(confusion_matrix(comparison_y_test, predicted_y))
print(classification_report(comparison_x_test, predicted_x))
print(classification_report(comparison_y_test, predicted_y))

def retrieve_data(dataset):
    dataset = ServiceXDataset(dataset)
    query = ServiceXDatasetSource(dataset)	\
        .Select('lambda e: (e.Electrons("Electrons"))') \
        .Select('lambda e: (e.Select(lambda e: e.pt()), \
                            e.Select(lambda e: e.eta()), \
			    			e.Select(lambda e: e.phi()), \
				    		e.Select(lambda e: e.e()))') \
        .AsAwkwardArray(('ElePt', 'EleEta', 'ElePhi', 'EleE')) \
        .value()

    return query

def loss_function():
