from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import cross_validation
import pandas as pd
import awkward
# import servicex
# from servicex import ServiceXDataset
# from func_adl_xAOD import ServiceXDatasetSource
# import uproot_methods
from numpy import genfromtxt

# Eventually, what I want here is to pull some dataset and train the neural network to separate noise from signal for two different variables.
# But also, those two different variables need to remain uncorrelated.
# for the time being, we won't use ServiceX though, so this is commented out

# data = retrieve_data("mc15_13TeV:mc15_13TeV.361106.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zee.merge.DAOD_STDM3.e3601_s2576_s2132_r6630_r6264_p2363_tid05630052_00")
# separation_variables = retrieve_data("this is just a placeholder") # note to self, this should eventually have two sets of data

# Here, we will be generating data using a simple function defined in this code
# We generate 5 sets of data: a signal dataset with a moderate correlation, a noise dataset with no correlation
# and 3 discriminating variables for the neural network to take as inputs. No, at the moment I have no idea how this works.

noise_data_x = generate_distribution(events = 5000, peak = 0.0, width = 10.0, height = 1.0, correlation = 0.0, width_factor = 1.0)
noise_data_y = generate_distribution(events = 5000, peak = 0.0, width = 10.0, height = 1.2, correlation = 0.0, width_factor = 1.0)
noise_data_z = generate_distribution(events = 5000, peak = 0.0, width = 10.0, height = 0.8, correlation = 0.0, width_factor = 1.0)

signal_data_x = generate_distribution(events = 500, peak = 15.0, width = 10.0, height = 3.0, correlation = 0.4, width_factor = 1.0)
signal_data_y = generate_distribution(events = 500, peak = 15.0, width = 10.0, height = 3.0, correlation = 0.4, width_factor = 1.0)
signal_data_z = generate_distribution(events = 500, peak = 15.0, width = 10.0, height = 3.0, correlation = 0.4, width_factor = 1.0)

data_train, data_test, comparison_train, comparison_test = train_test_split(data, separation_variables)

scaler = StandardScaler()

scaler.fit(data_train)

data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

neural_network = MLPClassifier(hidden_layer_sizes=(50,50), activation='tanh', max_iter=500)

neural_network.fit(data_train, comparison_train)

predicted_x = neural_network.predict(data_test)

# this will eventually be phased out, just need it to see how well the neural network is working
print(confusion_matrix(comparison_test, predicted_x))
print(classification_report(comparison_test, predicted_x))

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

def generate_distribution(events, peak, width, height, correlation, width_factor):
    x_interval = np.array([peak - (width/2), peak + (width/2)])
    y_interval = np.array([0, height])
    vector = np.vstack((x_interval, y_interval)).T

    means = lambda x, y : [x.mean(), y.mean()]
    stdvs = lambda x, y : [x.std() / width_factor, y.std() / width_factor]
    covs = [[stdvs(x_interval, y_interval)[0]**2, stdvs(x_interval, y_interval)[0] * stdvs(x_interval, y_interval)[1] * correlation],
            [stdvs(x_interval, y_interval)[1] * stdvs(x_interval, y_interval)[1] * correlation, stdvs(x_interval, y_interval)[1]**2]]

    generated_data = np.random.multivariate_normal(means(x_interval, y_interval), covs, events).T

    return generated_data


def loss_function():
