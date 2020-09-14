from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import awkward
# import servicex
# from servicex import ServiceXDataset
# from func_adl_xAOD import ServiceXDatasetSource
# import uproot_methods
from numpy import genfromtxt

# Here we initialize some variables for ease of control

noise_events = 5000
signal_events = 1500

# Eventually, what I want here is to pull some dataset and train the neural network to separate noise from signal for two different variables.
# But also, those two different variables need to remain uncorrelated.
# for the time being, we won't use ServiceX though, so this is commented out

# data = retrieve_data("mc15_13TeV:mc15_13TeV.361106.PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zee.merge.DAOD_STDM3.e3601_s2576_s2132_r6630_r6264_p2363_tid05630052_00")
# separation_variables = retrieve_data("this is just a placeholder") # note to self, this should eventually have two sets of data

# Here, we will be generating data using a simple function defined in this code
# This generates six 2-dimensional histograms. At the moment, we are only interested in one of those dimensions, but it may come to pass that we need both.

noise_data_x = generate_distribution(events = noise_events, peak = 0.0, width = 10.0, height = 1.0, correlation = 0.0, width_factor = 1.0)
noise_data_y = generate_distribution(events = noise_events, peak = 0.0, width = 10.0, height = 1.2, correlation = 0.0, width_factor = 1.0)
noise_data_z = generate_distribution(events = noise_events, peak = 0.0, width = 10.0, height = 0.8, correlation = 0.0, width_factor = 1.0)

signal_data_x = generate_distribution(events = signal_events, peak = 15.0, width = 10.0, height = 3.0, correlation = 0.4, width_factor = 1.0)
signal_data_y = generate_distribution(events = signal_events, peak = -15.0, width = 10.0, height = 3.0, correlation = 0.4, width_factor = 1.0)
signal_data_z = generate_distribution(events = signal_events, peak = 15.0, width = 10.0, height = 3.0, correlation = 0.4, width_factor = 1.0)

# Here, we separate the histograms and prepare the data for use for the neural network.

noise_list = [0] * noise_events
signal_list = [1] * signal_events

var_x = noise_data_x + signal_data_x
var_y = noise_data_y + signal_data_y
var_z = noise_data_z + signal_data_z
signal_or_noise_axis_1 = noise_list + signal_list
signal_or_noise_axis_2 = noise_list + signal_list

data_dict = {'var_1':var_x, 'var_2':var_y, 'var_3':var_z}
output_dict = {'sn1':signal_or_noise_axis_1, 'sn2':signal_or_noise_axis_2}

# Construct a pandas dataframe with all our data and prepare it for analysis:

input_data = pd.DataFrame(data_dict)
output_data = pd.DataFrame(output_dict)

input_train, input_test, output_train, output_test = train_test_split(input_data, output_data)

svc = SVC(gamma="scale")
neural_network = MultiOutputClassifier(estimator=svc)

# Train the actual neural network

neural_network.fit(input_train, output_train)

# See how the trained neural network performs
predicted_x = neural_network.predict(input_test)

# this will eventually be phased out, just need it to see how well the neural network is working
print(confusion_matrix(output_test, predicted_x))
print(classification_report(output_test, predicted_x))

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
