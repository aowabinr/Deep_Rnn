#importing keras modules to
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Input, Merge
from keras.layers import merge
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping

#from keras.utils.visualize_util import plot
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from hyperopt import Trials, STATUS_OK, tpe
#from hyperas import optim
#from hyperas.distributions import choice, uniform, conditional
from hyperopt import fmin, tpe, hp
from tempfile import TemporaryFile


#importing graphics and numpy
import numpy
import pydot
#print pydot.find_graphviz()

#importing Custom Libraries
#manual functions
import ArrayFunctions
import MathFunctions
import DataFunctions
import NNFunctions
import PlotFunctions
import InterpolateFunctions
import NNFun_PSB
import build_lstm_v1


#define hyperopt search space
import hyperopt.pyll.stochastic

from sklearn.metrics import r2_score
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.feature_selection import chi2

#scipy library
from scipy.stats import pearsonr

seed = 7
numpy.random.seed(seed)

from datetime import  datetime

#Fetch data for training
date_start = '5/19/15 12:00 AM'
date_end = '5/13/16 11:59 PM'
std_inv = 60 #in minutes
std_inv2 = 10 #in minutes

#Read data
#The energy data is seggregated by end uses.
#The idea is to aggregate 5 min data to 10-min and 1-hour resolutions for a given end-use of our choice
#This will allow us to make predictions at 5-minute and 10-minute resolutions
data = DataFunctions.read_PSB_csv(date_start, date_end, 5, 20) #read data at 5 min resolutions
conv_critical, crac_critical, crac_normal, conv_normal, HVAC_critical, Elev, HVAC_normal, elec_total = DataFunctions.PSB_elec_split(data) #split by end uses
H_t2 = DataFunctions.prepare_energy_PSB(HVAC_critical, std_inv2) #Energy Consumption Data for multi timescale analysis

#performing linear interpolation on H_t2 first.
#To make the best use of available data, we are performing linear interpolation on missing data first.
small_list2, large_list2 = InterpolateFunctions.interpolate_main_v2(H_t2, 30) #Allowing for 30 consecutive timesteps
H_t2[0, :] = 0
H_t2 = InterpolateFunctions.interp_linear(H_t2, small_list2)

#H_t = DataFunctions.prepare_energy_PSB(HVAC_critical, std_inv)
H_t2 = H_t2[:, None]
H_t= DataFunctions.fix_energy_intervals(H_t2, std_inv2, std_inv)
H_t = DataFunctions.fix_high_points(H_t)

#H_t -> 1-hour resolution
#H_t2 -> 10-min resolution

#Weather Data for training
#weather train -> 1-hour resolution
#weather_train2 -> 10-min resolution
weather_train = DataFunctions.prepare_weather_WBB(date_start, date_end, std_inv)
weather_train2 = DataFunctions.prepare_weather_WBB(date_start, date_end, std_inv2)

#feature vectors for date-related variables
#X_sch_t -> 1-hour resolution
#X_sch_t2 -> 10-min resolution
X_sch_t = DataFunctions.compile_features(H_t, date_start, std_inv)
X_sch_t2 = DataFunctions.get_feature_low_res(X_sch_t, std_inv, std_inv2)

#Concatenating All features
train_data = numpy.concatenate((weather_train[:, 0:2], X_sch_t), axis=1)
train_data2 = numpy.concatenate((weather_train2[:, 0:2], X_sch_t2), axis=1)

#Normalizing the targets H_t
H_rms = MathFunctions.rms_flat(H_t)
#normalizing data
H_min, H_max = DataFunctions.get_normalize_params(H_t)
H_t = H_t/H_max

#normalizing H_t2
H_min2, H_max2 = DataFunctions.get_normalize_params(H_t2)
H_t2 = H_t2/H_max2

#Computing Entropy
S, unique, pk = DataFunctions.calculate_entropy(H_t)
#print "The entropy value is: ", S

# Block to interpolate
cons_points = 5
s0 = 245
start_day = 250
end_day  = 260
s1 = 265

choice = 1

#This block is to fill out the large chunks.
#I've saved the missing chunks and stored into a file
if choice == 1:
    H_t = numpy.load('H1_fill_HVAC1.npy')
elif choice == 2:
    X_t, Y_t, X_e, Y_e, train_list, test_list = InterpolateFunctions.train_test_split(train_data, H_t)
    best = InterpolateFunctions.imputate_optimize(X_t, Y_t)
    Y_p = NNFun_PSB.PSB_model_DL(X_t, Y_t, X_e, best)
    H_t = InterpolateFunctions.organize_pred(H_t, Y_p, test_list)
    #numpy.save('Ht_file_HVAC1.npy', H_t)
else:
    H1 = H_t.copy()
    small_list, large_list = InterpolateFunctions.interpolate_main(train_data, H_t, start_day, end_day, cons_points)
    H_t = InterpolateFunctions.interp_linear(H_t, small_list) #H_t is beign changed as numpy arrays are mutable
    H1 = InterpolateFunctions.interp_linear(H1, small_list)

    PlotFunctions.Plot_interp_params()
    H_t = H_t[:, None] #changing numpy array shape to fit the function
    train_interp, dummy1, dummy2 = DataFunctions.normalize_103(train_data, train_data, train_data)
    Y_t, Y_NN = InterpolateFunctions.interp_LSTM(train_interp, H_t, large_list)

    PlotFunctions.Plot_interpolate(H1[s0*24:start_day*24], Y_t[start_day*24:end_day*24], Y_NN[start_day*24:end_day*24], H1[start_day*24:end_day*24], H1[end_day*24:s1*24])
    e_interp = InterpolateFunctions.interpolate_calculate_rms(H1[start_day*24:end_day*24], Y_t[start_day*24:end_day*24])
    e_NN = InterpolateFunctions.interpolate_calculate_rms(H1[start_day*24:end_day*24], Y_NN[start_day*24:end_day*24])

    print e_interp
    print e_NN
    H_t = Y_t.copy()
    #H_t = Y_NN.copy()
    #numpy.save('H1_fill_HVAC1.npy', H_t)


##block to fill in points for H_t2
choice = 0

#filling out the large chunks in H_t2
#small_list-> List that can be filled out by linear interpolation, i.e. missing segment < 5 hours
#large_list -> need to implement implementation scheme for this
small_list2, large_list2 = InterpolateFunctions.interpolate_main_v2(H_t2, 30) #Allowing for 30 consecutive timesteps
small_list, large_list = InterpolateFunctions.interpolate_main_v2(H_t, 5)

####This block is to fill up using low-res predictions to high resolution missing data
H_new = DataFunctions.datafill_low_to_high(H_t.copy()*H_max, H_t2.copy(), large_list2, (std_inv/std_inv2), H_max2)

#Aggregating data on a daily basis
H_t = H_t[:, None]
conv_hour_to_day = 24
H_mean_t, H_sum_t, H_min_t, H_max_t = DataFunctions.aggregate_data(H_t, conv_hour_to_day)
w_mean_t, w_sum_t, w_min_t, w_max_t = DataFunctions.aggregate_data(weather_train, conv_hour_to_day)

#gettomg features for a single day
#PlotFunctions.Plot_single(H_mean_t)
X_day_t = DataFunctions.compile_features(H_sum_t, date_start, 24*60)

##########################################################################################
#Repeating the same procedure for validation data
#Read data
#Read data
date_start = '5/12/16 12:00 AM'
date_end = '5/18/16 11:59 PM'

data = DataFunctions.read_PSB_csv(date_start, date_end, 5, 20) #read data at 5 min resolutions
conv_critical, crac_critical, crac_normal, conv_normal, HVAC_critical, Elev, HVAC_normal, elec_total = DataFunctions.PSB_elec_split(data) #split by end uses
H_val2 = DataFunctions.prepare_energy_PSB(HVAC_critical, std_inv2) #Energy Consumption Data for multi timescale analysis

#performing linear interpolation on H_t2 firs
small_list2, large_list2 = InterpolateFunctions.interpolate_main_v2(H_val2, 30) #Allowing for 30 consecutive timesteps
H_val2[0, :] = 0
H_val2 = InterpolateFunctions.interp_linear(H_val2, small_list2)

#H_t = DataFunctions.prepare_energy_PSB(HVAC_critical, std_inv)
H_val2 = H_val2[:, None]
H_val= DataFunctions.fix_energy_intervals(H_val2, std_inv2, std_inv)
H_val = DataFunctions.fix_high_points(H_val)

#Weather Data for training
weather_val = DataFunctions.prepare_weather_WBB(date_start, date_end, std_inv)
weather_val2 = DataFunctions.prepare_weather_WBB(date_start, date_end, std_inv2)

#feature vectors
X_sch_val = DataFunctions.compile_features(H_val, date_start, std_inv)
X_sch_val2 = DataFunctions.get_feature_low_res(X_sch_val, std_inv, std_inv2)

val_data = numpy.concatenate((weather_val[:, 0:2], X_sch_val), axis=1)
val_data2 = numpy.concatenate((weather_val2[:, 0:2], X_sch_val2), axis=1)

#normalizing data
H_val = H_val/H_max
#X_sch_t = DataFunctions.normalize_vector(X_sch_t, X_min, X_max)

#normalizing H_t2
H_val2 = H_val2/H_max2

choice = 0

if choice == 1:
    H_val = numpy.load('Hv_file_total.npy')
else:
    X_t, Y_t, X_e, Y_e, train_list, test_list = InterpolateFunctions.train_test_split(val_data, H_val)
    best = InterpolateFunctions.imputate_optimize(X_t, Y_t)
    Y_p = NNFun_PSB.PSB_model_DL(X_t, Y_t, X_e, best)
    H_val = InterpolateFunctions.organize_pred(H_val, Y_p, test_list)
    numpy.save('Hv_file_total.npy', H_val)

#filling out the large chunks in H_t2
small_list2, large_list2 = InterpolateFunctions.interpolate_main_v2(H_t2, 30) #Allowing for 30 consecutive timesteps
small_list, large_list = InterpolateFunctions.interpolate_main_v2(H_t, 5)

####This block is to fill up using low-res predictions to high resolution missing data
H_new = DataFunctions.datafill_low_to_high(H_t.copy()*H_max, H_t2.copy(), large_list2, (std_inv/std_inv2), H_max2)

#Aggregating data on a daily basis
H_mean_v, H_sum_v, H_min_v, H_max_v = DataFunctions.aggregate_data(H_val, conv_hour_to_day)
w_mean_v, w_sum_v, w_min_v, w_max_v = DataFunctions.aggregate_data(weather_val, conv_hour_to_day)

#gettomg features for a single day
X_day_val = DataFunctions.compile_features(H_sum_v, date_start, 24*60)

#######################################################################################
#Repeating the same procedure for test data
#Read data
###Test data: 2016
date_start = '5/19/16 12:00 AM'
date_end = '8/7/16 11:59 PM'

data = DataFunctions.read_PSB_csv(date_start, date_end, 5, 20) #read data at 5 min resolutions
conv_critical, crac_critical, crac_normal, conv_normal, HVAC_critical, Elev, HVAC_normal, elec_total = DataFunctions.PSB_elec_split(data) #split by end uses
H_e2 = DataFunctions.prepare_energy_PSB(HVAC_critical, std_inv2) #Energy Consumption Data for multi timescale analysis

#performing linear interpolation on H_t2 firs
small_list2, large_list2 = InterpolateFunctions.interpolate_main_v2(H_e2, 30) #Allowing for 30 consecutive timesteps
H_e2[0, :] = 0
H_e2 = InterpolateFunctions.interp_linear(H_e2, small_list2)

#H_t = DataFunctions.prepare_energy_PSB(HVAC_critical, std_inv)
H_e2 = H_e2[:, None]
H_e= DataFunctions.fix_energy_intervals(H_e2, std_inv2, std_inv)
H_e = DataFunctions.fix_high_points(H_e)


#Weather Data for training
weather_test = DataFunctions.prepare_weather_WBB(date_start, date_end, std_inv)
weather_test2 = DataFunctions.prepare_weather_WBB(date_start, date_end, std_inv2)

#feature vectors
X_sch_e = DataFunctions.compile_features(H_e, date_start, std_inv)
X_sch_e2 = DataFunctions.get_feature_low_res(X_sch_e, std_inv, std_inv2)

test_data = numpy.concatenate((weather_test[:, 0:2], X_sch_e), axis=1)
test_data2 = numpy.concatenate((weather_test2[:, 0:2], X_sch_e2), axis=1)

#Normalizing Data:
train_data, val_data, test_data = DataFunctions.normalize_103(train_data, val_data, test_data)
train_data2, val_data2, test_data2 = DataFunctions.normalize_103(train_data2, val_data2, test_data2)


#Aggregating data on a daily basis
H_mean_e, H_sum_e, H_min_e, H_max_e = DataFunctions.aggregate_data(H_e, conv_hour_to_day)
w_mean_e, w_sum_e, w_min_e, w_max_e = DataFunctions.aggregate_data(weather_test, conv_hour_to_day)

#gettomg features for a single day
X_day_e = DataFunctions.compile_features(H_sum_e, date_start, 24*60)

#normalizing H_e
H_e = H_e/H_max

#normalizing H_t2
H_e2 = H_e2/H_max2

choice = 0

if choice == 1:
    H_e = numpy.load('He_file_total.npy')
else:
    X_t, Y_t, X_e, Y_e, train_list, test_list = InterpolateFunctions.train_test_split(test_data, H_e)
    best = InterpolateFunctions.imputate_optimize(X_t, Y_t)
    Y_p = NNFun_PSB.PSB_model_DL(X_t, Y_t, X_e, best)
    H_e = InterpolateFunctions.organize_pred(H_e, Y_p, test_list)
    numpy.save('He_file_total.npy', H_e)



#Saving variables for MLP neural network
X1 = train_data.copy()
X2 = val_data.copy()
X3 = test_data.copy()
H1 = H_t.copy()
H2 = H_val.copy()
H3 = H_e.copy()


####Saving block
numpy.save('Ht2_HVAC1.npy', H_t2)
numpy.save('Hv2_HVAC1.npy', H_val2)
numpy.save('He2_HVAC1.npy', H_e2)


#print H_mean_t.shape
#I need to reshape the array because the input to my RNN model in keras is (N, 24. |T|)
# Reshaping array into (#of days, 24-hour timesteps, #features)
train_data = numpy.reshape(train_data, (X_day_t.shape[0], 24, train_data.shape[1]))
val_data = numpy.reshape(val_data, (X_day_val.shape[0], 24, val_data.shape[1]))
test_data = numpy.reshape(test_data, (X_day_e.shape[0], 24, test_data.shape[1]))

X_sch_t = numpy.reshape(X_sch_t, (X_day_t.shape[0], 24, X_sch_t.shape[1]))
X_sch_val = numpy.reshape(X_sch_val, (X_day_val.shape[0], 24, X_sch_val.shape[1]))
X_sch_e = numpy.reshape(X_sch_e, (X_day_e.shape[0], 24, X_sch_e.shape[1]))

H_t = numpy.reshape(H_t, (H_mean_t.shape[0], 24))
H_val = numpy.reshape(H_val, (H_mean_v.shape[0], 24))
H_e = numpy.reshape(H_e, (H_mean_e.shape[0], 24))


####Reshaping into an array
#Reshaping array into (#of days, 10-min, #features)
#
train_data2 = numpy.reshape(train_data2, (conv_hour_to_day*X_day_t.shape[0], int(std_inv/std_inv2), train_data2.shape[1]))
val_data2 = numpy.reshape(val_data2, (conv_hour_to_day*X_day_val.shape[0], int(std_inv/std_inv2), val_data2.shape[1]))
test_data2 = numpy.reshape(test_data2, (conv_hour_to_day*X_day_e.shape[0], int(std_inv/std_inv2), test_data2.shape[1]))
H_t2 = numpy.reshape(H_t2, (conv_hour_to_day*H_mean_t.shape[0], int(std_inv/std_inv2)))
H_val2 = numpy.reshape(H_val2, (conv_hour_to_day*H_mean_v.shape[0], int(std_inv/std_inv2)))
H_e2 = numpy.reshape(H_e2, (conv_hour_to_day*H_mean_e.shape[0], int(std_inv/std_inv2)))




#This block is for optimizing LSTM layers
#This is the search space over which I will look for optimal hyper-paramters
space = {
        'Layer1': hp.quniform('Layer1', 10, 100, 5),
        'Layer2': hp.quniform('Layer2', 10, 100, 5),
        'Layer3': hp.quniform('Layer3', 5, 20, 1),
         'activ_l3': hp.choice('activ_l3', ['relu',  'sigmoid']),
         'activ_l4': hp.choice('activ_l4', ['linear'])
         }


space2 = {
        'Layer1': hp.quniform('Layer1', 10, 100, 5),
        'Layer2': hp.quniform('Layer2', 10, 100, 5),
        'Layer3': hp.quniform('Layer3', 5, 100, 1),
        'activ_l3': hp.choice('activ_l3', [ 'sigmoid', 'relu']),
        'activ_l4': hp.choice('activ_l4', ['linear'])

         }


idx_1 = numpy.isnan(H_t2)
idx_2 = numpy.isnan(H_val2)
idx_3 = numpy.isnan(H_e2)


H_t2[idx_1]= 0
H_val2[idx_2]= 0
H_e2[idx_3] = 0

#################
#Optimizing for space 1
#Running tests over 40 evals, 20 epochs each
#I will use lstm_model_110

def objective(params):
    optimize_model = build_lstm_v1.lstm_model_110(params, train_data.shape[2], 24)
    # for epochs in range(5):
    for ep in range(20):
        optimize_history = optimize_model.fit(train_data, H_t, batch_size=1, nb_epoch=1,
                                              validation_data=(val_data, H_val), shuffle=False)
        optimize_model.reset_states()

    loss_v = optimize_history.history['val_loss']
    print loss_v

    loss_out = loss_v[-1]

    return {'loss': loss_out, 'status': STATUS_OK}


trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=40)
#best-> optimal set of hyper-paraemters


#Building Stateful Model
lstm_hidden = hyperopt.space_eval(space, best)
print lstm_hidden
tsteps = 24
out_dim = 24

#Now I've found the best hyper-parameters I will fit my actual model
#lstm_model = build_lstm_v1.lstm_model_102(lstm_hidden, train_data.shape[2], out_dim, tsteps)
lstm_model = build_lstm_v1.lstm_model_110(lstm_hidden, train_data.shape[2], tsteps)
save_model = lstm_model

##callbacks for Early Stopping
callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

#parameters for simulation
attempt_max = 5 #number of tries -> each with a different weight initialisation
epoch_max = 200 #maximum number of epochs to prevent over-fitting
min_epoch = 20 #I will at least have 20 epochs

#Criterion for early stopping
tau = 10
e_mat = numpy.zeros((epoch_max, attempt_max))
e_temp = numpy.zeros((tau, ))

tol = 0
count = 0
val_loss_v = []
epsilon = 1 #initialzing error
loss_old = 1
loss_val = 1

#Fit model

for attempts in range(attempt_max):
    lstm_model = build_lstm_v1.lstm_model_110(lstm_hidden, train_data.shape[2], tsteps)
    print "New model Initialized"

    for ep in range(epoch_max):
        lstm_history = lstm_model.fit(train_data, H_t, batch_size=1, nb_epoch=1, validation_split=0, shuffle=False)

        loss_old = loss_val
        loss_val = lstm_history.history['loss']

        # testing alternative block
        #lstm_model.reset_states()
        y_val = lstm_model.predict(val_data, batch_size=1, verbose=0)
        e1, e2 = DataFunctions.find_error(H_t, H_val, y_val)
        print e1, e2
        val_loss_check = e2
        val_loss_v.append(val_loss_check)
        e_mat[ep, attempts] = val_loss_check

        if val_loss_v[count] < epsilon and loss_val < loss_old:
            epsilon = val_loss_v[count]
            save_model = lstm_model
            save_model.save_weights('my_model_weights.h5', overwrite=True)
            #test_model = lstm_model
            #Y_lstm = test_model.predict(test_data, batch_size=1, verbose=0)
            #e_1, e_2 = DataFunctions.find_error(H_t, H_e, Y_lstm)
            #test_model.reset_states()
            #print e_1
            #print e_2

        count = count + 1
        lstm_model.reset_states()


        #This block is for early stopping
        if ep>=min_epoch:
            e_temp = e_mat[ep - tau + 1: ep + 1, attempts]
            e_local = e_mat[ep-tau, attempts]

            #print e_temp
            #print e_local

            if numpy.all(e_temp > e_local):
                break



        #if val_loss_check < tol:
            #break


###########Extracting Data for multi-timescale Analysis
#This is being inserted for long-term forecasts
tsteps2 = 6
get_1st_layer_output = K.function([save_model.layers[0].input], [save_model.layers[3].output])
lstm_h2 = int(lstm_hidden['Layer2'])
h_input = numpy.zeros((len(train_data2), tsteps2, lstm_h2))
h_val = numpy.zeros((len(val_data2), tsteps2, lstm_h2))
h_test = numpy.zeros((len(test_data2), tsteps2, lstm_h2))


for i in range(0, len(train_data)):
    X_temp = train_data[i, :, :]
    X_temp = X_temp[None, :, :]
    h_temp = numpy.asarray(get_1st_layer_output([X_temp]))
    h_dim = h_temp.shape[3]
    h_temp2 = numpy.reshape(h_temp, (24, h_dim))

    for j in range(0, tsteps2):
        h_input[i*24:(i+1)*24, j, :] = h_temp2

###Getting validation data
for i in range(0, len(val_data)):
    X_temp = val_data[i, :, :]
    X_temp = X_temp[None, :, :]
    h_dim = h_temp.shape[3]
    h_temp2 = numpy.reshape(h_temp, (24, h_dim))

    for j in range(0, tsteps2):
        h_val[i * 24:(i + 1) * 24, j, :] = h_temp2

for i in range(0, len(test_data)):
    X_temp = test_data[i, :, :]
    X_temp = X_temp[None, :, :]
    h_temp = numpy.asarray(get_1st_layer_output([X_temp]))
    h_dim = h_temp.shape[3]
    h_temp2 = numpy.reshape(h_temp, (24, h_dim))

    for j in range(0, tsteps2):
        h_test[i * 24:(i + 1) * 24, j, :] = h_temp2


print h_input.shape
print h_val.shape
print h_test.shape
####--------------------------------------------------------------------

save_model.load_weights('my_model_weights.h5')
Y_lstm2 = save_model.predict(test_data, batch_size=1, verbose=0)
numpy.save('HVAC_critical_B.npy', Y_lstm2)

#### Error analysis
H_t = numpy.reshape(H_t, (H_t.shape[0]*24, 1))
H_e = numpy.reshape(H_e, (H_e.shape[0]*24, 1))
#Y_lstm = numpy.reshape(Y_lstm, (Y_lstm.shape[0]*24, 1))
Y_lstm2 = numpy.reshape(Y_lstm2, (Y_lstm2.shape[0]*24, 1))
t_train = numpy.arange(0, len(H_t))
t_test = numpy.arange(len(H_t), len(H_t)+len(Y_lstm2))
t_array = numpy.arange(0, len(Y_lstm2))

e_deep = (MathFunctions.rms_flat(Y_lstm2 - H_e))/(MathFunctions.rms_flat(H_e))
e_deep2 = (MathFunctions.rms_flat(Y_lstm2 - H_e))/(MathFunctions.rms_flat(H_t))

print e_deep
print e_deep2

S, unique, pk = DataFunctions.calculate_entropy(H_t)
print "The entropy value is: ", S


### Reshape arrays for daily neural network
X_day_t = numpy.reshape(X_day_t, (X_day_t.shape[0], 1, X_day_t.shape[1]))
X_day_e = numpy.reshape(X_day_e, (X_day_e.shape[0], 1, X_day_e.shape[1]))
H_day_t = numpy.concatenate((H_mean_t, H_max_t, H_min_t), axis=1)
H_day_e = numpy.concatenate((H_mean_e, H_max_e, H_min_e), axis=1)


#### Implement MLP Neural Network for providing a baseline
best_NN = NNFunctions.NN_optimizeNN_v21(X1, H1, X2, H2)
NN_model = NNFunctions.CreateRealSchedule_v21(best_NN, X1.shape[1])
NN_savemodel = NN_model

epsilon = 1
val_loss_v = []

for attempts in range(0, 5):
    NN_model = NNFunctions.CreateRealSchedule_v21(best_NN, X1.shape[1])
    NN_history = NN_model.fit(X1, H1, validation_data=(X2, H2), nb_epoch=50, batch_size=1, verbose=0, callbacks=callbacks)

    loss_v = NN_history.history['val_loss']
    val_loss_check = loss_v[-1]
    #print val_loss_check
    val_loss_v.append(val_loss_check)

    if val_loss_v[attempts] < epsilon:
        epsilon = val_loss_v[attempts]
        NN_savemodel = NN_model



Y_NN = NN_savemodel.predict(X3)
e_NN = (MathFunctions.rms_flat(Y_NN - H3))/(MathFunctions.rms_flat(H3))
e_NN2 = (MathFunctions.rms_flat(Y_NN - H3))/(MathFunctions.rms_flat(H1))



#### Plotting
PlotFunctions.Plot_double(t_array, H_e, t_array, Y_lstm2, 'Actual conv power','LSTM conv power', 'k-', 'r-', "fig_HVAC1a.eps")
PlotFunctions.Plot_triple(t_train, H_t, t_test, Y_lstm2, t_test, H_e, 'Training Data', 'Model B predictions', 'Test Data (actual)', 'k-', 'r-', 'b-', "fig_HVACA1b.eps")
#PlotFunctions.Plot_triple(t_train, H_t, t_test, Y_lstm, t_test, H_e, 'Training Data', 'Model B predictions', 'Test Data (actual)', 'k-', 'r-', 'b-', "fig_HVAC1c.eps")
PlotFunctions.Plot_quadruple(t_train, H_t, t_test, Y_lstm2, t_test, Y_NN, t_test, H_e, 'Training Data', 'Model B predictions', 'MLP Predictions', 'Test Data (actual)', 'k-', 'r-', 'y-', 'b-', "fig_HVACA1d.eps")

###----------------------------------------------------------------
#Fitting model for MT-LSTM
def objective(params):
    optimize_model = build_lstm_v1.lstm_multi_101b(params, train_data2.shape[2], lstm_h2, tsteps2) #Check code here, relu entering
    loss_out = NNFunctions.model_optimizer_101(optimize_model, [train_data2, h_input], H_t2, [val_data2, h_val], H_val2, 5)
    return {'loss': loss_out, 'status': STATUS_OK}


trials = Trials()
best2 = fmin(objective, space2, algo=tpe.suggest, trials=trials, max_evals=10)


#Building Stateful Model
lstm_hidden = hyperopt.space_eval(space2, best2)
print lstm_hidden

#Building model
lstm_model = build_lstm_v1.lstm_multi_101b(lstm_hidden, train_data2.shape[2], lstm_h2, tsteps2)
save_model = lstm_model

##callbacks for Early Stopping
callbacks = [EarlyStopping(monitor='val_loss', patience=5)]

#parameters for simulation
attempt_max = 3
epoch_max = 20
min_epoch = 10

#Criterion for early stopping
tau = 5
e_mat = numpy.zeros((epoch_max, attempt_max))
e_temp = numpy.zeros((tau, ))

tol = 0
count = 0
val_loss_v = []
epsilon = 1 #initialzing error
loss_old = 1
loss_val = 1


for attempts in range(attempt_max):
    lstm_model = build_lstm_v1.lstm_multi_101b(lstm_hidden, train_data2.shape[2], lstm_h2, tsteps2)
    print "New model Initialized"

    for ep in range(epoch_max):
        lstm_history = lstm_model.fit([train_data2, h_input], H_t2, batch_size=1, nb_epoch=1, validation_split=0, shuffle=False)
        loss_old = loss_val
        loss_val = lstm_history.history['loss']

        # testing alternative block
        y_val2 = lstm_model.predict([val_data2, h_val], batch_size=1, verbose=0)
        e1, e2 = DataFunctions.find_error2(H_t2, H_val2, y_val2, tsteps2)
        print e1, e2
        val_loss_check = e2
        val_loss_v.append(val_loss_check)
        e_mat[ep, attempts] = val_loss_check

        if val_loss_v[count] < epsilon and loss_val < loss_old:
            epsilon = val_loss_v[count]
            save_model = lstm_model
            #save_model.save('multi_model_HVAC1.h5')
            test_model = lstm_model
            Y_lstm = test_model.predict([test_data2, h_test], batch_size=1, verbose=0)
            e_1, e_2 = DataFunctions.find_error2(H_t2, H_e2, Y_lstm, tsteps2)
            test_model.reset_states()
            print e_1
            print e_2

        count = count + 1
        lstm_model.reset_states()


        #This block is for early stopping
        if ep>=min_epoch:
            e_temp = e_mat[ep - tau + 1: ep + 1, attempts]
            e_local = e_mat[ep-tau, attempts]

            if numpy.all(e_temp > e_local):
                break



#save_model = load_model('multi_model_HVAC1.h5')
Y_lstm2 = save_model.predict([test_data2, h_test], batch_size=1, verbose=0)

#### Error analysis
H_t2 = numpy.reshape(H_t2, (H_t.shape[0]*tsteps2, 1))
H_e2 = numpy.reshape(H_e2, (H_e.shape[0]*tsteps2, 1))
Y_lstm = numpy.reshape(Y_lstm, (Y_lstm.shape[0]*tsteps2, 1))
Y_lstm2 = numpy.reshape(Y_lstm2, (Y_lstm2.shape[0]*tsteps2, 1))
t_train2 = numpy.arange(0, len(H_t2))
t_test2 = numpy.arange(len(H_t2), len(H_t2)+len(Y_lstm2))
t_array2 = numpy.arange(0, len(Y_lstm2))
numpy.save('HVAC_critical_10min.npy', Y_lstm2)

e_deep = (MathFunctions.rms_flat(Y_lstm2 - H_e2))/(MathFunctions.rms_flat(H_e2))
e_deep2 = (MathFunctions.rms_flat(Y_lstm2 - H_e2))/(MathFunctions.rms_flat(H_t2))

print "10-min errrors: "
print e_deep
print e_deep2

Y_lstm2 = Y_lstm2*H_max2 #Converting to actual value
Y_lstm2 = Y_lstm2/6 #Converting to KW-h
Y_hour = DataFunctions.reduce_by_sum(Y_lstm2, 6)/H_max

print "H values"
print Y_hour
print H_e

e_deep3 = (MathFunctions.rms_flat(Y_hour - H_e))/(MathFunctions.rms_flat(H_e))
e_deep4 = (MathFunctions.rms_flat(Y_hour - H_e))/(MathFunctions.rms_flat(H_t))

print "Errors: "
print e_deep3
print e_deep4

