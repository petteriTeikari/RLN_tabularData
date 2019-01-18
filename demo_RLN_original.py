# from:
# https://github.com/irashavitt/regularization_learning_networks/blob/master/Implementations/Keras_tutorial.ipynb
import numpy as np
from numpy.random import seed
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from regularization_learning_networks_local.Implementations import RLNCallback

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from keras import regularizers
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from keras.backend import eval as keras_eval
import warnings
warnings.filterwarnings("ignore")

MJTCP_SEED = 32292 # Michael Jordan total career points


def base_model(layers=4, l1=0):
    assert layers > 1

    def build_fn():
        inner_l1 = l1
        # create model
        model = Sequential()
        # Construct the layers of the model to form a geometric series
        prev_width = INPUT_DIM
        for width in np.exp(np.log(INPUT_DIM) * np.arange(layers - 1, 0, -1) / layers):
            width = int(np.round(width))
            model.add(Dense(width, input_dim=prev_width, kernel_initializer='glorot_normal', activation='relu',
                            kernel_regularizer=regularizers.l1(inner_l1)))
            # For efficiency we only regularized the first layer
            inner_l1 = 0
            prev_width = width

        model.add(Dense(1, kernel_initializer='glorot_normal'))

        # Compile model
        model.compile(loss='mean_squared_error', optimizer='rmsprop')

        # print('BASE MODEL SUMMARY')
        # print(model.summary())

        return model

    return build_fn


def RLN(layers=4, **rln_kwargs):
    def build_fn():
        model = base_model(layers=layers)()

        # For efficiency we only regularized the first layer
        rln_callback = RLNCallback(model.layers[0], **rln_kwargs)

        # Change the fit function of the model to except rln_callback:
        orig_fit = model.fit

        def rln_fit(*args, **fit_kwargs):
            orig_callbacks = fit_kwargs.get('callbacks', [])
            rln_callbacks = orig_callbacks + [rln_callback]
            return orig_fit(*args, callbacks=rln_callbacks, **fit_kwargs)

        model.fit = rln_fit

        return model

    return build_fn

def test_regression_model(build_fn, modle_name, num_repeats=10):

    # NOTE!
    # The negative MSE might be confusing, but it depends a bit what parts of sklearn you use
    # https://github.com/scikit-learn/scikit-learn/issues/2439
    # So the closer to Zero the MSE, the better

    # For regression, see e.g.
    # https://stackoverflow.com/questions/44132652/keras-how-to-perform-a-prediction-using-kerasregressor

    seed(MJTCP_SEED)
    results = np.zeros(num_repeats)
    acc = np.zeros(num_repeats)
    results_cv = np.zeros(num_repeats)
    for i in range(num_repeats):

        print('Repeat #', i+1, '/', num_repeats)
        reg = KerasRegressor(build_fn=build_fn, epochs=100, batch_size=10, verbose=0)
        reg.fit(x_train, y_train)
        # prediction = reg.predict(x_train)
        # acc[i] = accuracy_score(y_test, prediction) # TODO!
        results[i] = reg.score(x_test, y_test)
        print('   MSE (no cross-validation) = ', results[i])

        # n_splits = 10
        # kfold = KFold(n_splits=10, random_state=seed)
        # results_cv[i] = cross_val_score(reg, x_train, y_train, cv=kfold)
        # print('   MSE (cross-validation, ', n_splits, ' splits) = ', results_cv[i])
        # # print('   Accuracy = ', acc[i])
        # sys.stdout.flush()

    print("\n%s: Mean of MSE from %i repeats: %.2f (stdev = %.2f)" % (
        modle_name, num_repeats, results.mean(), results.std()))
    # print("%s: Mean of MSE with cross validation from %i repeats: %.2f (stdev = %.2f)" % (
    #     modle_name, num_repeats, results_cv.mean(), results_cv.std()))
    # print("%s: Mean Accuracy from %i repeats: %.2f (stdev = %.2f)" % (
    #     modle_name, num_repeats, acc.mean(), acc.std()))

    return results.mean()

def import_data(path = '/home/petteri/Dropbox/manuscriptDrafts/deepPLR/code/RLN_tabularData/test_data',
                 train_data = 'canonical_train.csv',
                 test_data = 'canonical_test.csv',
                 debug_mode = False):

    train_data = np.loadtxt(os.path.join(path, train_data), delimiter=',', skiprows=1)
    test_data = np.loadtxt(os.path.join(path, test_data), delimiter=',', skiprows=1)

    shape_in = train_data.shape
    no_rows = shape_in[0]
    no_cols = shape_in[1]

    # The label is on the first column, like in UCR dataset
    x_train = train_data[:, 1:no_cols]
    y_train = train_data[:, 0]

    x_test = test_data[:, 1:no_cols]
    y_test = test_data[:, 0]

    if debug_mode:
        print('  - Debug mode ON, choosing just 2 first features to make sure that the code works!')
        x_train = train_data[:, 0:2]
        x_test = test_data[:, 0:2]

    # print(x_test[0,:])
    # print(y_test)

    return(x_train, y_train, x_test, y_test)


# MAIN
this_dir = sys.path[0]
os.chdir(this_dir)
path_out = os.path.join(this_dir, 'MODELS_OUT')
path_in = os.path.join(this_dir, 'test_data')
print('Output path is = ', path_out)

# Get the data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
#x_train, y_train, x_test, y_test = import_data(path = path_in, debug_mode = True)
print('Train set size = ', x_train.shape, ', labels = ', y_train.shape)
print('Test set size = ', x_test.shape, ' labels = ', y_test.shape)
print('Unique labels = ', np.unique(y_test))
INPUT_DIM = x_train.shape[1]
print('Number of input features = ', INPUT_DIM)

# Scale features, i.e. z-standardize
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train, y_train)
x_test = scaler.transform(x_test, y_test)



# Hyperparameters
layers = 4
best_rln_score = np.inf
Theta, learning_rate = None, None


for cur_Theta, log_learning_rate in [(-8, 6), (-10, 5), (-10, 6), (-10, 7), (-12, 6)]:

    cur_learning_rate = np.power(10, log_learning_rate)
    cur_score = test_regression_model(RLN(layers=layers, norm=1, avg_reg=cur_Theta, learning_rate=cur_learning_rate),
                           "RLN with Theta=%s and learning_rate=%.1E" % (cur_Theta, cur_learning_rate))

    if cur_score < best_rln_score:
        Theta, learning_rate = cur_Theta, cur_learning_rate
        best_rln_score = cur_score

print("The best RLN is achieved with Theta=%d and learning_rate=%.1E" % (Theta, learning_rate))