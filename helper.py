# (basic) dependencies
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm as tqdm

# sklearn dependencies
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# tensorflow dependencies
from tensorflow.keras import datasets, layers, models, regularizers, backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

if os.environ['WANDB_API_KEY'] is not '':
    import wandb
    from wandb.keras import WandbCallback

MODEL_DIR = 'models'
max_num_objects = 25  # have one event with 18 objects, add some for compatibility


def metrics(ytrue, ypred, verbose=True):
    # calculate confusion matrix
    possible_labels = ytrue.unique()
    conf_mat = confusion_matrix(ytrue, ypred, possible_labels)
    
    # calculate accuracy
    accuracy = accuracy_score(ytrue, ypred)
    
    if verbose:
        print(possible_labels)
        print(conf_mat)
        print("Accuracy: {:.5f}".format(accuracy))
        
    return conf_mat, accuracy


def learn(model, train, validation, features, target, verbose=True):
    """Train and validate model. The provided model should 
    support the following methods: fit and predict"""
    if verbose:
        print("Fitting classifier...")
        
    # train the model on the training data
    model.fit(train[features], train[target])
    
    # calculate predictions for the validation data
    ypred = model.predict(validation[features])
    
    # calculate (and show) metrics
    conf_mat, accuracy = metrics(validation[target], ypred, verbose)
        
    return model, ypred, conf_mat, accuracy


def select_features(df, cat_type='one-hot', trans='both', num_objects=-1):
    """
    Select features. Options:
    - cat_type: one-hot encoding of obj category (one-hot), numerical representation (num), or both (both)
    - trans: for MET, E and pt, select either log, plain or both version(s)
    - num objects: number of objects to include (default -1 includes all)
    """

    # start with angle features (including event-level)
    # selects METphi, phi1, eta1, phi2, eta2, ...
    # use ^phi to omit METphi
    features = df.filter(regex='phi|eta').columns

    # select plain, log transformed version or both
    # of MET, energy and transverse momentum
    if trans == 'log' or trans == 'both':
        add_features = df.filter(regex='MET_log|E\d+_log|pt\d+_log').columns
        features = features.append(add_features)
    if trans == 'plain' or trans == 'both':
        add_features = df.filter(regex='MET$|E\d+$|pt\d+$').columns
        features = features.append(add_features)

    # select categorical representation
    if cat_type == 'one-hot' or cat_type == 'both':
        # add one-hot obj columns: objx_y, with y one charater and +- optional
        add_features = df.filter(regex='obj\d+_[a-z].$').columns
        features = features.append(add_features)
    if cat_type == 'num' or cat_type == 'both':
        # add num obj columns: objx_num
        add_features = df.filter(regex='obj\d+_num$').columns
        features = features.append(add_features)
    
    # Note: use this block last
    # select number of objects to include (-1 for all)
    if num_objects >= 0:
        # filter objx, Ex, ptx, etax, phix for x > num_objects
        # also filter Ex_log and ptx_log
        # all features belonging to these objects are captured 
        # with a regex when their number is present in the feature name
        # this can also select higher numbers, but that is fine in this case
        for num in range(1+num_objects, 1+max_num_objects):
            filter_features = df.filter(regex='{}'.format(num)).columns
            features = features.difference(filter_features)

    return features


def load_data(data_folder='data', norm='minmax', cat_type='one-hot', trans='log', num_objects=-1):
    """Load train and validation data from data_folder. 
    Also load a feature set. Assumes target is 'target'. 
    Available normalizations: minmax and zscore."""
    
    # load dataset
    data = pd.read_csv("{}/data_{}.csv".format(data_folder, norm))
    
    # select set of features and assume target label
    features = select_features(df=data, cat_type=cat_type,
                               trans=trans, num_objects=num_objects)
    target = 'target'
    
    return data, features, target


def cross_validation(clf, train, features, target, cv):
    scores = cross_val_score(clf, train[features], train[target], cv=cv)
    print("Mean score: {:.5f}. All: {}".format(scores.mean(), scores))
    sns.barplot(scores)  # shows std visually
    plt.show()


def ROC(pred, validation, target, start=0.3, stop=0.7, stepsize=0.001):
    # vary the classification threshold and calculate the TPR (no. positive classified as positive / no. positive) 
    # and FPR (no. negative classified as positive / no. positive)
    # start with point in top right corner
    TPR, FPR = [1], [1]
    
    for thres in np.arange(start, stop, stepsize):
        # classify using the treshold and obtain metrics
        classification = (pred < thres).astype(int)
        # conf mat has: 0,0=TN, 1,1=TP, 1,0=FN?, 0,1=FP?
        conf_mat, accuracy = metrics(validation[target], classification, verbose=False)
        num_positive = conf_mat[1,:].sum()
        TPR.append(conf_mat[1,1]/num_positive)
        FPR.append(conf_mat[0,1]/num_positive)
    
    # add final dot in bottom left corner
    TPR.append(0); FPR.append(0)
    
    f, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.plot(FPR, TPR, '.:')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.plot(np.linspace(0,1), np.linspace(0,1), ':', alpha=0.5)
    plt.show()
    
    # calculate area under the curve
    AUC = auc(FPR, TPR)
    
    return TPR, FPR, AUC


def make(name):
    if not os.path.exists("{}/{}".format(MODEL_DIR, name)):
        os.makedirs("{}/{}".format(MODEL_DIR, name))


def save(args, wandb_init=True):
    make(args.id)
    args.to_csv("{}/{}/args.csv".format(MODEL_DIR, args.id), header=False)
    # start new Weights and Biases instance and save parameters
    if wandb_init:
        wandb.init(project=os.environ['WANDB_PROJECT'], name=args.id, config=dict(args))


# setting the default options here affects all subsequent calls
def make_args(name='Net', architecture='', loss='binary_crossentropy', 
              activation='relu', final_activation='sigmoid', epochs=20, dropout = 0.0,
              batch_size=32, lr=1e-3, optimizer='adam', 
              norm='minmax', trans='plain',
              nodes1=128, nodes2=64, nodes3=32, nodes4=0, nodes5=0,
              dropout1=0, dropout2=0, dropout3=0, dropout4=0, dropout5=0,
              batch_norm=[], lr_decay=0,
              verbose=True, wandb_init=False):
    # Specify parameters
    args = pd.Series({
        'id': name,
        'architecture': architecture,
        'loss': loss,
        'activation': activation,
        'final_activation': final_activation,
        'dropout': dropout,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr, #default 1e-3
        'lr_decay': lr_decay,
        'optimizer': optimizer,
        'norm': norm,
        'trans': trans,
        'nodes1': nodes1,
        'nodes2': nodes2,
        'nodes3': nodes3,
        'nodes4': nodes4,
        'nodes5': nodes5,
        'dropout1': dropout1,
        'dropout2': dropout2,
        'dropout3': dropout3,
        'dropout4': dropout4,
        'dropout5': dropout5,
        'batch_norm': batch_norm,
    })
    # create directory for id
    save(args, wandb_init=wandb_init)
    if verbose:
        print(args)
    return args


def build_architecture(args, input_shape):
    """Hyperparameter sweep friendly model builder"""
    # clear previous sessions
    backend.clear_session()
    
    # LeakyReLU cannot be used as string
    activation = args.activation
    if activation == 'LeakyReLU':
        activation = LeakyReLU()
    
    # build model
    model = models.Sequential()
    for layer_num in range(1, 1+5):
        n, dropout = args['nodes{}'.format(layer_num)], args['dropout{}'.format(layer_num)]
        if n == 0:
            break # skip

        # add layer
        if layer_num == 1:
            # add input size to first layer
            if args.activation == 'PReLU': activation = PReLU() # has trainable params, so redef.
            model.add(layers.Dense(n, activation=activation, input_shape=input_shape))
        else:
            if args.activation == 'PReLU': activation = PReLU() # has trainable params, so redef.
            model.add(layers.Dense(n, activation=activation))
        
        # add batch norm layer?
        if layer_num in args.batch_norm:
            model.add(layers.BatchNormalization())
        
        # add dropout, if enabled
        if dropout > 0:
            model.add(layers.Dropout(rate=dropout))
    
    # add output label
    model.add(layers.Dense(1, activation=args.final_activation))
    
    # compile and return model
    optimizer = args.optimizer
    if optimizer == 'adam':
        optimizer = Adam(lr=args.lr)
        
    model.compile(optimizer=optimizer, loss=args.loss,
            metrics=['acc'])
    
    return model


def train_net(args, train_features, train_labels, val_features, val_labels, model=None, 
          keras_verbose=0, verbose=True, evaluate=True, prediction_threshold='infer'):

    # convert input data to numpy arrays, if DataFrames are provided
    if isinstance(train_features, pd.DataFrame):
        train_features = train_features.values
    if isinstance(val_features, pd.DataFrame):
        val_features = val_features.values

    # Set up the model, if not provided
    if model is None:
        model = build_architecture(args, input_shape=(train_features.shape[1],))
    
    if verbose:
        model.summary()

    # initialize Weights & Biases callback, if applicable
    if os.environ['WANDB_API_KEY'] is not '':
        callbacks = [WandbCallback()]
    else:
        callbacks = []

    # add learning rate decay, if set
    if args.lr_decay > 0:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=args.lr_decay,
                                      patience=7, min_lr=1e-6)
        callbacks.append(reduce_lr)

    # train the model
    history = model.fit(train_features, train_labels, 
                        validation_data=(val_features, val_labels), 
                        batch_size=args.batch_size,
                        epochs=args.epochs, verbose=keras_verbose, callbacks=callbacks)

    if evaluate:
        # evaluate model performance
        # if prediction threshold needs to be inferred, try different values for
        # the training set and use the optimal value for the validation predictions
        if prediction_threshold == 'infer':
            print("Inferring optimal prediction threshold..")
            accs = [] 
            # skip unlikely values <0.2 and >0.8
            search_space = np.linspace(0.2, 0.8, num=31)
            for threshold in tqdm(search_space):
                # calculate predictions for the validation data
                ypred = model.predict(train_features)
                pred = (ypred > threshold)

                # calculate (and show) metrics
                conf_mat, accuracy = metrics(train_labels, pred, verbose=False)
                accs.append(accuracy)
            # select optimal value for train set as threshold
            prediction_threshold = search_space[np.argmax(accs)]
            if verbose:
                print("Optimal threshold: {}".format(prediction_threshold))

        # calculate predictions for the validation data
        ypred = model.predict(validation[features])
        pred = (ypred > prediction_threshold)

        # calculate (and show) metrics
        conf_mat, accuracy = metrics(validation[target], pred, verbose=True)

    return model, history


def train_args(args_default):
    def train():
        # setup WandB with default configurations
        run = wandb.init(magic=True)

        # transfer config to args Series
        args = args_default.copy()
        for key, val in dict(run.config).items():
            args['{}'.format(key)] = val

        # read train and validation data
        train_set, validation, features, target = load_data('data', trans=args.trans, norm=args.norm)

        # build model
        model = build_architecture(args, input_shape=(train_set[features].shape[1],))

        # train model
        model, history = train_net(args, train_set[features], train_set[target], validation[features], validation[target],
                  model=model, verbose=False, keras_verbose=0, evaluate=False)
        print("Train acc: {}, val acc: {}".format(history.history['acc'][-1], history.history['val_acc'][-1]))

        return history
    return train

