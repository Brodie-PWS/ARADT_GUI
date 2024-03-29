# Import ML Libaries
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as ssig
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import math
import librosa
import joblib
import time
import glob
import pickle
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from feature_extraction import LinearityDegreeFeatures, HighPowerFrequencyFeatures, extract_lpcc, calc_stft, _stft
from tkinter import messagebox

SVM_TYPES = ['SVM', 'svm']
MLP_TYPES = ['MLP', 'mlp', 'NN', 'nn']
RF_TYPES = ['rf', 'RF']
KNN_TYPES = ['knn', 'KNN']

def extract_features(associations_dict):
    # Initialize parameters:
    W = 14
    # Peak selection threshold:
    omega = 0.3
    # Number of points in each signal segment (window size):
    nperseg = 512
    # Hop length of the window is 25% nperseg (with 75% overlap):
    noverlap = 512-128
    # Number of FFT points:
    nfft = 2048
    # Calculate the number of segments k in S_pow:
    k = int((nfft/2 + 1) / W)

    # Create an empty Numpy array to store extracted features as well as corresponding labels:
    feature_vect = np.zeros((len(associations_dict.keys()), 98))
    idx = 0
    for sample_path, label in associations_dict.items():
        sample_label = label

        # ------ Stage 1: Signal transformation ------
        # Read the input signal:
        signal, _ = librosa.load(sample_path, sr=16000)

        # Compute STFT for the input signal:
        sig_stft = _stft(signal)

        # Compute S_pow from STFT:
        S_pow = np.sum(np.abs(sig_stft)**2/nfft, axis=1)

        # ------ Stage 2: Feature Extraction ------
        # Calculate the sum of power in each segment (in total k segments):
        power_vec = np.zeros(k)
        for i in np.arange(k):
            power_vec[i] = np.sum(S_pow[i*W:(i+1)*W])
        # Normalize power_vec as power_normal:
        power_normal = power_vec / np.sum(power_vec)

        # Feature 1: FV_LFP - low frequencies power features
        FV_LFP = power_normal[0:48] * 100
        #print(FV_LFP)

        # Feature 2: FV_LDF - signal power linearity degree features
        _, FV_LDF = LinearityDegreeFeatures(power_normal)
        #FV_LDF = np.zeros(2)

        # Feature 3: FV_HPF - high power frequency features
        FV_HPF = HighPowerFrequencyFeatures(FV_LFP, omega)
        #FV_HPF = np.zeros(35)

        # Feature 4: FV_LPC - linear prediction cesptrum coefficients
        FV_LPC = extract_lpcc((sample_path), 12)
        #FV_LPC = np.zeros(12)

        # ------ Stage 3: Attack Detection ------
        # Normalize each sub-feature:
        '''
        mean_LFP = np.mean(FV_LFP)
        FV_LFP = (FV_LFP - mean_LFP) / (FV_LFP.max() - FV_LFP.min())
        mean_LDF = np.mean(FV_LDF)
        FV_LDF = (FV_LDF - mean_LDF) / (FV_LDF.max() - FV_LDF.min())
        mean_HPF = np.mean(FV_HPF)
        FV_HPF = (FV_HPF - mean_HPF) / (FV_HPF.max() - FV_HPF.min())
        mean_LPC = np.mean(FV_LPC)
        FV_LPC = (FV_LPC - mean_LPC) / (FV_LPC.max() - FV_LPC.min())
        '''

        # Construct the final feature of length 97 (= 2 + 35 + 12 + 48):
        FV_Void = np.concatenate((FV_LDF, FV_HPF, FV_LPC, FV_LFP))

        feature_vect[idx, 0:97] = FV_Void
        feature_vect[idx, 97] = label
        idx += 1

    return feature_vect

def store_extracted_features(extracted_features):
    current_timestamp = time.strftime("%d-%m-%Y-(%H%M)")
    filename = f'Extracted_Features/{current_timestamp}'
    np.save(filename, extracted_features)

def predict_pipeline(model_fpath, chosen_samples):
    # Initialize parameters:
    W = 14
    # Peak selection threshold:
    omega = 0.3
    # Number of points in each signal segment (window size):
    nperseg = 512
    # Hop length of the window is 25% nperseg (with 75% overlap):
    noverlap = 512-128
    # Number of FFT points:
    nfft = 2048
    # Calculate the number of segments k in S_pow:
    k = int((nfft/2 + 1) / W)

    # Load the Model
    pipe = joblib.load(model_fpath)

    prediction_list = []
    for sample in chosen_samples:
        # ------ Stage 1: Signal transformation ------
        # Read the input signal:
        signal, _ = librosa.load(sample, sr=16000)

        # Compute STFT for the input signal:
        sig_stft = _stft(signal)

        # Compute S_pow from STFT:
        S_pow = np.sum(np.abs(sig_stft)**2/nfft, axis=1)

        # ------ Stage 2: Feature Extraction ------
        # Calculate the sum of power in each segment (in total k segments):
        power_vec = np.zeros(k)
        for i in np.arange(k):
            power_vec[i] = np.sum(S_pow[i*W:(i+1)*W])
        # Normalize power_vec as power_normal:
        power_normal = power_vec / np.sum(power_vec)

        # Feature 1: FV_LFP - low frequencies power features
        FV_LFP = power_normal[0:48] * 100
        #print(FV_LFP)

        # Feature 2: FV_LDF - signal power linearity degree features
        _, FV_LDF = LinearityDegreeFeatures(power_normal)
        #FV_LDF = np.zeros(2)

        # Feature 3: FV_HPF - high power frequency features
        FV_HPF = HighPowerFrequencyFeatures(FV_LFP, omega)
        #FV_HPF = np.zeros(35)

        # Feature 4: FV_LPC - linear prediction cesptrum coefficients
        FV_LPC = extract_lpcc((sample), 12)
        #FV_LPC = np.zeros(12)

        # ------ Stage 3: Attack Detection ------
        # Normalize each sub-feature:
        '''
        mean_LFP = np.mean(FV_LFP)
        FV_LFP = (FV_LFP - mean_LFP) / (FV_LFP.max() - FV_LFP.min())
        mean_LDF = np.mean(FV_LDF)
        FV_LDF = (FV_LDF - mean_LDF) / (FV_LDF.max() - FV_LDF.min())
        mean_HPF = np.mean(FV_HPF)
        FV_HPF = (FV_HPF - mean_HPF) / (FV_HPF.max() - FV_HPF.min())
        mean_LPC = np.mean(FV_LPC)
        FV_LPC = (FV_LPC - mean_LPC) / (FV_LPC.max() - FV_LPC.min())
        '''

        # Construct the final feature of length 97 (= 2 + 35 + 12 + 48):
        FV_Void = np.concatenate((FV_LDF, FV_HPF, FV_LPC, FV_LFP))
        # Reshape the Feature Vector
        reshaped_FV_Void = FV_Void.reshape(1, -1)

        pred = predict(pipe, reshaped_FV_Void)

        # Reformat Prediction
        predictions = pred.tolist()
        if predictions[0] == 1:
            prediction = f'[{os.path.basename(sample)}] is [Genuine] Voice'
        else:
            prediction = f'[{os.path.basename(sample)}] is [Spoof] Voice'
        prediction_list.append(prediction)

    return prediction_list

def predict(model, feature_vector):
    # Ask Model to make prediction on the extracted Features of the Sample
    pred = model.predict(feature_vector)
    return pred

def make_model(model_type, model_params):
    print('\n-------------------- CREATING NEW MODEL --------------------')
    print(f'Model Type: {model_type}\nModel Parameters:{model_params}')
    classifier = None
    if model_type.lower() in SVM_TYPES:
        c_val, kernel = unpack_dict(model_params, 'C Parameter', 'Kernel Type')
        classifier = svm.SVC(C=float(c_val), kernel=kernel.lower(), gamma='auto', class_weight=None)
        return classifier
    elif model_type.lower() in MLP_TYPES:
        solver_val, activation_val, max_iter_val = unpack_dict(model_params, 'Solver Type', 'Activation Function Type', 'Number of Iterations')
        classifier = MLPClassifier(solver=solver_val.lower(), activation=activation_val.lower(), max_iter=max_iter_val)
        return classifier
    elif model_type.lower() in RF_TYPES:
        n_estimators_val, max_depth_val = unpack_dict(model_params, 'Number of Estimators', 'Max Depth')
        classifier = RandomForestClassifier(n_estimators=n_estimators_val, max_depth=max_depth_val)
        return classifier
    elif model_type.lower() in KNN_TYPES:
        n_neighbors_val, algorithm_val = unpack_dict(model_params, 'Number of Neighbors', 'Algorithm Type')
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors_val, algorithm=algorithm_val.lower())
        return classifier
    else:
        print(f'Could not create a Model with type: {model_type}')
        return classifier

def make_optimised_model(model_type, features_labels_fpath=None):
    print('\n-------------------- CREATING NEW OPTIMIZED MODEL --------------------')
    print(f'Model Type: {model_type}')
    classifier = None
    parameter_space = {}
    if model_type in SVM_TYPES:
        classifier = svm.SVC()
        parameter_space = {
            'C': [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 3, 6.0, 12.0, 24.0, 48.0],
            'kernel': ['rbf', 'linear', 'poly']
        }
    elif model_type in MLP_TYPES:
        classifier = MLPClassifier(max_iter=10000)
        parameter_space = {
            'hidden_layer_sizes': [(25,25,25), (50,50,50), (50, 100, 50), (100,)],
            'activation': ['relu', 'tanh'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01, 0.05],
            'learning_rate': ['constant', 'adaptive']
        }
    elif model_type in RF_TYPES:
        classifier = RandomForestClassifier()
        parameter_space = {
            'n_estimators': [25, 50, 100, 250, 500, 1000],
            'criterion': ['gini', 'entropy'],
            'max_features': ['auto', 'log2']
        }
    elif model_type in KNN_TYPES:
        classifier = KNeighborsClassifier()
        parameter_space = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'algorithm': ['auto', 'kd_tree', 'ball_tree', 'brute'],
            'p': [1, 2, 4, 8, 16]
        }
    else:
        print(f'Cannot create a Classifier with Model Type: {model_type}')

    if classifier is not None and parameter_space:
        print(f'\nPerforming Automated Parameter Tuning on {classifier.__class__.__name__} using: \n{parameter_space}')
        # If a Path was provided use it, otherwise use the latest file
        if features_labels_fpath:
            file_to_load = features_labels_fpath
        else:
            file_list =  glob.glob('Extracted_Features/*')
            # Determine the latest file
            try:
                file_to_load = max(file_list, key=os.path.getctime)
            except Exception as ex:
                messagebox.showinfo('Error!', 'There are no files in the Extracted Features Directory')
                return

        print(f'Using File [{file_to_load}] to Train the Model')

        # Load File
        extracted_features = np.load(file_to_load, allow_pickle=True)
        # Load the Void features and corresponding labels:
        x_train, y_train = np.split(extracted_features, indices_or_sections=(97,), axis=1)

        clf = GridSearchCV(classifier, parameter_space, n_jobs=-1, cv=2)
        clf.fit(x_train, y_train.ravel())

        report_str = '----- FINISHED AUTOMATED PARAMETER TUNING -----\n\n'
        report_str += f'Best Parameters found: \n{clf.best_params_}\n'

        report_str += '\nSummary of all Parameter Combinations:'
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            report_str += ("\n%0.3f (+/-%0.03f) with Parameters: %r" % (mean, std * 2, params))

        # Save Model
        s = pickle.dumps(clf)
        save_name = os.path.join(os.getcwd(), 'Models', f'{clf.__class__.__name__}({classifier.__class__.__name__}).pkl')
        f = open(save_name, 'wb+')
        f.write(s)
        f.close()
        print("Model saved into " + save_name)
        return report_str

def train_model(model, feature_file=None, test_run=None):
    message = None
    if model is None:
        return f'Cannot Train Model: {model}'
    print('\n-------------------- TRAINING NEW MODEL --------------------')

    # If a Feature File was not specified prior, use the latest one instead
    if not feature_file:
        # Get list of all Txt files in Extracted Features directory
        file_list =  glob.glob('Extracted_Features/*')
        # Determine the latest file
        try:
            feature_file = max(file_list, key=os.path.getctime)
            if not test_run:
                messagebox.showinfo('Using latest Extracted Features', f'No file containing Extracted Features was specified\nUsing {feature_file} to train the model')
        except Exception as ex:
            messagebox.showinfo('Error!', 'There are no previous Predictions to show')
            return

    print(f'Using File [{feature_file}] to Train the Model')

    # Load File
    extracted_features = np.load(feature_file, allow_pickle=True)

    # Load the Void features and corresponding labels:
    x_train, y_train = np.split(extracted_features, indices_or_sections=(97,), axis=1)
    model.fit(x_train, y_train.ravel())

    # Save Model if this method has NOT been called from the Unit Tests
    if not test_run:
        s = pickle.dumps(model)
        current_timestamp = time.strftime("%d-%m(%H%M)")
        save_name = os.path.join(os.getcwd(), 'Models', f'{model.__class__.__name__}({current_timestamp}).pkl')
        f = open(save_name, 'wb+')
        f.write(s)
        f.close()
        print("Model saved into " + save_name)

    message = f'{model.__class__.__name__} Model saved into Models Directory'
    return message

def unpack_dict(dict, *keys):
    for key in keys:
        yield dict[key]
