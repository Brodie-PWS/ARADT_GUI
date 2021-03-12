import unittest
import mock

import record
import predict
import os
from ddt import ddt, data, unpack
from pathlib import Path

@ddt
class TestMakeModel(unittest.TestCase):
    @unpack
    @data(
        {'model_type': 'svm', 'c_val': 1, 'kernel': 'rbf', 'expected': 'SVC(C=1, gamma=\'auto\')'},
        {'model_type': 'SVM', 'c_val': 2, 'kernel': 'rbf', 'expected': 'SVC(C=2, gamma=\'auto\')'},
        {'model_type': 'svm', 'c_val': 1, 'kernel': 'poly', 'expected': 'SVC(C=1, gamma=\'auto\', kernel=\'poly\')'},
        {'model_type': 'SVM', 'c_val': 2, 'kernel': 'poly', 'expected': 'SVC(C=2, gamma=\'auto\', kernel=\'poly\')'},
        {'model_type': 'svm', 'c_val': 1, 'kernel': 'linear', 'expected': 'SVC(C=1, gamma=\'auto\', kernel=\'linear\')'},
        {'model_type': 'SVM', 'c_val': 2, 'kernel': 'linear', 'expected': 'SVC(C=2, gamma=\'auto\', kernel=\'linear\')'},
        {'model_type': 'invalid_model', 'c_val': 1, 'kernel': 'rbf', 'expected': 'None'},
        {'model_type': 'invalid_model', 'c_val': 1, 'kernel': 'linear', 'expected': 'None'},
    )
    def test_make_svm(self, model_type, c_val, kernel, expected):
        model_params = {}
        model_params['c_val'] = c_val
        model_params['kernel'] =  kernel
        model = predict.make_model(model_type, model_params)
        self.assertEqual(expected, f'{model}')

    @unpack
    @data(
        {'model_type': 'nn', 'solver_val': 'lbfgs', 'activation_val': 'identity',
        'max_iter_val': 1000, 'expected': 'MLPClassifier(activation=\'identity\', max_iter=1000, solver=\'lbfgs\')'},
        {'model_type': 'NN', 'solver_val': 'lbfgs', 'activation_val': 'identity',
        'max_iter_val': 1000, 'expected': 'MLPClassifier(activation=\'identity\', max_iter=1000, solver=\'lbfgs\')'},
        {'model_type': 'mlp', 'solver_val': 'lbfgs', 'activation_val': 'identity',
        'max_iter_val': 1000, 'expected': 'MLPClassifier(activation=\'identity\', max_iter=1000, solver=\'lbfgs\')'},
        {'model_type': 'MLP', 'solver_val': 'lbfgs', 'activation_val': 'identity',
        'max_iter_val': 1000, 'expected': 'MLPClassifier(activation=\'identity\', max_iter=1000, solver=\'lbfgs\')'},

        {'model_type': 'nn', 'solver_val': 'sgd', 'activation_val': 'identity',
        'max_iter_val': 1000, 'expected': 'MLPClassifier(activation=\'identity\', max_iter=1000, solver=\'sgd\')'},
        {'model_type': 'NN', 'solver_val': 'sgd', 'activation_val': 'logistic',
        'max_iter_val': 1000, 'expected': 'MLPClassifier(activation=\'logistic\', max_iter=1000, solver=\'sgd\')'},
        {'model_type': 'mlp', 'solver_val': 'sgd', 'activation_val': 'tanh',
        'max_iter_val': 1000, 'expected': 'MLPClassifier(activation=\'tanh\', max_iter=1000, solver=\'sgd\')'},
        {'model_type': 'MLP', 'solver_val': 'sgd', 'activation_val': 'relu',
        'max_iter_val': 1000, 'expected': 'MLPClassifier(max_iter=1000, solver=\'sgd\')'},

        {'model_type': 'nn', 'solver_val': 'adam', 'activation_val': 'identity',
        'max_iter_val': 1000, 'expected': 'MLPClassifier(activation=\'identity\', max_iter=1000)'},
        {'model_type': 'NN', 'solver_val': 'adam', 'activation_val': 'logistic',
        'max_iter_val': 1000, 'expected': 'MLPClassifier(activation=\'logistic\', max_iter=1000)'},
        {'model_type': 'mlp', 'solver_val': 'adam', 'activation_val': 'tanh',
        'max_iter_val': 1000, 'expected': 'MLPClassifier(activation=\'tanh\', max_iter=1000)'},
        {'model_type': 'MLP', 'solver_val': 'adam', 'activation_val': 'relu',
        'max_iter_val': 1000, 'expected': 'MLPClassifier(max_iter=1000)'},

        {'model_type': 'invalid_model', 'solver_val': 'adam', 'activation_val': 'identity',
        'max_iter_val': 1000, 'expected': 'None'},
        {'model_type': 'invalid_model', 'solver_val': 'adam', 'activation_val': 'logistic',
        'max_iter_val': 1000, 'expected': 'None'},
        {'model_type': 'invalid_model', 'solver_val': 'adam', 'activation_val': 'tanh',
        'max_iter_val': 1000, 'expected': 'None'},
        {'model_type': 'invalid_model', 'solver_val': 'adam', 'activation_val': 'relu',
        'max_iter_val': 1000, 'expected': 'None'},
    )
    def test_make_nn(self, model_type, solver_val, activation_val, max_iter_val, expected):
        model_params = {}
        model_params['solver_val'] = solver_val
        model_params['activation_val'] =  activation_val
        model_params['max_iter_val'] = max_iter_val
        model = predict.make_model(model_type, model_params)
        self.assertEqual(expected, f'{model}')

    @unpack
    @data(
        {'model_type': 'rf', 'n_estimators_val': 100, 'max_depth_val': None,
        'expected': 'RandomForestClassifier()'},
        {'model_type': 'RF', 'n_estimators_val': 100, 'max_depth_val': None,
        'expected': 'RandomForestClassifier()'},
        {'model_type': 'rf', 'n_estimators_val': None, 'max_depth_val': None,
        'expected': 'RandomForestClassifier(n_estimators=None)'},
        {'model_type': 'RF', 'n_estimators_val': None, 'max_depth_val': None,
        'expected': 'RandomForestClassifier(n_estimators=None)'},

        {'model_type': 'invalid', 'n_estimators_val': 100, 'max_depth_val': None,
        'expected': 'None'},
        {'model_type': 'invalid', 'n_estimators_val': None, 'max_depth_val': None,
        'expected': 'None'},
    )
    def test_make_rf(self, model_type, n_estimators_val, max_depth_val, expected):
        model_params = {}
        model_params['n_estimators_val'] = n_estimators_val
        model_params['max_depth_val'] = max_depth_val
        model = predict.make_model(model_type, model_params)
        self.assertEqual(expected, f'{model}')

    @unpack
    @data(
        {'model_type': 'knn', 'n_neighbors_val': 3, 'algorithm_val': 'auto',
        'expected': 'KNeighborsClassifier(n_neighbors=3)'},
        {'model_type': 'knn', 'n_neighbors_val': 3, 'algorithm_val': 'kd_tree',
        'expected': 'KNeighborsClassifier(algorithm=\'kd_tree\', n_neighbors=3)'},
        {'model_type': 'knn', 'n_neighbors_val': 3, 'algorithm_val': 'ball_tree',
        'expected': 'KNeighborsClassifier(algorithm=\'ball_tree\', n_neighbors=3)'},
        {'model_type': 'knn', 'n_neighbors_val': 3, 'algorithm_val': 'brute',
        'expected': 'KNeighborsClassifier(algorithm=\'brute\', n_neighbors=3)'},
    )
    def test_make_knn(self, model_type, n_neighbors_val, algorithm_val, expected):
        model_params = {}
        model_params['n_neighbors_val'] = n_neighbors_val
        model_params['algorithm_val'] = algorithm_val
        model = predict.make_model(model_type, model_params)
        self.assertEqual(expected, f'{model}')

@ddt
class TestTrainModel(unittest.TestCase):
    @unpack
    @data(
        {'model_type': 'svm', 'c_val': 1, 'kernel': 'rbf', 'expected': 'SVC Model saved into Models Directory'},
        {'model_type': 'SVM', 'c_val': 1, 'kernel': 'rbf', 'expected': 'SVC Model saved into Models Directory'},
        {'model_type': 'svm', 'c_val': 1, 'kernel': 'linear', 'expected': 'SVC Model saved into Models Directory'},
        {'model_type': 'SVM', 'c_val': 1, 'kernel': 'poly', 'expected': 'SVC Model saved into Models Directory'},

        {'model_type': 'invalid', 'c_val': 1, 'kernel': 'rbf', 'expected': 'Cannot Train Model: None'},
        {'model_type': 'invalid', 'c_val': 1, 'kernel': 'rbf', 'expected': 'Cannot Train Model: None'},
    )
    def test_train_svm_model(self, model_type, c_val, kernel, expected):
        # Create new Model
        model_params = {}
        model_params['c_val'] = c_val
        model_params['kernel'] =  kernel
        model = predict.make_model(model_type, model_params)

        return_str = predict.train_model(model, test_run=True)

        self.assertEqual(expected, return_str)

    @unpack
    @data(
        {'model_type': 'nn', 'solver_val': 'lbfgs', 'activation_val': 'identity',
        'max_iter_val': 1000, 'expected': 'MLPClassifier Model saved into Models Directory'},
        {'model_type': 'NN', 'solver_val': 'lbfgs', 'activation_val': 'identity',
        'max_iter_val': 1000, 'expected': 'MLPClassifier Model saved into Models Directory'},
        {'model_type': 'mlp', 'solver_val': 'lbfgs', 'activation_val': 'identity',
        'max_iter_val': 1000, 'expected': 'MLPClassifier Model saved into Models Directory'},
        {'model_type': 'MLP', 'solver_val': 'lbfgs', 'activation_val': 'identity',
        'max_iter_val': 1000, 'expected': 'MLPClassifier Model saved into Models Directory'},

        {'model_type': 'invalid', 'solver_val': 'lbfgs', 'activation_val': 'identity',
        'max_iter_val': 1000, 'expected': 'Cannot Train Model: None'},
    )
    def test_train_nn_model(self, model_type, solver_val, activation_val, max_iter_val, expected):
        # Create new Model
        model_params = {}
        model_params['solver_val'] = solver_val
        model_params['activation_val'] =  activation_val
        model_params['max_iter_val'] = max_iter_val
        model = predict.make_model(model_type, model_params)

        return_str = predict.train_model(model, test_run=True)

        self.assertEqual(expected, return_str)

    @unpack
    @data(
        {'model_type': 'rf', 'n_estimators_val': 100, 'max_depth_val': None,
        'expected': 'RandomForestClassifier Model saved into Models Directory'},
        {'model_type': 'RF', 'n_estimators_val': 100, 'max_depth_val': None,
        'expected': 'RandomForestClassifier Model saved into Models Directory'},
        {'model_type': 'rf', 'n_estimators_val': 50, 'max_depth_val': None,
        'expected': 'RandomForestClassifier Model saved into Models Directory'},
        {'model_type': 'RF', 'n_estimators_val': 50, 'max_depth_val': None,
        'expected': 'RandomForestClassifier Model saved into Models Directory'},

        {'model_type': 'invalid', 'n_estimators_val': 50, 'max_depth_val': None,
        'expected': 'Cannot Train Model: None'},
    )
    def test_train_rf_model(self, model_type, n_estimators_val, max_depth_val, expected):
        # Create new Model
        model_params = {}
        model_params['n_estimators_val'] = n_estimators_val
        model_params['max_depth_val'] = max_depth_val
        model = predict.make_model(model_type, model_params)

        return_str = predict.train_model(model, test_run=True)

        self.assertEqual(expected, return_str)

    @unpack
    @data(
        {'model_type': 'knn', 'n_neighbors_val': 3, 'algorithm_val': 'auto',
        'expected': 'KNeighborsClassifier Model saved into Models Directory'},
        {'model_type': 'knn', 'n_neighbors_val': 3, 'algorithm_val': 'kd_tree',
        'expected': 'KNeighborsClassifier Model saved into Models Directory'},
        {'model_type': 'knn', 'n_neighbors_val': 3, 'algorithm_val': 'ball_tree',
        'expected': 'KNeighborsClassifier Model saved into Models Directory'},
        {'model_type': 'knn', 'n_neighbors_val': 3, 'algorithm_val': 'brute',
        'expected': 'KNeighborsClassifier Model saved into Models Directory'},
    )
    def test_train_knn_model(self, model_type, n_neighbors_val, algorithm_val, expected):
        # Create new Model
        model_params = {}
        model_params['n_neighbors_val'] = n_neighbors_val
        model_params['algorithm_val'] = algorithm_val
        model = predict.make_model(model_type, model_params)

        return_str = predict.train_model(model, test_run=True)

        self.assertEqual(expected, return_str)

@ddt
class TestPredict(unittest.TestCase):
    @unpack
    @data(
    {'model_fpath': 'Models/test_model.pkl', 'chosen_samples': ['Samples/E_1000001.wav'],
    'expected': '[\'[E_1000001.wav] is [Spoof] Voice\']'},
    {'model_fpath': 'Models/test_model.pkl', 'chosen_samples': ['Samples/E_1000002.wav'],
    'expected': '[\'[E_1000002.wav] is [Spoof] Voice\']'},
    {'model_fpath': 'Models/test_model.pkl', 'chosen_samples': ['Samples/E_1000010.wav'],
    'expected': '[\'[E_1000010.wav] is [Genuine] Voice\']'},
    {'model_fpath': 'Models/test_model.pkl', 'chosen_samples': ['Samples/E_1000020.wav'],
    'expected': '[\'[E_1000020.wav] is [Spoof] Voice\']'},
    {'model_fpath': 'Models/test_model.pkl', 'chosen_samples': ['Samples/E_1000010.wav', 'Samples/E_1000020.wav'],
    'expected': '[\'[E_1000010.wav] is [Genuine] Voice\', \'[E_1000020.wav] is [Spoof] Voice\']'},
    {'model_fpath': 'Models/test_model.pkl', 'chosen_samples': ['Samples/E_1000040.wav', 'Samples/E_1000050.wav'],
    'expected': '[\'[E_1000040.wav] is [Spoof] Voice\', \'[E_1000050.wav] is [Genuine] Voice\']'},
    )
    def test_prediction_pipeline(self, model_fpath, chosen_samples, expected):
        predictions = predict.predict_pipeline(model_fpath, chosen_samples)
        self.assertEqual(expected, f'{predictions}')

if __name__ == '__main__':
    # Setting Buffer = True suppresses the print statements from functions
    unittest.main(buffer=True)
