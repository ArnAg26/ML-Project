import logging
from warnings import warn
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from matplotlib import pyplot
import logging
from time import time

def log_execution(func):
    def wrapped(*args, **kwargs):
        logging.debug('Executing %s...', func.__name__)
        start = time()
        result = func(*args, **kwargs)
        end = time()
        logging.debug('Completed %s (%.3fs)', func.__name__, end - start)
        return result
    return wrapped
@log_execution
def train(feature_images, truth_images, probability):
    feature_images = np.array(feature_images) 
    flat_image = feature_images.reshape(-1, feature_images.shape[-1])
    flat_truth = np.ravel(truth_images)
    base_estimator = SVC(gamma='auto', probability=probability)
    num_estimators = feature_images.shape[0]
    num_samples = np.prod(feature_images.shape[1:3])
    print("Features done")
    model = BaggingClassifier(base_estimator, n_estimators=2, max_samples=num_samples)
    model.fit(flat_image, flat_truth) 
    print("Fit done")

    pickle.dump(model, open('models/model2.p', 'wb'))

@log_execution
def classify(feature_image):
    model = pickle.load(open('models/model2.p', 'rb'))
    shape = feature_image.shape[:2]
    flat_image = feature_image.reshape(-1, feature_image.shape[-1])
    probabilities = model.predict_proba(flat_image) 
    probabilities = probabilities[:, 1].reshape(shape) 
    prediction = np.where(probabilities >= 0.5, True, False) 

    return probabilities, prediction

@log_execution
def assess(truth, prediction):
    true_positive = np.count_nonzero(np.logical_and(truth, prediction))
    true_negative = np.count_nonzero(np.logical_and(~truth, ~prediction))
    false_positive = np.count_nonzero(np.logical_and(~truth, prediction))
    false_negative = np.count_nonzero(np.logical_and(truth, ~prediction))
    precision = true_positive / np.count_nonzero(prediction)
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive +
                                                  false_negative)
    logging.info('The Models Precision: %f', precision)
    logging.info('The Models Recall/Sensitivity: %f', sensitivity)
    logging.info('The Models Specificity: %f', specificity)
    logging.info('The Models Accuracy: %f', accuracy)

