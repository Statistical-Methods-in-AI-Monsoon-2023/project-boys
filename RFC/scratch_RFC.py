import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from tqdm import tqdm

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from tqdm import tqdm
from collections import Counter

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, class_weights='balanced_subsample', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weights = class_weights
        self.random_state = random_state
        self.estimators = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)

        if self.class_weights == 'balanced_subsample':
            class_sample_weights = compute_class_sample_weights(y)
        else:
            class_sample_weights = None

        for _ in tqdm(range(self.n_estimators), desc="Fitting RFC"):
            X_bootstrap, y_bootstrap = resample(X, y, replace=True, random_state=self.random_state, stratify=y)

            if self.class_weights == 'balanced_subsample':
                tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            else:
                tree = DecisionTreeClassifier(max_depth=self.max_depth, class_weight=self.class_weights, random_state=self.random_state)

            tree.fit(X_bootstrap, y_bootstrap, sample_weight=class_sample_weights)
            self.estimators.append(tree)

    def predict(self, X):
        if not self.estimators:
            raise ValueError("Random Forest is not fitted. Call the fit method first.")

        class_predictions = np.zeros((X.shape[0], len(self.classes_)))

        for estimator in self.estimators:
            estimator_predictions = estimator.predict(X)
            for i, class_label in enumerate(self.classes_):
                class_indices = (estimator_predictions == class_label)
                class_predictions[:, i] += class_indices

        final_predictions = self.classes_[np.argmax(class_predictions, axis=1)]

        return final_predictions

def compute_class_sample_weights(y):
    class_weights = {}
    class_counts = Counter(y)
    majority_class = max(class_counts, key=class_counts.get)

    for class_label, count in class_counts.items():
        weight = len(y) / (len(class_counts) * count)
        class_weights[class_label] = weight / class_counts[majority_class]

    sample_weights = [class_weights[label] for label in y]

    return sample_weights