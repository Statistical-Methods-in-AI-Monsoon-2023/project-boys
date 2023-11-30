import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from tqdm import tqdm
from collections import Counter

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, max_features=None, class_weights='None', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.class_weights = class_weights
        self.random_state = random_state
        self.estimators = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        if self.class_weights == 'balanced_subsample':
            class_sample_weights = compute_class_sample_weights(y)
        else:
            class_sample_weights = None

        for _ in tqdm(range(self.n_estimators), desc="Fitting RFC"):
            feature_indices = np.random.choice(n_features, size=int(0.7*n_features), replace=False) if self.max_features else None
            row_indices = np.random.choice(n_samples, size=int(0.6*n_samples), replace=True)
            X_bootstrap, y_bootstrap = X[row_indices, feature_indices], y[row_indices]
            X_bootstrap = np.squeeze(X_bootstrap)

            if self.class_weights == 'balanced_subsample':
                tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            else:
                tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)

            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators.append((tree, feature_indices))

    def predict(self, X):
        if not self.estimators:
            raise ValueError("Random Forest is not fitted. Call the fit method first.")

        class_predictions = np.zeros((X.shape[0], len(self.classes_)))

        for estimator, feature_indices in self.estimators:
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