from sklearn.datasets import load_iris


def test_iris_dataset_shape():
    iris = load_iris()
    assert iris.data.shape == (150, 4)
    assert len(set(iris.target)) == 3


def test_expected_features_exist():
    iris = load_iris(as_frame=True)
    assert len(iris.feature_names) == 4
    assert all(name in iris.data.columns for name in iris.feature_names)
