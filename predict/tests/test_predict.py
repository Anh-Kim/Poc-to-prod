import unittest
from unittest.mock import MagicMock
import tempfile

from train.train import run
from predict.predict.run import TextPredictionModel
from preprocessing.preprocessing import utils
from train.tests import test_model_train


class TestPredict(unittest.TestCase):

    def test_predict(self):
        # create a dictionary params for train conf
        params = {
            'batch_size': 1,
            'epochs': 5,
            'dense_dim': 64,
            'min_samples_per_label': 1,
            'verbose': 1
        }
        utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=test_model_train.load_dataset_mock())

        # we create a temporary file to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:
            # run a training
            accuracy, artefacts_path = run.train('fake_dataset_path',
                                                 params,
                                                 r"C:\Users\anhki\Documents\EPF\5A\From Poc to Prod\poc-to-prod-capstone\poc-to-prod-capstone\train\data\artefacts\2023-01-06-11-39-27",
                                                 True)

        model = TextPredictionModel.from_artefacts(artefacts_path)
        prediction = model.predict(["Are there any automocking libraries in the Java world?"],1)

        self.assertEqual([['php']], prediction)