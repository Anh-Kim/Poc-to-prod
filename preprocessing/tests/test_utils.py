import unittest
import pandas as pd
import brotli
from unittest.mock import MagicMock

from preprocessing.preprocessing import utils


class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        # TODO: CODE HERE
        # integer_floor((self._get_num_train_samples()) / (self.batch_size))
        # batch_size = 20
        # get_num_train_samples : integer_floor(self._get_num_samples() * self.train_ratio)
        # get_num_samples : NotImplementedError : 100
        # We mock get_num_train_samples
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # 100 * 0.8 = 125
        base._get_num_train_samples = MagicMock(return_value=125)
        # 125 / 20
        self.assertEqual(base._get_num_train_batches(), 6)
        # pass

    def test__get_num_test_batches(self):
        # TODO: CODE
        # integer_floor(self._get_num_test_samples / self.batch_size)
        # get_num_test_samples: integer_floor(self._get_num_train_samples())*(1-self.train_ratio)
        # get_num_train_samples : integer_floor(self._get_num_samples() * self.train_ratio)
        # get_num_samples
        # batch_size = 20 and train_ratio = 0.8
        # We mock get_num_train_samples
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # 100
        base._get_num_samples = MagicMock(return_value=100)
        #
        self.assertEqual(base._get_num_test_batches(), 1)
        # pass

    def test_get_index_to_label_map(self):
        # TODO: CODE HERE
        # dict(map(enumerate(self._get_label_list)))
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['a', 'b', 'c'])
        self.assertEqual(base.get_index_to_label_map(), {0: 'a', 1: 'b', 2: 'c'})
        # pass

    def test_index_to_label_and_label_to_index_are_identity(self):
        # TODO: CODE HERE

        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value={0: 'a', 1: 'b', 2: 'c'})
        self.assertEqual(base.get_label_to_index_map(), {0: 0, 1: 1, 2: 2})
        # pass


    def test_to_indexes(self):
        # TODO: CODE HERE
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['a', 'b', 'c'])
        base.get_index_to_label_map()
        self.assertEqual(base.to_indexes(['a', 'c']), [0,2])
        # pass


class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path",1)
        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })
        # we confirm that the dataset and what we expected to be are the same thing
        print(dataset)
        pd.testing.assert_frame_equal(dataset, expected)

    def test__get_num_samples_is_correct(self):
        #len(self._dataset)
        # TODO: CODE HERE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6'],
            'tag_name': ['tag_a', 'tag_b', 'tag_c', 'tag_a', 'tag_b', 'tag_c'],
            'tag_id': [1, 2, 3, 1, 2, 3],
            'tag_position': [0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6']
        }))
        dataset = utils.LocalTextCategorizationDataset("fake_path", batch_size=1, train_ratio=0.6, min_samples_per_label=2)

        self.assertEqual(dataset._get_num_samples(), 6)
        # pass

    def test_get_train_batch_returns_expected_shape(self):
        # TODO: CODE HERE
        # create mock datatset
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6'],
            'tag_name': ['tag_a', 'tag_b', 'tag_c', 'tag_a', 'tag_b', 'tag_c'],
            'tag_id': [1, 2, 3, 1, 2, 3],
            'tag_position': [0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6']
        }))

        dataset = utils.LocalTextCategorizationDataset("fake_path", batch_size=1, train_ratio=0.5, min_samples_per_label=1)
        x, y = dataset.get_train_batch()
        self.assertTupleEqual(x.shape, (1,)) and self.assertTupleEqual(y.shape, (1, 5))
        # pass

    def test_get_test_batch_returns_expected_shape(self):
        # TODO: CODE HERE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6'],
            'tag_name': ['tag_a', 'tag_b', 'tag_c', 'tag_a', 'tag_b', 'tag_c'],
            'tag_id': [1, 2, 3, 1, 2, 3],
            'tag_position': [0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6']
        }))
        dataset = utils.LocalTextCategorizationDataset("fake_path", batch_size=1, train_ratio=0.5,
                                                       min_samples_per_label=1)

        x, y = dataset.get_test_batch()
        self.assertTupleEqual(x.shape, (1,)) and self.assertTupleEqual(y.shape, (1, 5))
    # pass

    def test_get_train_batch_raises_assertion_error(self):
        # TODO: CODE HERE
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_a'],
            'tag_id': [1, 2],
            'tag_position': [0, 0],
            'title': ['title_1', 'title_2']
        }))
        with self.assertRaises(AssertionError):
            utils.LocalTextCategorizationDataset("fake_path", batch_size=3, train_ratio=0.5, min_samples_per_label=1)
    # pass
