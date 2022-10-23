from unittest import TestCase
from unittest import mock
from clustintime.clustintime import validate_data_file


class TestInputValidation(TestCase):

    def test_not_a_str(self):
        try:
            validate_data_file(data_file=1)
        except Exception as ex:
            assert isinstance(ex, ValueError)
            assert ex.args[0] == 'data_file should be str.'

    def test_non_existent(self):
        try:
            validate_data_file(data_file='file_not_exist')
        except Exception as ex:
            assert isinstance(ex, ValueError)
            assert ex.args[0] == 'data_file does not exist.'

    def test_not_a_file(self):
        mock_exists = mock.patch('clustintime.clustintime.exists').start()
        mock_exists.return_value = True

        mock_path = mock.patch('clustintime.clustintime.Path').start()
        mock_path.return_value.is_file.return_value = False

        expected_exception = None

        try:
            validate_data_file(data_file='file_not_exist')
        except Exception as ex:
            expected_exception = ex

        assert isinstance(expected_exception, ValueError)
        assert expected_exception.args[0] == 'data_file is not a file.'


