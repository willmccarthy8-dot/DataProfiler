"""Tests for JSON serialization/deserialization edge cases.

Covers ProfileEncoder and json_decoder.
"""

import json
import unittest
import warnings
from datetime import datetime
from unittest import mock

import numpy as np
import pandas as pd

from dataprofiler.profilers.json_decoder import (
    get_column_profiler_class,
    get_compiler_class,
    get_option_class,
    get_profiler_class,
    get_structured_col_profiler_class,
    load_column_profile,
    load_compiler,
    load_option,
    load_profiler,
    load_structured_col_profiler,
)
from dataprofiler.profilers.json_encoder import ProfileEncoder


class TestProfileEncoderUnsupportedTypes(unittest.TestCase):

    def test_unsupported_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            json.dumps(object(), cls=ProfileEncoder)

    def test_unsupported_plain_int_raises_type_error(self):
        encoder = ProfileEncoder()
        with self.assertRaises(TypeError):
            encoder.default(object())

    def test_bytes_raises_type_error(self):
        with self.assertRaises(TypeError):
            json.dumps({"val": b"raw bytes"}, cls=ProfileEncoder)


class TestProfileEncoderUnstructuredProfiler(unittest.TestCase):

    def test_unstructured_profiler_raises_not_implemented(self):
        from dataprofiler.profilers.profile_builder import UnstructuredProfiler

        mock_profiler = mock.MagicMock(spec=UnstructuredProfiler)
        mock_profiler.__class__ = UnstructuredProfiler
        encoder = ProfileEncoder()
        with self.assertRaises(NotImplementedError) as ctx:
            encoder.default(mock_profiler)
        self.assertIn(
            "UnstructuredProfiler serialization not supported",
            str(ctx.exception),
        )


class TestProfileEncoderSetSerialization(unittest.TestCase):

    def test_set_serialized_to_list(self):
        encoder = ProfileEncoder()
        result = encoder.default({1, 2, 3})
        self.assertIsInstance(result, list)
        self.assertCountEqual(result, [1, 2, 3])

    def test_empty_set_serialized_to_empty_list(self):
        encoder = ProfileEncoder()
        result = encoder.default(set())
        self.assertEqual(result, [])

    def test_set_in_json_dumps(self):
        from dataprofiler.profilers.profiler_options import BooleanOption

        option = BooleanOption()
        option.test_attr = {"a", "b", "c"}
        serialized = json.dumps(option, cls=ProfileEncoder)
        deserialized = json.loads(serialized)
        self.assertIsInstance(deserialized["data"]["test_attr"], list)
        self.assertCountEqual(deserialized["data"]["test_attr"], ["a", "b", "c"])


class TestProfileEncoderNumpyTypes(unittest.TestCase):

    def test_numpy_integer_serialized_to_int(self):
        encoder = ProfileEncoder()
        for np_type in [np.int8, np.int16, np.int32, np.int64]:
            val = np_type(42)
            result = encoder.default(val)
            self.assertIsInstance(result, int)
            self.assertEqual(result, 42)

    def test_numpy_array_serialized_to_list(self):
        encoder = ProfileEncoder()
        arr = np.array([1, 2, 3, 4, 5])
        result = encoder.default(arr)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1, 2, 3, 4, 5])

    def test_numpy_2d_array_serialized_to_nested_list(self):
        encoder = ProfileEncoder()
        arr = np.array([[1, 2], [3, 4]])
        result = encoder.default(arr)
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_numpy_empty_array(self):
        encoder = ProfileEncoder()
        arr = np.array([])
        result = encoder.default(arr)
        self.assertEqual(result, [])

    def test_numpy_float_array(self):
        encoder = ProfileEncoder()
        arr = np.array([1.5, 2.5, 3.5])
        result = encoder.default(arr)
        self.assertEqual(result, [1.5, 2.5, 3.5])


class TestProfileEncoderDatetimeTypes(unittest.TestCase):

    def test_datetime_serialized_to_isoformat(self):
        encoder = ProfileEncoder()
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = encoder.default(dt)
        self.assertEqual(result, "2024-01-15T10:30:00")

    def test_pandas_timestamp_serialized_to_isoformat(self):
        encoder = ProfileEncoder()
        ts = pd.Timestamp("2024-06-15 14:30:00")
        result = encoder.default(ts)
        self.assertEqual(result, "2024-06-15T14:30:00")

    def test_datetime_round_trip_via_json(self):
        dt = datetime(2024, 3, 20, 8, 45, 30)
        encoded = json.dumps(dt, cls=ProfileEncoder)
        decoded = json.loads(encoded)
        restored = datetime.fromisoformat(decoded)
        self.assertEqual(restored, dt)


class TestProfileEncoderCallable(unittest.TestCase):

    def test_callable_function_serialized_to_name(self):
        encoder = ProfileEncoder()

        def my_func():
            pass

        result = encoder.default(my_func)
        self.assertEqual(result, "my_func")

    def test_builtin_callable_serialized(self):
        encoder = ProfileEncoder()
        result = encoder.default(len)
        self.assertEqual(result, "len")


class TestProfileEncoderDataLabeler(unittest.TestCase):

    def test_labeler_without_model_loc_raises_value_error(self):
        from dataprofiler.labelers.base_data_labeler import BaseDataLabeler

        mock_labeler = mock.MagicMock(spec=BaseDataLabeler)
        mock_labeler.__class__ = BaseDataLabeler
        mock_labeler._default_model_loc = None
        encoder = ProfileEncoder()
        with self.assertRaises(ValueError) as ctx:
            encoder.default(mock_labeler)
        self.assertIn("_default_model_loc not set", str(ctx.exception))

    def test_labeler_with_model_loc_returns_dict(self):
        from dataprofiler.labelers.base_data_labeler import BaseDataLabeler

        mock_labeler = mock.MagicMock(spec=BaseDataLabeler)
        mock_labeler.__class__ = BaseDataLabeler
        mock_labeler._default_model_loc = "structured_model"
        encoder = ProfileEncoder()
        result = encoder.default(mock_labeler)
        self.assertEqual(result, {"from_library": "structured_model"})


class TestProfileEncoderProfilerObjects(unittest.TestCase):

    def test_base_option_serialized_with_class_and_data(self):
        from dataprofiler.profilers.profiler_options import BooleanOption

        option = BooleanOption(is_enabled=True)
        serialized = json.dumps(option, cls=ProfileEncoder)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized["class"], "BooleanOption")
        self.assertIn("data", deserialized)
        self.assertIsInstance(deserialized["data"], dict)

    def test_base_column_profiler_serialized(self):
        from dataprofiler.profilers.base_column_profilers import BaseColumnProfiler

        with mock.patch.multiple(BaseColumnProfiler, __abstractmethods__=set()):
            profiler = BaseColumnProfiler(name="test_col")
        serialized = json.dumps(profiler, cls=ProfileEncoder)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized["class"], "BaseColumnProfiler")
        self.assertEqual(deserialized["data"]["name"], "test_col")

    def test_compiler_serialized(self):
        from dataprofiler.profilers.column_profile_compilers import (
            ColumnStatsProfileCompiler,
        )

        compiler = ColumnStatsProfileCompiler()
        serialized = json.dumps(compiler, cls=ProfileEncoder)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized["class"], "ColumnStatsProfileCompiler")
        self.assertIn("data", deserialized)


class TestDecoderGetClassInvalidNames(unittest.TestCase):

    def test_get_column_profiler_class_invalid_name(self):
        with self.assertRaises(ValueError) as ctx:
            get_column_profiler_class("NonExistentProfiler")
        self.assertIn("Invalid profiler class NonExistentProfiler", str(ctx.exception))

    def test_get_compiler_class_invalid_name(self):
        with self.assertRaises(ValueError) as ctx:
            get_compiler_class("NonExistentCompiler")
        self.assertIn("Invalid compiler class NonExistentCompiler", str(ctx.exception))

    def test_get_option_class_invalid_name(self):
        with self.assertRaises(ValueError) as ctx:
            get_option_class("NonExistentOption")
        self.assertIn("Invalid option class NonExistentOption", str(ctx.exception))

    def test_get_profiler_class_invalid_name(self):
        with self.assertRaises(ValueError) as ctx:
            get_profiler_class("NonExistentProfiler")
        self.assertIn("Invalid profiler class NonExistentProfiler", str(ctx.exception))

    def test_get_structured_col_profiler_class_invalid_name(self):
        with self.assertRaises(ValueError) as ctx:
            get_structured_col_profiler_class("NonExistentProfiler")
        self.assertIn(
            "Invalid structured col profiler class NonExistentProfiler",
            str(ctx.exception),
        )


class TestDecoderHistogramOptionDeprecation(unittest.TestCase):

    def test_histogram_option_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cls = get_option_class("HistogramOption")
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertIn("HistogramOption", str(w[0].message))
        from dataprofiler.profilers.profiler_options import (
            HistogramAndQuantilesOption,
        )

        self.assertIs(cls, HistogramAndQuantilesOption)


class TestDecoderMalformedInput(unittest.TestCase):

    def test_load_column_profile_missing_class_key(self):
        with self.assertRaises(KeyError):
            load_column_profile({"data": {}})

    def test_load_column_profile_invalid_class(self):
        with self.assertRaises(ValueError):
            load_column_profile({"class": "BogusClass", "data": {}})

    def test_load_compiler_missing_class_key(self):
        with self.assertRaises(KeyError):
            load_compiler({"data": {}})

    def test_load_compiler_invalid_class(self):
        with self.assertRaises(ValueError):
            load_compiler({"class": "BogusCompiler", "data": {}})

    def test_load_option_missing_class_key(self):
        with self.assertRaises(KeyError):
            load_option({"data": {}})

    def test_load_option_invalid_class(self):
        with self.assertRaises(ValueError):
            load_option({"class": "BogusOption", "data": {}})

    def test_load_profiler_missing_class_key(self):
        with self.assertRaises(KeyError):
            load_profiler({"data": {}})

    def test_load_profiler_invalid_class(self):
        with self.assertRaises(ValueError):
            load_profiler({"class": "BogusProfiler", "data": {}})

    def test_load_structured_col_profiler_missing_class_key(self):
        with self.assertRaises(KeyError):
            load_structured_col_profiler({"data": {}})

    def test_load_structured_col_profiler_invalid_class(self):
        with self.assertRaises(ValueError):
            load_structured_col_profiler({"class": "BogusProfiler", "data": {}})

    def test_load_column_profile_empty_dict(self):
        with self.assertRaises(KeyError):
            load_column_profile({})

    def test_load_profiler_with_none_input(self):
        with self.assertRaises((KeyError, TypeError)):
            load_profiler(None)


class TestRoundTripEncodeDecode(unittest.TestCase):

    def test_round_trip_boolean_option(self):
        from dataprofiler.profilers.profiler_options import BooleanOption

        original = BooleanOption(is_enabled=True)
        serialized = json.dumps(original, cls=ProfileEncoder)
        deserialized_dict = json.loads(serialized)
        restored = load_option(deserialized_dict)
        self.assertIsInstance(restored, BooleanOption)
        self.assertEqual(restored.is_enabled, original.is_enabled)

    def test_round_trip_histogram_and_quantiles_option(self):
        from dataprofiler.profilers.profiler_options import (
            HistogramAndQuantilesOption,
        )

        original = HistogramAndQuantilesOption()
        original.is_enabled = True
        original.bin_count_or_method = "auto"
        serialized = json.dumps(original, cls=ProfileEncoder)
        deserialized_dict = json.loads(serialized)
        restored = load_option(deserialized_dict)
        self.assertIsInstance(restored, HistogramAndQuantilesOption)
        self.assertEqual(restored.is_enabled, original.is_enabled)
        self.assertEqual(restored.bin_count_or_method, original.bin_count_or_method)

    def test_round_trip_column_stats_compiler(self):
        from dataprofiler.profilers.column_profile_compilers import (
            ColumnStatsProfileCompiler,
        )

        from . import utils as test_utils

        original = ColumnStatsProfileCompiler()
        serialized = json.dumps(original, cls=ProfileEncoder)
        deserialized_dict = json.loads(serialized)
        restored = load_compiler(deserialized_dict)
        test_utils.assert_profiles_equal(original, restored)

    def test_round_trip_primitive_type_compiler(self):
        from dataprofiler.profilers.column_profile_compilers import (
            ColumnPrimitiveTypeProfileCompiler,
        )

        from . import utils as test_utils

        original = ColumnPrimitiveTypeProfileCompiler()
        serialized = json.dumps(original, cls=ProfileEncoder)
        deserialized_dict = json.loads(serialized)
        restored = load_compiler(deserialized_dict)
        test_utils.assert_profiles_equal(original, restored)

    def test_round_trip_nested_option_structure(self):
        from dataprofiler.profilers.profiler_options import NumericalOptions

        original = NumericalOptions()
        original.is_enabled = True
        serialized = json.dumps(original, cls=ProfileEncoder)
        deserialized_dict = json.loads(serialized)
        restored = load_option(deserialized_dict)
        self.assertIsInstance(restored, NumericalOptions)
        self.assertEqual(restored.is_enabled, original.is_enabled)

    def test_round_trip_preserves_numpy_array_data(self):
        from dataprofiler.profilers.profiler_options import BooleanOption

        option = BooleanOption(is_enabled=True)
        option.test_array = np.array([10, 20, 30])
        serialized = json.dumps(option, cls=ProfileEncoder)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized["data"]["test_array"], [10, 20, 30])

    def test_round_trip_preserves_datetime_data(self):
        from dataprofiler.profilers.profiler_options import BooleanOption

        option = BooleanOption(is_enabled=True)
        option.test_dt = datetime(2024, 6, 15, 12, 0, 0)
        serialized = json.dumps(option, cls=ProfileEncoder)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized["data"]["test_dt"], "2024-06-15T12:00:00")


class TestDecoderLoadProfiler(unittest.TestCase):

    def test_load_profiler_valid_structured(self):
        from dataprofiler.profilers.profile_builder import StructuredProfiler

        with mock.patch.object(
            StructuredProfiler, "load_from_dict", return_value="mocked"
        ) as mock_load:
            result = load_profiler(
                {"class": "StructuredProfiler", "data": {"key": "value"}}
            )
        mock_load.assert_called_once_with({"key": "value"}, None)
        self.assertEqual(result, "mocked")

    def test_load_structured_col_profiler_valid(self):
        from dataprofiler.profilers.profile_builder import StructuredColProfiler

        with mock.patch.object(
            StructuredColProfiler, "load_from_dict", return_value="mocked"
        ) as mock_load:
            result = load_structured_col_profiler(
                {"class": "StructuredColProfiler", "data": {"key": "value"}}
            )
        mock_load.assert_called_once_with({"key": "value"}, None)
        self.assertEqual(result, "mocked")


if __name__ == "__main__":
    unittest.main()
