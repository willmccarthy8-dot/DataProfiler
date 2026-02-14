"""Tests for backward compatibility of profile serialization/deserialization."""
import json
import pickle
import unittest
import warnings
from collections import defaultdict
from io import BytesIO, StringIO
from unittest import mock

import numpy as np
import pandas as pd

import dataprofiler as dp
from dataprofiler.labelers.base_data_labeler import BaseDataLabeler
from dataprofiler.profilers.column_profile_compilers import (
    ColumnPrimitiveTypeProfileCompiler,
    ColumnStatsProfileCompiler,
)
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
from dataprofiler.profilers.profile_builder import (
    BaseProfiler,
    Profiler,
    StructuredColProfiler,
    StructuredProfiler,
    UnstructuredProfiler,
)
from dataprofiler.profilers.profiler_options import (
    BooleanOption,
    HistogramAndQuantilesOption,
    ProfilerOptions,
    StructuredOptions,
    UnstructuredOptions,
)

from . import utils as test_utils


def setup_save_mock_bytes_open(mock_open):
    mock_file = BytesIO()
    mock_file.close = lambda: None
    mock_open.side_effect = lambda *args, **kwargs: mock_file
    return mock_file


def setup_save_mock_string_open(mock_open):
    mock_file = StringIO()
    mock_file.close = lambda: None
    mock_open.side_effect = lambda *args, **kwargs: mock_file
    return mock_file


class TestPickleBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility for pickle-based save/load."""

    def test_load_deprecated_format_no_profiler_class_structured(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        save_profile = dp.StructuredProfiler(data, options=profile_options)

        with mock.patch("builtins.open") as m:
            mock_file = setup_save_mock_bytes_open(m)
            save_profile.save()
            mock_file.seek(0)
            data_dict = pickle.load(mock_file)

        data_dict.pop("profiler_class", None)

        mock_file_no_class = BytesIO()
        pickle.dump(data_dict, mock_file_no_class)
        mock_file_no_class.seek(0)

        with mock.patch("builtins.open") as m:
            m.return_value.__enter__ = mock.Mock(return_value=mock_file_no_class)
            m.return_value.__exit__ = mock.Mock(return_value=False)
            mock_file_no_class.close = lambda: None
            m.side_effect = lambda *args, **kwargs: mock_file_no_class

            with mock.patch("dataprofiler.profilers.profile_builder.DataLabeler"):
                loaded = BaseProfiler.load("mock.pkl", "pickle")

        self.assertIsInstance(loaded, StructuredProfiler)

    def test_load_deprecated_format_no_profiler_class_unstructured(self):
        data = "This is some test text data for profiling"
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        save_profile = dp.UnstructuredProfiler(data, options=profile_options)

        with mock.patch("builtins.open") as m:
            mock_file = setup_save_mock_bytes_open(m)
            save_profile.save()
            mock_file.seek(0)
            data_dict = pickle.load(mock_file)

        data_dict.pop("profiler_class", None)

        mock_file_no_class = BytesIO()
        pickle.dump(data_dict, mock_file_no_class)
        mock_file_no_class.seek(0)

        with mock.patch("builtins.open") as m:
            mock_file_no_class.close = lambda: None
            m.side_effect = lambda *args, **kwargs: mock_file_no_class

            with mock.patch("dataprofiler.profilers.profile_builder.DataLabeler"):
                loaded = BaseProfiler.load("mock.pkl", "pickle")

        self.assertIsInstance(loaded, UnstructuredProfiler)

    def test_load_invalid_profiler_class_raises_error(self):
        data_dict = {
            "profiler_class": "NonExistentProfiler",
            "total_samples": 0,
        }

        mock_file = BytesIO()
        pickle.dump(data_dict, mock_file)
        mock_file.seek(0)

        with mock.patch("builtins.open") as m:
            mock_file.close = lambda: None
            m.side_effect = lambda *args, **kwargs: mock_file

            with self.assertRaisesRegex(
                ValueError, "Invalid profiler class NonExistentProfiler"
            ):
                BaseProfiler.load("mock.pkl", "pickle")

    def test_load_invalid_load_method_raises_error(self):
        with self.assertRaisesRegex(
            ValueError,
            "Please specify a valid load_method",
        ):
            BaseProfiler.load("mock.pkl", "csv")

    def test_load_corrupt_pickle_with_explicit_pickle_method(self):
        mock_file = BytesIO(b"this is not a valid pickle")
        mock_file.close = lambda: None

        with mock.patch("builtins.open") as m:
            m.side_effect = lambda *args, **kwargs: mock_file

            with self.assertRaises(Exception):
                BaseProfiler.load("mock.pkl", "pickle")

    def test_structured_save_load_round_trip_preserves_data(self):
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        save_profile = dp.StructuredProfiler(data, options=profile_options)

        with mock.patch("builtins.open") as m:
            mock_file = setup_save_mock_bytes_open(m)
            save_profile.save()
            mock_file.seek(0)

            with mock.patch("dataprofiler.profilers.profile_builder.DataLabeler"):
                load_profile = dp.StructuredProfiler.load("mock.pkl", "pickle")

        save_report = test_utils.clean_report(save_profile.report())
        load_report = test_utils.clean_report(load_profile.report())
        np.testing.assert_equal(save_report, load_report)

    def test_unstructured_save_load_round_trip_preserves_data(self):
        data = "This is test text for the unstructured profiler"
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        save_profile = dp.UnstructuredProfiler(data, options=profile_options)

        with mock.patch("builtins.open") as m:
            mock_file = setup_save_mock_bytes_open(m)
            save_profile.save()
            mock_file.seek(0)

            with mock.patch("dataprofiler.profilers.profile_builder.DataLabeler"):
                load_profile = dp.UnstructuredProfiler.load("mock.pkl", "pickle")

        save_report = save_profile.report()
        load_report = load_profile.report()
        self.assertEqual(
            save_report["global_stats"]["samples_used"],
            load_report["global_stats"]["samples_used"],
        )

    def test_loaded_profile_remains_updatable(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        save_profile = dp.StructuredProfiler(data, options=profile_options)

        with mock.patch("builtins.open") as m:
            mock_file = setup_save_mock_bytes_open(m)
            save_profile.save()
            mock_file.seek(0)

            with mock.patch("dataprofiler.profilers.profile_builder.DataLabeler"):
                load_profile = dp.StructuredProfiler.load("mock.pkl", "pickle")

        initial_total = load_profile.total_samples
        new_data = pd.DataFrame({"a": [4, 5, 6]})
        load_profile.update_profile(new_data)

        self.assertEqual(load_profile.total_samples, initial_total + 3)
        report = load_profile.report()
        self.assertIn("global_stats", report)
        self.assertIn("data_stats", report)

    def test_pickle_with_extra_attributes_in_data_dict(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        save_profile = dp.StructuredProfiler(data, options=profile_options)

        with mock.patch("builtins.open") as m:
            mock_file = setup_save_mock_bytes_open(m)
            save_profile.save()
            mock_file.seek(0)
            data_dict = pickle.load(mock_file)

        data_dict["future_attribute"] = "some_future_value"
        data_dict["another_new_field"] = 42

        mock_file_extra = BytesIO()
        pickle.dump(data_dict, mock_file_extra)
        mock_file_extra.seek(0)

        with mock.patch("builtins.open") as m:
            mock_file_extra.close = lambda: None
            m.side_effect = lambda *args, **kwargs: mock_file_extra

            with mock.patch("dataprofiler.profilers.profile_builder.DataLabeler"):
                loaded = BaseProfiler.load("mock.pkl", "pickle")

        self.assertIsInstance(loaded, StructuredProfiler)
        self.assertTrue(hasattr(loaded, "future_attribute"))
        self.assertEqual(loaded.future_attribute, "some_future_value")

    def test_profiler_load_delegates_to_base(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        save_profile = dp.StructuredProfiler(data, options=profile_options)

        with mock.patch("builtins.open") as m:
            mock_file = setup_save_mock_bytes_open(m)
            save_profile.save()
            mock_file.seek(0)

            with mock.patch("dataprofiler.profilers.profile_builder.DataLabeler"):
                load_profile = Profiler.load("mock.pkl", "pickle")

        self.assertIsInstance(load_profile, StructuredProfiler)

    def test_save_method_validation(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        profile = dp.StructuredProfiler(data, options=profile_options)

        with self.assertRaisesRegex(
            ValueError, 'save_method must be "json" or "pickle".'
        ):
            profile.save(save_method="csv")

    def test_unstructured_save_method_validation(self):
        data = "Test text"
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        profile = dp.UnstructuredProfiler(data, options=profile_options)

        with self.assertRaisesRegex(
            ValueError, 'save_method must be "json" or "pickle".'
        ):
            profile.save(save_method="xml")


class TestJSONBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility for JSON-based save/load."""

    def test_deprecated_histogram_option_class_name(self):
        with self.assertWarns(DeprecationWarning) as cm:
            option_cls = get_option_class("HistogramOption")
        self.assertIs(option_cls, HistogramAndQuantilesOption)
        self.assertIn("HistogramOption will be deprecated", str(cm.warning))

    def test_histogram_option_loads_as_histogram_and_quantiles(self):
        serialized = {
            "class": "HistogramOption",
            "data": {
                "is_enabled": True,
                "bin_count_or_method": "auto",
                "num_quantiles": 1000,
            },
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            loaded = load_option(serialized)
        self.assertIsInstance(loaded, HistogramAndQuantilesOption)
        self.assertTrue(loaded.is_enabled)

    def test_invalid_column_profiler_class_raises_error(self):
        with self.assertRaisesRegex(
            ValueError, "Invalid profiler class FakeColumnProfiler"
        ):
            get_column_profiler_class("FakeColumnProfiler")

    def test_invalid_compiler_class_raises_error(self):
        with self.assertRaisesRegex(
            ValueError, "Invalid compiler class FakeCompiler"
        ):
            get_compiler_class("FakeCompiler")

    def test_invalid_option_class_raises_error(self):
        with self.assertRaisesRegex(
            ValueError, "Invalid option class FakeOption"
        ):
            get_option_class("FakeOption")

    def test_invalid_profiler_class_raises_error(self):
        with self.assertRaisesRegex(
            ValueError, "Invalid profiler class FakeProfiler"
        ):
            get_profiler_class("FakeProfiler")

    def test_invalid_structured_col_profiler_class_raises_error(self):
        with self.assertRaisesRegex(
            ValueError, "Invalid structured col profiler class FakeStructuredCol"
        ):
            get_structured_col_profiler_class("FakeStructuredCol")

    def test_load_column_profile_invalid_class(self):
        serialized = {"class": "NonExistentColumn", "data": {}}
        with self.assertRaisesRegex(
            ValueError, "Invalid profiler class NonExistentColumn"
        ):
            load_column_profile(serialized)

    def test_load_compiler_invalid_class(self):
        serialized = {"class": "NonExistentCompiler", "data": {}}
        with self.assertRaisesRegex(
            ValueError, "Invalid compiler class NonExistentCompiler"
        ):
            load_compiler(serialized)

    def test_load_profiler_invalid_class(self):
        serialized = {"class": "NonExistentProfiler", "data": {}}
        with self.assertRaisesRegex(
            ValueError, "Invalid profiler class NonExistentProfiler"
        ):
            load_profiler(serialized)

    @mock.patch(
        "dataprofiler.profilers.data_labeler_column_profile.DataLabelerColumn.update"
    )
    @mock.patch(
        "dataprofiler.profilers.profile_builder.DataLabeler",
        spec=BaseDataLabeler,
    )
    def test_json_round_trip_structured(self, *mocks):
        mock_labeler = mocks[0].return_value
        mock_labeler._default_model_loc = "structured_model"
        mocks[0].load_from_library.return_value = mock_labeler

        data = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"multiprocess.is_enabled": False})
        save_profile = dp.StructuredProfiler(data, options=profile_options)

        with mock.patch("builtins.open") as m:
            mock_file = setup_save_mock_string_open(m)
            save_profile.save(save_method="json")
            mock_file.seek(0)

            with mock.patch(
                "dataprofiler.profilers.profiler_utils.DataLabeler.load_from_library",
                return_value=mock_labeler,
            ):
                load_profile = dp.StructuredProfiler.load("mock.json", "JSON")

        test_utils.assert_profiles_equal(save_profile, load_profile)

    def test_json_round_trip_structured_no_labeler(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        save_profile = dp.StructuredProfiler(data, options=profile_options)

        with mock.patch("builtins.open") as m:
            mock_file = setup_save_mock_string_open(m)
            save_profile.save(save_method="json")
            mock_file.seek(0)

            load_profile = dp.StructuredProfiler.load("mock.json", "json")

        test_utils.assert_profiles_equal(save_profile, load_profile)

    def test_json_load_with_none_method_fallback(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        save_profile = dp.StructuredProfiler(data, options=profile_options)

        with mock.patch("builtins.open") as m:
            mock_file_json = setup_save_mock_string_open(m)
            save_profile.save(save_method="json")
            json_content = mock_file_json.getvalue()

        mock_file_bytes = BytesIO(json_content.encode("utf-8"))
        mock_file_str = StringIO(json_content)
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_file_bytes
            return mock_file_str

        with mock.patch("builtins.open") as m:
            mock_file_bytes.close = lambda: None
            mock_file_str.close = lambda: None
            m.side_effect = side_effect
            load_profile = BaseProfiler.load("mock.json", load_method=None)

        self.assertIsInstance(load_profile, StructuredProfiler)

    def test_json_encoder_handles_numpy_types(self):
        encoder = ProfileEncoder()
        self.assertEqual(encoder.default(np.int64(42)), 42)
        np.testing.assert_array_equal(
            encoder.default(np.array([1, 2, 3])), [1, 2, 3]
        )

    def test_json_encoder_handles_sets(self):
        encoder = ProfileEncoder()
        result = encoder.default({1, 2, 3})
        self.assertIsInstance(result, list)
        self.assertEqual(sorted(result), [1, 2, 3])

    def test_json_encoder_handles_callables(self):
        encoder = ProfileEncoder()

        def my_func():
            pass

        result = encoder.default(my_func)
        self.assertEqual(result, "my_func")

    def test_json_encoder_raises_on_unserializable(self):
        encoder = ProfileEncoder()
        with self.assertRaises(TypeError):
            encoder.default(object())


class TestVersionCompatibilityWarningsAndErrors(unittest.TestCase):
    """Test warnings and errors for version incompatibility scenarios."""

    def test_merge_profiles_type_mismatch_raises_error(self):
        data_structured = pd.DataFrame({"a": [1, 2, 3]})
        data_unstructured = "Some text data"

        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})

        structured = dp.StructuredProfiler(data_structured, options=profile_options)
        unstructured = dp.UnstructuredProfiler(
            data_unstructured, options=profile_options
        )

        with self.assertRaisesRegex(
            TypeError,
            "`UnstructuredProfiler` and `StructuredProfiler` are not of the "
            "same profiler type.",
        ):
            unstructured + structured

    def test_merge_profiles_same_type_succeeds(self):
        data1 = pd.DataFrame({"a": [1, 2, 3]})
        data2 = pd.DataFrame({"a": [4, 5, 6]})

        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})

        profile1 = dp.StructuredProfiler(data1, options=profile_options)
        profile2 = dp.StructuredProfiler(data2, options=profile_options)

        merged = profile1 + profile2
        self.assertIsInstance(merged, StructuredProfiler)
        self.assertEqual(merged.total_samples, 6)

    def test_diff_type_mismatch_raises_error(self):
        data_structured = pd.DataFrame({"a": [1, 2, 3]})
        data_unstructured = "Some text data"

        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})

        structured = dp.StructuredProfiler(data_structured, options=profile_options)
        unstructured = dp.UnstructuredProfiler(
            data_unstructured, options=profile_options
        )

        with self.assertRaisesRegex(
            TypeError,
            "are not of the same profiler type",
        ):
            structured.diff(unstructured)

    def test_merge_calculation_warning_on_disabled_mismatch(self):
        data1 = pd.DataFrame({"a": [1, 2, 3]})
        data2 = pd.DataFrame({"a": [4, 5, 6]})

        options1 = dp.ProfilerOptions()
        options1.set({"data_labeler.is_enabled": False})
        profile1 = dp.StructuredProfiler(data1, options=options1)

        options2 = dp.ProfilerOptions()
        options2.set({
            "data_labeler.is_enabled": False,
            "structured_options.int.max.is_enabled": False,
        })
        profile2 = dp.StructuredProfiler(data2, options=options2)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            profile1 + profile2
            runtime_warnings = [
                x for x in w if issubclass(x.category, RuntimeWarning)
            ]
            disabled_warnings = [
                x
                for x in runtime_warnings
                if "disabled because it is not enabled in both profiles" in str(x.message)
            ]
            self.assertGreater(len(disabled_warnings), 0)

    def test_load_structured_preserves_profiler_class(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        save_profile = dp.StructuredProfiler(data, options=profile_options)

        with mock.patch("builtins.open") as m:
            mock_file = setup_save_mock_bytes_open(m)
            save_profile.save()
            mock_file.seek(0)
            data_dict = pickle.load(mock_file)

        self.assertEqual(data_dict["profiler_class"], "StructuredProfiler")

    def test_load_unstructured_preserves_profiler_class(self):
        data = "Test text for profiling"
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        save_profile = dp.UnstructuredProfiler(data, options=profile_options)

        with mock.patch("builtins.open") as m:
            mock_file = setup_save_mock_bytes_open(m)
            save_profile.save()
            mock_file.seek(0)
            data_dict = pickle.load(mock_file)

        self.assertEqual(data_dict["profiler_class"], "UnstructuredProfiler")


class TestSchemaBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility for schema changes across versions."""

    def test_structured_load_from_dict_handles_int_col_name_to_idx(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        profile = dp.StructuredProfiler(data, options=profile_options)

        serialized = json.loads(json.dumps(profile, cls=ProfileEncoder))
        profile_data = serialized["data"]

        profile_data["_col_name_to_idx"] = {"0": [0]}

        loaded = StructuredProfiler.load_from_dict(profile_data)
        self.assertIsInstance(loaded, StructuredProfiler)
        self.assertIn(0, loaded._col_name_to_idx)

    def test_structured_load_from_dict_handles_string_col_name_to_idx(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        profile = dp.StructuredProfiler(data, options=profile_options)

        serialized = json.loads(json.dumps(profile, cls=ProfileEncoder))
        profile_data = serialized["data"]

        profile_data["_col_name_to_idx"] = {"col_a": [0]}

        loaded = StructuredProfiler.load_from_dict(profile_data)
        self.assertIsInstance(loaded, StructuredProfiler)
        self.assertIn("col_a", loaded._col_name_to_idx)

    def test_structured_load_from_dict_handles_none_chi2_matrix(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        profile = dp.StructuredProfiler(data, options=profile_options)

        serialized = json.loads(json.dumps(profile, cls=ProfileEncoder))
        profile_data = serialized["data"]
        profile_data["chi2_matrix"] = None
        profile_data["correlation_matrix"] = None

        loaded = StructuredProfiler.load_from_dict(profile_data)
        self.assertIsNone(loaded.chi2_matrix)
        self.assertIsNone(loaded.correlation_matrix)

    def test_structured_load_from_dict_handles_array_matrices(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        profile = dp.StructuredProfiler(data, options=profile_options)

        serialized = json.loads(json.dumps(profile, cls=ProfileEncoder))
        profile_data = serialized["data"]
        profile_data["chi2_matrix"] = [[1.0, 0.5], [0.5, 1.0]]
        profile_data["correlation_matrix"] = [[1.0, 0.8], [0.8, 1.0]]

        loaded = StructuredProfiler.load_from_dict(profile_data)
        np.testing.assert_array_equal(
            loaded.chi2_matrix, np.array([[1.0, 0.5], [0.5, 1.0]])
        )
        np.testing.assert_array_equal(
            loaded.correlation_matrix, np.array([[1.0, 0.8], [0.8, 1.0]])
        )

    def test_base_profiler_load_from_dict_times_default_dict(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        profile = dp.StructuredProfiler(data, options=profile_options)

        serialized = json.loads(json.dumps(profile, cls=ProfileEncoder))
        profile_data = serialized["data"]

        loaded = StructuredProfiler.load_from_dict(profile_data)
        self.assertIsInstance(loaded.times, defaultdict)

    def test_unstructured_load_from_dict_with_none_profile(self):
        options_dict = {
            "class": "UnstructuredOptions",
            "data": {
                "text": {
                    "class": "TextProfilerOptions",
                    "data": {
                        "is_enabled": True,
                        "is_case_sensitive": True,
                        "stop_words": None,
                        "top_k_chars": None,
                        "top_k_words": None,
                        "vocab": {
                            "class": "BooleanOption",
                            "data": {"is_enabled": True},
                        },
                        "words": {
                            "class": "BooleanOption",
                            "data": {"is_enabled": True},
                        },
                    },
                },
                "data_labeler": {
                    "class": "DataLabelerOptions",
                    "data": {
                        "is_enabled": False,
                        "data_labeler_dirpath": None,
                        "data_labeler_object": None,
                        "max_sample_size": None,
                    },
                },
            },
        }
        profile_data = {
            "options": options_dict,
            "_profile": None,
            "total_samples": 0,
            "encoding": None,
            "file_type": None,
            "_samples_per_update": None,
            "_min_true_samples": 0,
            "times": {},
        }

        loaded = UnstructuredProfiler.load_from_dict(profile_data)
        self.assertIsInstance(loaded, UnstructuredProfiler)
        self.assertIsNone(loaded._profile)


class TestColumnProfilerBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility for individual column profilers."""

    def test_base_column_profiler_load_from_dict_missing_calculation(self):
        from dataprofiler.profilers.int_column_profile import IntColumn

        profiler = IntColumn("test_col")
        serialized = json.loads(json.dumps(profiler, cls=ProfileEncoder))
        profile_data = serialized["data"]

        profile_data["_IntColumn__calculations"]["nonexistent_metric"] = (
            "_nonexistent_method"
        )

        with self.assertRaises(AttributeError):
            IntColumn.load_from_dict(profile_data)

    def test_int_column_json_round_trip(self):
        from dataprofiler.profilers.int_column_profile import IntColumn

        profiler = IntColumn("test_col")
        profiler.update(pd.Series([1, 2, 3, 4, 5]))

        serialized = json.loads(json.dumps(profiler, cls=ProfileEncoder))
        loaded = load_column_profile(serialized)

        self.assertIsInstance(loaded, IntColumn)
        self.assertEqual(loaded.name, "test_col")
        self.assertEqual(loaded.match_count, profiler.match_count)

    def test_float_column_json_round_trip(self):
        from dataprofiler.profilers.float_column_profile import FloatColumn

        profiler = FloatColumn("test_col")
        profiler.update(pd.Series(["1.1", "2.2", "3.3"]))

        serialized = json.loads(json.dumps(profiler, cls=ProfileEncoder))
        loaded = load_column_profile(serialized)

        self.assertIsInstance(loaded, FloatColumn)
        self.assertEqual(loaded.name, "test_col")

    def test_datetime_column_json_round_trip(self):
        from dataprofiler.profilers.datetime_column_profile import DateTimeColumn

        profiler = DateTimeColumn("test_col")
        profiler.update(pd.Series(["2021-01-01", "2021-06-15", "2021-12-31"]))

        serialized = json.loads(json.dumps(profiler, cls=ProfileEncoder))
        loaded = load_column_profile(serialized)

        self.assertIsInstance(loaded, DateTimeColumn)
        self.assertEqual(loaded.name, "test_col")

    def test_categorical_column_json_round_trip(self):
        from dataprofiler.profilers.categorical_column_profile import (
            CategoricalColumn,
        )

        profiler = CategoricalColumn("test_col")
        profiler.update(pd.Series(["a", "b", "a", "c", "b", "a"]))

        serialized = json.loads(json.dumps(profiler, cls=ProfileEncoder))
        loaded = load_column_profile(serialized)

        self.assertIsInstance(loaded, CategoricalColumn)
        self.assertEqual(loaded.name, "test_col")

    def test_order_column_json_round_trip(self):
        from dataprofiler.profilers.order_column_profile import OrderColumn

        profiler = OrderColumn("test_col")
        profiler.update(pd.Series([1, 2, 3, 4, 5]))

        serialized = json.loads(json.dumps(profiler, cls=ProfileEncoder))
        loaded = load_column_profile(serialized)

        self.assertIsInstance(loaded, OrderColumn)
        self.assertEqual(loaded.name, "test_col")

    def test_text_column_json_round_trip(self):
        from dataprofiler.profilers.text_column_profile import TextColumn

        profiler = TextColumn("test_col")
        profiler.update(pd.Series(["hello", "world", "test"]))

        serialized = json.loads(json.dumps(profiler, cls=ProfileEncoder))
        loaded = load_column_profile(serialized)

        self.assertIsInstance(loaded, TextColumn)
        self.assertEqual(loaded.name, "test_col")


class TestCompilerBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility for compiler serialization."""

    def test_column_stats_compiler_json_round_trip(self):
        data = pd.Series(["a", "b", "c", "b", "a"], name="test")
        compiler = ColumnStatsProfileCompiler(data)

        serialized = json.loads(json.dumps(compiler, cls=ProfileEncoder))
        loaded = load_compiler(serialized)

        self.assertIsInstance(loaded, ColumnStatsProfileCompiler)
        self.assertEqual(loaded.name, "test")

    def test_primitive_type_compiler_json_round_trip(self):
        data = pd.Series(["hello", "world", "test"], name="test")
        compiler = ColumnPrimitiveTypeProfileCompiler(data)

        serialized = json.loads(json.dumps(compiler, cls=ProfileEncoder))
        loaded = load_compiler(serialized)

        self.assertIsInstance(loaded, ColumnPrimitiveTypeProfileCompiler)
        self.assertEqual(loaded.name, "test")

    def test_compiler_load_from_dict_preserves_profile_order(self):
        data = pd.Series(["hello", "world", "test"], name="test")
        compiler = ColumnPrimitiveTypeProfileCompiler(data)

        serialized = json.loads(json.dumps(compiler, cls=ProfileEncoder))
        loaded = load_compiler(serialized)

        original_keys = list(compiler._profiles.keys())
        loaded_keys = list(loaded._profiles.keys())
        self.assertEqual(original_keys, loaded_keys)


class TestOptionsBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility for profiler options serialization."""

    def test_structured_options_json_round_trip(self):
        options = StructuredOptions()
        serialized = json.loads(json.dumps(options, cls=ProfileEncoder))
        loaded = load_option(serialized)
        self.assertIsInstance(loaded, StructuredOptions)

    def test_unstructured_options_json_round_trip(self):
        options = UnstructuredOptions()
        serialized = json.loads(json.dumps(options, cls=ProfileEncoder))
        loaded = load_option(serialized)
        self.assertIsInstance(loaded, UnstructuredOptions)

    def test_profiler_options_json_round_trip(self):
        options = ProfilerOptions()
        serialized = json.loads(json.dumps(options, cls=ProfileEncoder))
        loaded = load_option(serialized)
        self.assertIsInstance(loaded, ProfilerOptions)

    def test_boolean_option_json_round_trip(self):
        option = BooleanOption(is_enabled=False)
        serialized = json.loads(json.dumps(option, cls=ProfileEncoder))
        loaded = load_option(serialized)
        self.assertIsInstance(loaded, BooleanOption)
        self.assertFalse(loaded.is_enabled)

    def test_option_load_from_dict_with_nested_options(self):
        options = ProfilerOptions()
        options.set({"structured_options.int.max.is_enabled": False})

        serialized = json.loads(json.dumps(options, cls=ProfileEncoder))
        loaded = load_option(serialized)

        self.assertIsInstance(loaded, ProfilerOptions)
        self.assertFalse(loaded.structured_options.int.max.is_enabled)

    def test_option_load_from_dict_with_extra_attributes(self):
        option = BooleanOption(is_enabled=True)
        serialized = json.loads(json.dumps(option, cls=ProfileEncoder))
        serialized["data"]["future_option"] = "test"

        loaded = load_option(serialized)
        self.assertIsInstance(loaded, BooleanOption)
        self.assertTrue(hasattr(loaded, "future_option"))


class TestStructuredColProfilerBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility for StructuredColProfiler serialization."""

    def test_structured_col_profiler_json_round_trip(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        profiler = dp.StructuredProfiler(data, options=profile_options)

        col_profiler = profiler.profile[0]
        serialized = json.loads(json.dumps(col_profiler, cls=ProfileEncoder))
        loaded = load_structured_col_profiler(serialized)

        self.assertIsInstance(loaded, StructuredColProfiler)

    def test_structured_col_profiler_preserves_profiles_dict(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        profiler = dp.StructuredProfiler(data, options=profile_options)

        col_profiler = profiler.profile[0]
        serialized = json.loads(json.dumps(col_profiler, cls=ProfileEncoder))
        loaded = load_structured_col_profiler(serialized)

        original_profile_keys = set(col_profiler.profiles.keys())
        loaded_profile_keys = set(loaded.profiles.keys())
        self.assertEqual(original_profile_keys, loaded_profile_keys)


class TestSaveLoadMethodCombinations(unittest.TestCase):
    """Test various combinations of save/load methods for compatibility."""

    def test_structured_pkl_save_json_load_fails_gracefully(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        profile = dp.StructuredProfiler(data, options=profile_options)

        with mock.patch("builtins.open") as m:
            mock_file = setup_save_mock_bytes_open(m)
            profile.save(save_method="pickle")
            pkl_content = mock_file.getvalue()

        mock_file_pkl = BytesIO(pkl_content)
        mock_file_pkl.close = lambda: None

        with mock.patch("builtins.open") as m:
            m.side_effect = lambda *args, **kwargs: mock_file_pkl

            with self.assertRaises(Exception):
                BaseProfiler.load("mock.pkl", "json")

    def test_structured_json_save_pkl_load_falls_back_to_json(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        profile = dp.StructuredProfiler(data, options=profile_options)

        with mock.patch("builtins.open") as m:
            mock_file = setup_save_mock_string_open(m)
            profile.save(save_method="json")
            json_content = mock_file.getvalue()

        mock_bytes = BytesIO(json_content.encode("utf-8"))
        mock_str = StringIO(json_content)
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_bytes
            return mock_str

        with mock.patch("builtins.open") as m:
            mock_bytes.close = lambda: None
            mock_str.close = lambda: None
            m.side_effect = side_effect

            loaded = BaseProfiler.load("mock.json")

        self.assertIsInstance(loaded, StructuredProfiler)

    def test_load_none_method_tries_pickle_then_json(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        profile = dp.StructuredProfiler(data, options=profile_options)

        with mock.patch("builtins.open") as m:
            mock_file = setup_save_mock_bytes_open(m)
            profile.save(save_method="pickle")
            mock_file.seek(0)

            with mock.patch("dataprofiler.profilers.profile_builder.DataLabeler"):
                loaded = BaseProfiler.load("mock.pkl", load_method=None)

        self.assertIsInstance(loaded, StructuredProfiler)

    def test_default_filepath_generation_pkl(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        profile = dp.StructuredProfiler(data, options=profile_options)

        with mock.patch("builtins.open") as m:
            setup_save_mock_bytes_open(m)
            profile.save()
            call_args = m.call_args[0][0]
            self.assertTrue(call_args.startswith("profile-"))
            self.assertTrue(call_args.endswith(".pkl"))

    def test_default_filepath_generation_json(self):
        data = pd.DataFrame({"a": [1, 2, 3]})
        profile_options = dp.ProfilerOptions()
        profile_options.set({"data_labeler.is_enabled": False})
        profile = dp.StructuredProfiler(data, options=profile_options)

        with mock.patch("builtins.open") as m:
            setup_save_mock_string_open(m)
            profile.save(save_method="json")
            call_args = m.call_args[0][0]
            self.assertTrue(call_args.startswith("profile-"))
            self.assertTrue(call_args.endswith(".json"))


if __name__ == "__main__":
    unittest.main()
