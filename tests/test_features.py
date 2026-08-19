"""Tests for src/features/features.py."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.features import (
    FEATURE_COLUMNS,
    KEEP_COLUMNS,
    engineer_features,
    feature_day_of_week,
    feature_has_question_mark,
    feature_hour_of_day,
    feature_sender_frequency,
    feature_sender_is_leadership,
    feature_subject_length,
    feature_thread_depth,
    feature_urgency_keyword_count,
    fit_subject_vectorizer,
    normalize_sender,
    parse_dates,
    transform_subjects,
)


@pytest.fixture
def email_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uid": ["1", "2", "3", "4"],
            "from": [
                "kenneth.lay@enron.com",
                "jane.doe@example.com",
                '"Lay, Kenneth" <kenneth.lay@enron.com>',
                "bob@example.com",
            ],
            "to": [None, "x@y.com", None, None],
            "subject": [
                "URGENT: budget",
                "Team lunch plans",
                "RE: RE: FW: reorg?",
                "deadline Friday",
            ],
            "date": [
                "2001-09-14 14:05:43-07:00",
                "2001-09-15 02:30:00-07:00",
                "not-a-date",
                "2001-09-17 09:00:00-07:00",
            ],
            "body": ["a", "b", "c", "d"],
            "priority": [1, 0, 1, 1],
        }
    )


class TestNormalizeSender:
    def test_bare_address(self):
        assert normalize_sender("Kenneth.Lay@enron.com") == "kenneth.lay@enron.com"

    def test_display_name_angle_brackets(self):
        raw = '"Lay, Kenneth" <kenneth.lay@enron.com>'
        assert normalize_sender(raw) == "kenneth.lay@enron.com"

    def test_null_returns_empty(self):
        assert normalize_sender(None) == ""

    def test_unparseable_falls_back_to_raw(self):
        assert normalize_sender("just a name") == "just a name"


class TestParseDates:
    def test_parses_utc(self):
        series = pd.Series(["2001-09-14 14:05:43-07:00"])
        parsed = parse_dates(series)
        assert str(parsed.iloc[0]) == "2001-09-14 21:05:43+00:00"

    def test_malformed_becomes_nat(self):
        parsed = parse_dates(pd.Series(["not-a-date"]))
        assert pd.isna(parsed.iloc[0])

    def test_preserves_length(self):
        series = pd.Series(["2001-09-14", "garbage", None, "2001-09-17"])
        assert len(parse_dates(series)) == 4


class TestFeatureSenderFrequency:
    def test_leadership_sender_counted_across_display_forms(self, email_df):
        result = feature_sender_frequency(email_df)
        assert result.iloc[0] == 2  # kenneth.lay@enron.com twice (rows 0 and 2)
        assert result.iloc[2] == 2

    def test_null_sender_yields_zero(self):
        df = pd.DataFrame({"from": [None, "a@x.com"]})
        result = feature_sender_frequency(df)
        assert result.iloc[0] == 0

    def test_dtype_is_int64(self, email_df):
        assert feature_sender_frequency(email_df).dtype == np.int64


class TestFeatureSenderIsLeadership:
    def test_leadership_sender_is_one(self, email_df):
        result = feature_sender_is_leadership(email_df)
        assert result.iloc[0] == 1

    def test_display_name_form_still_leadership(self, email_df):
        result = feature_sender_is_leadership(email_df)
        assert result.iloc[2] == 1

    def test_regular_sender_is_zero(self, email_df):
        result = feature_sender_is_leadership(email_df)
        assert result.iloc[1] == 0

    def test_null_sender_is_zero_and_dtype(self):
        df = pd.DataFrame({"from": [None, "jane@x.com"]})
        result = feature_sender_is_leadership(df)
        assert result.iloc[0] == 0
        assert result.dtype == np.int64


class TestFeatureHourOfDay:
    def test_parses_utc_hour(self, email_df):
        result = feature_hour_of_day(email_df)
        assert result.iloc[0] == 21  # 14:05 -07:00 -> 21:05 UTC
        assert result.iloc[1] == 9  # 02:30 -07:00 -> 09:30 UTC

    def test_malformed_date_yields_minus_one(self, email_df):
        result = feature_hour_of_day(email_df)
        assert result.iloc[2] == -1

    def test_dtype_is_int64(self, email_df):
        assert feature_hour_of_day(email_df).dtype == np.int64


class TestFeatureDayOfWeek:
    def test_friday_is_four(self, email_df):
        result = feature_day_of_week(email_df)
        assert result.iloc[0] == 4  # 2001-09-14 was a Friday

    def test_malformed_date_yields_minus_one(self, email_df):
        result = feature_day_of_week(email_df)
        assert result.iloc[2] == -1

    def test_dtype_is_int64(self, email_df):
        assert feature_day_of_week(email_df).dtype == np.int64


class TestFeatureUrgencyKeywordCount:
    def test_single_keyword(self, email_df):
        result = feature_urgency_keyword_count(email_df)
        assert result.iloc[0] == 1  # "URGENT"
        assert result.iloc[1] == 0
        assert result.iloc[3] == 1  # "deadline"

    def test_case_insensitive_and_distinct(self):
        df = pd.DataFrame({"subject": ["ASAP URGENT asap"]})
        result = feature_urgency_keyword_count(df)
        assert result.iloc[0] == 2  # distinct: asap, urgent

    def test_null_subject_yields_zero(self):
        df = pd.DataFrame({"subject": [None, ""]})
        result = feature_urgency_keyword_count(df)
        assert (result == 0).all()

    def test_dtype_is_int64(self, email_df):
        assert feature_urgency_keyword_count(email_df).dtype == np.int64


class TestFeatureSubjectLength:
    def test_counts_chars(self, email_df):
        result = feature_subject_length(email_df)
        assert result.iloc[0] == len("URGENT: budget")
        assert result.iloc[1] == len("Team lunch plans")

    def test_null_and_empty_yield_zero(self):
        df = pd.DataFrame({"subject": [None, ""]})
        result = feature_subject_length(df)
        assert (result == 0).all()

    def test_dtype_is_int64(self, email_df):
        assert feature_subject_length(email_df).dtype == np.int64


class TestFeatureHasQuestionMark:
    def test_question_subject_is_one(self, email_df):
        result = feature_has_question_mark(email_df)
        assert result.iloc[2] == 1  # "RE: RE: FW: reorg?"

    def test_no_question_is_zero(self, email_df):
        result = feature_has_question_mark(email_df)
        assert result.iloc[0] == 0

    def test_null_subject_is_zero(self):
        df = pd.DataFrame({"subject": [None, "no q"]})
        result = feature_has_question_mark(df)
        assert result.iloc[0] == 0

    def test_dtype_is_int64(self, email_df):
        assert feature_has_question_mark(email_df).dtype == np.int64


class TestFeatureThreadDepth:
    def test_counts_markers(self, email_df):
        result = feature_thread_depth(email_df)
        assert result.iloc[2] == 3  # RE: RE: FW:

    def test_bracket_count_rule(self):
        df = pd.DataFrame({"subject": ["RE[2]: topic", "FW: topic"]})
        result = feature_thread_depth(df)
        assert result.iloc[0] == 2
        assert result.iloc[1] == 1

    def test_no_marker_and_null_yield_zero(self):
        df = pd.DataFrame({"subject": ["topic", None, "RE:"]})
        result = feature_thread_depth(df)
        assert result.iloc[0] == 0
        assert result.iloc[1] == 0

    def test_dtype_is_int64(self, email_df):
        assert feature_thread_depth(email_df).dtype == np.int64


class TestFitSubjectVectorizer:
    def test_max_features_respected(self):
        subjects = pd.Series([f"token{i}" for i in range(200)])
        vectorizer = fit_subject_vectorizer(subjects)
        assert len(vectorizer.vocabulary_) == 100

    def test_handles_nan_subjects(self):
        subjects = pd.Series(["hello world", None, "another topic"])
        vectorizer = fit_subject_vectorizer(subjects)
        assert isinstance(vectorizer, TfidfVectorizer)
        assert len(vectorizer.vocabulary_) > 0

    def test_default_returns_tfidf_vectorizer(self):
        subjects = pd.Series(["urgent", "budget"])
        vectorizer = fit_subject_vectorizer(subjects)
        assert isinstance(vectorizer, TfidfVectorizer)


class TestTransformSubjects:
    def test_returns_csr_matrix(self):
        subjects = pd.Series(["urgent budget", "team lunch"])
        vectorizer = fit_subject_vectorizer(subjects)
        matrix = transform_subjects(vectorizer, subjects)
        assert isinstance(matrix, csr_matrix)

    def test_shape_matches_rows_and_vocab(self):
        subjects = pd.Series(["urgent budget", "team lunch", "reorg question"])
        vectorizer = fit_subject_vectorizer(subjects)
        matrix = transform_subjects(vectorizer, subjects)
        assert matrix.shape[0] == 3
        assert matrix.shape[1] == len(vectorizer.vocabulary_)

    def test_nan_subject_yields_all_zero_row(self):
        subjects = pd.Series(["urgent budget", None])
        vectorizer = fit_subject_vectorizer(subjects)
        matrix = transform_subjects(vectorizer, subjects)
        assert matrix.shape[0] == 2
        assert np.all(matrix.getrow(1).toarray() == 0)


class TestEngineerFeatures:
    def test_output_columns_exact(self, email_df):
        result = engineer_features(email_df)
        assert list(result.columns) == list(KEEP_COLUMNS) + list(FEATURE_COLUMNS)

    def test_body_and_to_dropped(self, email_df):
        result = engineer_features(email_df)
        assert "body" not in result.columns
        assert "to" not in result.columns

    def test_row_count_preserved(self, email_df):
        result = engineer_features(email_df)
        assert len(result) == len(email_df)

    def test_feature_values_populated(self, email_df):
        result = engineer_features(email_df)
        assert result["sender_is_leadership"].iloc[0] == 1
        assert result["urgency_keyword_count"].iloc[0] == 1
        assert result["hour_of_day"].iloc[2] == -1

    def test_all_feature_dtypes_int64(self, email_df):
        result = engineer_features(email_df)
        assert all(result[c].dtype == np.int64 for c in FEATURE_COLUMNS)
