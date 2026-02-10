"""Tests for CLI --data-source and --bigquery-project flags."""

from unittest.mock import patch
import subprocess
import sys


class TestMainCLIDataSource:
    """Test --data-source flag in src.main CLI."""

    def test_data_source_bigquery_in_help(self):
        """--data-source flag should appear in main.py help output."""
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--help"],
            capture_output=True,
            text=True,
            cwd="/Users/zacharyrothstein/Code/readmission_predictor_mimic",
        )
        assert "--data-source" in result.stdout

    def test_bigquery_project_in_help(self):
        """--bigquery-project flag should appear in main.py help output."""
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--help"],
            capture_output=True,
            text=True,
            cwd="/Users/zacharyrothstein/Code/readmission_predictor_mimic",
        )
        assert "--bigquery-project" in result.stdout


class TestIngestionCLIDataSource:
    """Test --data-source flag in src.ingestion CLI."""

    def test_data_source_in_help(self):
        """--data-source flag should appear in ingestion help output."""
        result = subprocess.run(
            [sys.executable, "-m", "src.ingestion", "--help"],
            capture_output=True,
            text=True,
            cwd="/Users/zacharyrothstein/Code/readmission_predictor_mimic",
        )
        assert "--data-source" in result.stdout

    def test_bigquery_project_in_help(self):
        """--bigquery-project flag should appear in ingestion help output."""
        result = subprocess.run(
            [sys.executable, "-m", "src.ingestion", "--help"],
            capture_output=True,
            text=True,
            cwd="/Users/zacharyrothstein/Code/readmission_predictor_mimic",
        )
        assert "--bigquery-project" in result.stdout
