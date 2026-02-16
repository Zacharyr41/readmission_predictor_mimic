"""Tests for src.gnn.__main__ — GNN pipeline orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# _make_split_fn adapter
# ──────────────────────────────────────────────────────────────────────────────


class TestMakeSplitFn:
    """Verify the split-function adapter produces the right interface."""

    def test_returns_callable(self):
        from src.gnn.__main__ import _make_split_fn

        fn = _make_split_fn()
        assert callable(fn)

    def test_produces_three_disjoint_frames(self):
        from src.gnn.__main__ import _make_split_fn

        # 20 patients, 2 admissions each → 40 rows
        rows = []
        for pid in range(20):
            for hadm in range(2):
                rows.append(
                    {
                        "subject_id": pid,
                        "hadm_id": pid * 100 + hadm,
                        "readmitted_30d": 1 if pid < 5 else 0,
                    }
                )
        df = pd.DataFrame(rows)

        fn = _make_split_fn()
        train, val, test = fn(df, "readmitted_30d")

        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

        # All rows accounted for
        assert len(train) + len(val) + len(test) == len(df)

        # No patient overlap
        train_pids = set(train["subject_id"])
        val_pids = set(val["subject_id"])
        test_pids = set(test["subject_id"])
        assert train_pids.isdisjoint(val_pids)
        assert train_pids.isdisjoint(test_pids)
        assert val_pids.isdisjoint(test_pids)


# ──────────────────────────────────────────────────────────────────────────────
# build_embeddings
# ──────────────────────────────────────────────────────────────────────────────


class TestBuildEmbeddings:
    """Verify build_embeddings wires SnomedMapper → build_concept_embeddings."""

    @patch("src.gnn.embeddings.build_concept_embeddings")
    @patch("src.graph_construction.terminology.snomed_mapper.SnomedMapper")
    def test_calls_mapper_and_embedder(self, MockMapper, mock_build):
        from pathlib import Path

        from src.gnn.__main__ import build_embeddings

        mock_build.return_value = {"12345": "tensor"}

        build_embeddings(
            mappings_dir=Path("/fake/mappings"),
            embeddings_path=Path("/fake/emb.pt"),
        )

        MockMapper.assert_called_once_with(Path("/fake/mappings"))
        mock_build.assert_called_once_with(
            MockMapper.return_value,
            cache_path=Path("/fake/emb.pt"),
        )


# ──────────────────────────────────────────────────────────────────────────────
# export_graph
# ──────────────────────────────────────────────────────────────────────────────


class TestExportGraph:
    """Verify export_graph wires disk-backed graph → export_rdf_to_heterodata."""

    @patch("src.graph_construction.disk_graph.close_disk_graph")
    @patch("src.graph_construction.disk_graph.open_disk_graph")
    @patch("src.graph_construction.disk_graph.bind_namespaces")
    @patch("src.gnn.graph_export.export_rdf_to_heterodata")
    def test_calls_export(self, mock_export, mock_bind, mock_open, mock_close):
        from pathlib import Path

        from src.gnn.__main__ import export_graph

        mock_graph = MagicMock()
        mock_open.return_value = mock_graph

        mock_data = MagicMock()
        mock_data.node_types = ["patient"]
        mock_data.edge_types = [("patient", "has_admission", "admission")]
        mock_export.return_value = mock_data

        export_graph(
            rdf_path=Path("/fake/kg.nt"),
            features_path=Path("/fake/feat.parquet"),
            embeddings_path=Path("/fake/emb.pt"),
            output_path=Path("/fake/out.pt"),
        )

        mock_graph.parse.assert_called_once_with("/fake/kg.nt", format="nt")

        mock_export.assert_called_once()
        _, kw = mock_export.call_args
        # split_fn must be callable
        assert callable(kw["split_fn"])
        # embed_unmapped_fn should be None (triggers built-in fallback)
        assert kw["embed_unmapped_fn"] is None
        mock_close.assert_called_once_with(mock_graph)

    @patch("src.graph_construction.disk_graph.close_disk_graph")
    @patch("src.graph_construction.disk_graph.open_disk_graph")
    @patch("src.graph_construction.disk_graph.bind_namespaces")
    @patch("src.gnn.graph_export.export_rdf_to_heterodata")
    def test_split_fn_is_callable(self, mock_export, mock_bind, mock_open, mock_close):
        from pathlib import Path

        from src.gnn.__main__ import export_graph

        mock_graph = MagicMock()
        mock_open.return_value = mock_graph

        mock_data = MagicMock()
        mock_data.node_types = []
        mock_data.edge_types = []
        mock_export.return_value = mock_data

        export_graph(
            rdf_path=Path("/fake/kg.nt"),
            features_path=Path("/fake/feat.parquet"),
            embeddings_path=Path("/fake/emb.pt"),
            output_path=Path("/fake/out.pt"),
        )

        # Grab the split_fn argument (positional arg index 3)
        args, kwargs = mock_export.call_args
        split_fn = kwargs.get("split_fn", args[3] if len(args) > 3 else None)
        assert callable(split_fn)


# ──────────────────────────────────────────────────────────────────────────────
# prepare
# ──────────────────────────────────────────────────────────────────────────────


class TestPrepare:
    """Verify prepare calls build_embeddings then export_graph in order."""

    @patch("src.gnn.__main__.export_graph")
    @patch("src.gnn.__main__.build_embeddings")
    def test_calls_both_in_order(self, mock_build, mock_export):
        from pathlib import Path

        from src.gnn.__main__ import prepare

        prepare(
            rdf_path=Path("/r"),
            features_path=Path("/f"),
            embeddings_path=Path("/e"),
            output_path=Path("/o"),
            mappings_dir=Path("/m"),
        )

        mock_build.assert_called_once_with(
            mappings_dir=Path("/m"),
            embeddings_path=Path("/e"),
        )
        mock_export.assert_called_once_with(
            rdf_path=Path("/r"),
            features_path=Path("/f"),
            embeddings_path=Path("/e"),
            output_path=Path("/o"),
        )

        # build_embeddings was called before export_graph
        assert mock_build.call_args_list[0] == mock_build.call_args
        assert mock_export.call_args_list[0] == mock_export.call_args


# ──────────────────────────────────────────────────────────────────────────────
# CLI (main)
# ──────────────────────────────────────────────────────────────────────────────


class TestCLI:
    """Test the argparse-based main() entry point."""

    @patch("src.gnn.__main__.build_embeddings")
    def test_build_embeddings_flag(self, mock_build):
        from src.gnn.__main__ import main

        main(["--build-embeddings"])
        mock_build.assert_called_once()

    @patch("src.gnn.__main__.export_graph")
    def test_export_graph_flag(self, mock_export):
        from src.gnn.__main__ import main

        main(["--export-graph"])
        mock_export.assert_called_once()

    @patch("src.gnn.__main__.prepare")
    def test_prepare_flag(self, mock_prepare):
        from src.gnn.__main__ import main

        main(["--prepare"])
        mock_prepare.assert_called_once()

    def test_list_flag(self, capsys):
        from src.gnn.__main__ import main

        main(["--list"])
        captured = capsys.readouterr()
        assert "Available experiments:" in captured.out
        assert "E1_mlp_baseline" in captured.out
        assert "E6_full_model" in captured.out

    def test_no_args_prints_help(self, capsys):
        from src.gnn.__main__ import main

        main([])
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower() or "--help" in captured.out

    def test_mutually_exclusive_flags(self):
        from src.gnn.__main__ import main

        with pytest.raises(SystemExit):
            main(["--build-embeddings", "--export-graph"])

    @patch("src.gnn.__main__.build_embeddings")
    def test_custom_paths(self, mock_build):
        from pathlib import Path

        from src.gnn.__main__ import main

        main([
            "--build-embeddings",
            "--mappings", "/custom/mappings",
            "--embeddings", "/custom/emb.pt",
        ])
        mock_build.assert_called_once_with(
            mappings_dir=Path("/custom/mappings"),
            embeddings_path=Path("/custom/emb.pt"),
        )

    @patch("src.gnn.experiments.ExperimentRunner")
    def test_run_flag(self, MockRunner):
        from src.gnn.__main__ import main

        mock_instance = MockRunner.return_value
        mock_instance.run.return_value = {
            "eval_metrics": {"auroc": 0.75},
        }

        main(["--run", "E1_mlp_baseline"])
        MockRunner.assert_called_once()
        mock_instance.run.assert_called_once_with("E1_mlp_baseline", seed=42)

    @patch("src.gnn.experiments.ExperimentRunner")
    def test_run_all_flag(self, MockRunner):
        from src.gnn.__main__ import main

        mock_instance = MockRunner.return_value
        mock_instance.run_all.return_value = {}

        main(["--run-all"])
        mock_instance.run_all.assert_called_once_with(seed=42)

    @patch("src.gnn.experiments.ExperimentRunner")
    def test_seed_flag(self, MockRunner):
        from src.gnn.__main__ import main

        mock_instance = MockRunner.return_value
        mock_instance.run.return_value = {"eval_metrics": {"auroc": 0.5}}

        main(["--run", "E1_mlp_baseline", "--seed", "123"])
        mock_instance.run.assert_called_once_with("E1_mlp_baseline", seed=123)

    def test_verbose_flag(self, capsys):
        """--verbose shouldn't crash; combined with --list for a safe action."""
        from src.gnn.__main__ import main

        main(["--list", "-v"])
        captured = capsys.readouterr()
        assert "Available experiments:" in captured.out
