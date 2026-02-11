"""Tests for MappingChain waterfall resolution."""

import pytest

from src.graph_construction.terminology.mapping_chain import MappingChain


class _FakeSource:
    """Minimal MappingSource stub for testing."""

    def __init__(self, name: str, data: dict[str, dict]) -> None:
        self._name = name
        self._data = data
        self.lookup_calls: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def lookup(self, loinc_code: str) -> dict | None:
        self.lookup_calls.append(loinc_code)
        return self._data.get(loinc_code)

    def lookup_batch(self, codes: list[str]) -> dict[str, dict]:
        results = {}
        for code in codes:
            hit = self.lookup(code)
            if hit is not None:
                results[code] = hit
        return results


class TestMappingChain:

    def test_first_source_wins(self) -> None:
        s1 = _FakeSource("s1", {"2160-0": {"snomed_code": "111", "snomed_term": "From S1"}})
        s2 = _FakeSource("s2", {"2160-0": {"snomed_code": "222", "snomed_term": "From S2"}})
        chain = MappingChain([s1, s2])
        result = chain.resolve("2160-0")
        assert result is not None
        assert result["snomed_code"] == "111"
        # s2 should NOT have been consulted
        assert "2160-0" not in s2.lookup_calls

    def test_fallthrough_to_second(self) -> None:
        s1 = _FakeSource("s1", {})
        s2 = _FakeSource("s2", {"2160-0": {"snomed_code": "222", "snomed_term": "From S2"}})
        chain = MappingChain([s1, s2])
        result = chain.resolve("2160-0")
        assert result is not None
        assert result["snomed_code"] == "222"

    def test_both_miss_returns_none(self) -> None:
        s1 = _FakeSource("s1", {})
        s2 = _FakeSource("s2", {})
        chain = MappingChain([s1, s2])
        assert chain.resolve("99999-9") is None

    def test_resolve_batch_waterfall(self) -> None:
        """Source 2 only sees codes unresolved by source 1."""
        s1 = _FakeSource("s1", {"A": {"snomed_code": "1", "snomed_term": "A"}})
        s2 = _FakeSource("s2", {"B": {"snomed_code": "2", "snomed_term": "B"}})
        chain = MappingChain([s1, s2])
        results = chain.resolve_batch(["A", "B", "C"])
        assert set(results.keys()) == {"A", "B"}
        # s2 should NOT have seen "A" (already resolved)
        assert "A" not in s2.lookup_calls

    def test_empty_chain_returns_none(self) -> None:
        chain = MappingChain([])
        assert chain.resolve("2160-0") is None
        assert chain.resolve_batch(["2160-0"]) == {}
