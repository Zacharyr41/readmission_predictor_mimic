"""Tests for SNOMED-CT terminology mapper."""

import json
import pytest
from pathlib import Path

from src.graph_construction.terminology.snomed_mapper import SnomedConcept, SnomedMapper


@pytest.fixture
def mappings_dir(tmp_path: Path) -> Path:
    """Create minimal mapping JSON files for testing."""
    # ICD-10-CM -> SNOMED
    icd_data = {
        "_metadata": {"source": "test"},
        "I639": {"snomed_code": "432504007", "snomed_term": "Cerebral infarction, unspecified"},
        "I63": {"snomed_code": "20059004", "snomed_term": "Cerebral artery occlusion"},
        "I630": {"snomed_code": "432504007", "snomed_term": "Cerebral infarction due to thrombosis of precerebral arteries"},
        "E119": {"snomed_code": "44054006", "snomed_term": "Type 2 diabetes mellitus without complications"},
    }
    (tmp_path / "icd10cm_to_snomed.json").write_text(json.dumps(icd_data))

    # Lab items -> SNOMED
    lab_data = {
        "_metadata": {"source": "test"},
        "50912": {"snomed_code": "70901006", "snomed_term": "Creatinine measurement", "loinc": "2160-0"},
        "50971": {"snomed_code": "104934005", "snomed_term": "Sodium measurement", "loinc": "2951-2"},
        "51265": {"snomed_code": "61928009", "snomed_term": "Platelet count", "loinc": "777-3"},
    }
    (tmp_path / "labitem_to_snomed.json").write_text(json.dumps(lab_data))

    # Chart items -> SNOMED
    chart_data = {
        "_metadata": {"source": "test"},
        "220045": {"snomed_code": "364075005", "snomed_term": "Heart rate", "loinc": "8867-4"},
        "220179": {"snomed_code": "271649006", "snomed_term": "Non-invasive systolic blood pressure", "loinc": "8480-6"},
    }
    (tmp_path / "chartitem_to_snomed.json").write_text(json.dumps(chart_data))

    # Drug -> SNOMED
    drug_data = {
        "_metadata": {"source": "test"},
        "vancomycin": {"snomed_code": "372735009", "snomed_term": "Vancomycin", "rxcui": "11124"},
        "ceftriaxone": {"snomed_code": "372670001", "snomed_term": "Ceftriaxone", "rxcui": "2193"},
        "aspirin": {"snomed_code": "387458008", "snomed_term": "Aspirin", "rxcui": "1191"},
    }
    (tmp_path / "drug_to_snomed.json").write_text(json.dumps(drug_data))

    # Organism -> SNOMED
    org_data = {
        "_metadata": {"source": "test"},
        "STAPHYLOCOCCUS AUREUS": {"snomed_code": "3092008", "snomed_term": "Staphylococcus aureus"},
        "ESCHERICHIA COLI": {"snomed_code": "112283007", "snomed_term": "Escherichia coli"},
    }
    (tmp_path / "organism_to_snomed.json").write_text(json.dumps(org_data))

    # Comorbidity -> SNOMED
    comorbidity_data = {
        "_metadata": {"source": "test"},
        "diabetes": {"snomed_code": "73211009", "snomed_term": "Diabetes mellitus"},
        "hypertension": {"snomed_code": "38341003", "snomed_term": "Essential hypertension"},
    }
    (tmp_path / "comorbidity_to_snomed.json").write_text(json.dumps(comorbidity_data))

    # LOINC -> SNOMED
    loinc_data = {
        "_metadata": {"source": "test"},
        "2160-0": {"snomed_code": "70901006", "snomed_term": "Creatinine measurement"},
        "2951-2": {"snomed_code": "104934005", "snomed_term": "Sodium measurement"},
    }
    (tmp_path / "loinc_to_snomed.json").write_text(json.dumps(loinc_data))

    return tmp_path


@pytest.fixture
def mapper(mappings_dir: Path) -> SnomedMapper:
    return SnomedMapper(mappings_dir)


class TestSnomedMapperICD:
    """ICD code lookup tests."""

    def test_icd10_exact_match_undotted(self, mapper: SnomedMapper) -> None:
        result = mapper.get_snomed_for_icd("I639", 10)
        assert result is not None
        assert result.code == "432504007"
        assert result.term == "Cerebral infarction, unspecified"

    def test_icd10_exact_match_dotted(self, mapper: SnomedMapper) -> None:
        result = mapper.get_snomed_for_icd("I63.9", 10)
        assert result is not None
        assert result.code == "432504007"

    def test_icd10_fallback_to_3char_prefix(self, mapper: SnomedMapper) -> None:
        # I635 not in map, but I63 is
        result = mapper.get_snomed_for_icd("I635", 10)
        assert result is not None
        assert result.code == "20059004"

    def test_icd9_returns_none(self, mapper: SnomedMapper) -> None:
        result = mapper.get_snomed_for_icd("4280", 9)
        assert result is None

    def test_unmapped_code_returns_none(self, mapper: SnomedMapper) -> None:
        result = mapper.get_snomed_for_icd("Z999", 10)
        assert result is None


class TestSnomedMapperLabItem:
    """Lab item lookup tests."""

    def test_labitem_lookup(self, mapper: SnomedMapper) -> None:
        result = mapper.get_snomed_for_labitem(50912)
        assert result is not None
        assert result.code == "70901006"
        assert result.term == "Creatinine measurement"

    def test_labitem_unmapped(self, mapper: SnomedMapper) -> None:
        result = mapper.get_snomed_for_labitem(99999)
        assert result is None


class TestSnomedMapperChartItem:
    """Chart item lookup tests."""

    def test_chartitem_lookup(self, mapper: SnomedMapper) -> None:
        result = mapper.get_snomed_for_chartitem(220045)
        assert result is not None
        assert result.code == "364075005"
        assert result.term == "Heart rate"

    def test_chartitem_unmapped(self, mapper: SnomedMapper) -> None:
        result = mapper.get_snomed_for_chartitem(99999)
        assert result is None


class TestSnomedMapperDrug:
    """Drug name lookup tests."""

    def test_drug_case_insensitive(self, mapper: SnomedMapper) -> None:
        result = mapper.get_snomed_for_drug("Vancomycin")
        assert result is not None
        assert result.code == "372735009"

    def test_drug_uppercase(self, mapper: SnomedMapper) -> None:
        result = mapper.get_snomed_for_drug("VANCOMYCIN")
        assert result is not None
        assert result.code == "372735009"

    def test_drug_with_dosage_stripped(self, mapper: SnomedMapper) -> None:
        result = mapper.get_snomed_for_drug("Aspirin 81mg")
        assert result is not None
        assert result.code == "387458008"

    def test_drug_unmapped(self, mapper: SnomedMapper) -> None:
        result = mapper.get_snomed_for_drug("UnknownDrug12345")
        assert result is None


class TestSnomedMapperOrganism:
    """Organism name lookup tests."""

    def test_organism_lookup(self, mapper: SnomedMapper) -> None:
        result = mapper.get_snomed_for_organism("STAPHYLOCOCCUS AUREUS")
        assert result is not None
        assert result.code == "3092008"

    def test_organism_case_insensitive(self, mapper: SnomedMapper) -> None:
        result = mapper.get_snomed_for_organism("staphylococcus aureus")
        assert result is not None
        assert result.code == "3092008"

    def test_organism_unmapped(self, mapper: SnomedMapper) -> None:
        result = mapper.get_snomed_for_organism("UNKNOWN ORGANISM")
        assert result is None


class TestSnomedMapperComorbidity:
    """Comorbidity name lookup tests."""

    def test_comorbidity_lookup(self, mapper: SnomedMapper) -> None:
        result = mapper.get_snomed_for_comorbidity("diabetes")
        assert result is not None
        assert result.code == "73211009"

    def test_comorbidity_unmapped(self, mapper: SnomedMapper) -> None:
        result = mapper.get_snomed_for_comorbidity("unknown_comorbidity")
        assert result is None


class TestSnomedMapperGracefulDegradation:
    """Graceful degradation when files are missing or malformed."""

    def test_missing_directory(self, tmp_path: Path) -> None:
        mapper = SnomedMapper(tmp_path / "nonexistent")
        result = mapper.get_snomed_for_icd("I639", 10)
        assert result is None

    def test_empty_directory(self, tmp_path: Path) -> None:
        mapper = SnomedMapper(tmp_path)
        result = mapper.get_snomed_for_icd("I639", 10)
        assert result is None
        result2 = mapper.get_snomed_for_labitem(50912)
        assert result2 is None

    def test_coverage_stats(self, mapper: SnomedMapper) -> None:
        stats = mapper.coverage_stats()
        assert stats["icd"] == 4
        assert stats["labitem"] == 3
        assert stats["chartitem"] == 2
        assert stats["drug"] == 3
        assert stats["organism"] == 2
        assert stats["comorbidity"] == 2
        assert stats["loinc"] == 2


class TestSnomedConcept:
    """Tests for the SnomedConcept dataclass."""

    def test_frozen(self) -> None:
        concept = SnomedConcept(code="12345", term="Test concept")
        with pytest.raises(AttributeError):
            concept.code = "99999"

    def test_equality(self) -> None:
        a = SnomedConcept(code="12345", term="Test")
        b = SnomedConcept(code="12345", term="Test")
        assert a == b


class TestCUIRejection:
    """SCTID validation: CUI codes must be rejected."""

    def test_cui_code_returns_none(self, mappings_dir: Path) -> None:
        """get_snomed_for_labitem() rejects CUI-format snomed_codes."""
        # Write a lab entry with a CUI instead of a real SCTID
        lab_data = {
            "_metadata": {"source": "test"},
            "99999": {"snomed_code": "C0201985", "snomed_term": "Fake CUI entry", "loinc": "2160-0"},
        }
        (mappings_dir / "labitem_to_snomed.json").write_text(json.dumps(lab_data))
        mapper = SnomedMapper(mappings_dir)
        result = mapper.get_snomed_for_labitem(99999)
        assert result is None, f"CUI code should be rejected, got {result}"

    def test_valid_sctid_is_accepted(self, mapper: SnomedMapper) -> None:
        """get_snomed_for_labitem() accepts numeric SCTIDs."""
        result = mapper.get_snomed_for_labitem(50912)
        assert result is not None
        assert result.code == "70901006"


class TestLoincLookup:
    """LOINC code to SNOMED concept lookup."""

    @pytest.fixture
    def loinc_mapper(self, mappings_dir: Path) -> SnomedMapper:
        loinc_data = {
            "_metadata": {"source": "test"},
            "2160-0": {"snomed_code": "70901006", "snomed_term": "Creatinine measurement"},
            "2951-2": {"snomed_code": "104934005", "snomed_term": "Sodium measurement"},
        }
        (mappings_dir / "loinc_to_snomed.json").write_text(json.dumps(loinc_data))
        return SnomedMapper(mappings_dir)

    def test_loinc_lookup_returns_concept(self, loinc_mapper: SnomedMapper) -> None:
        result = loinc_mapper.get_snomed_for_loinc("2160-0")
        assert result is not None
        assert result.code == "70901006"

    def test_loinc_lookup_unmapped_returns_none(self, loinc_mapper: SnomedMapper) -> None:
        result = loinc_mapper.get_snomed_for_loinc("99999-9")
        assert result is None

    def test_loinc_lookup_term_is_populated(self, loinc_mapper: SnomedMapper) -> None:
        result = loinc_mapper.get_snomed_for_loinc("2160-0")
        assert result is not None
        assert result.term and len(result.term) > 0
