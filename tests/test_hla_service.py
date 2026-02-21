"""Unit tests for HLAService — HLA biological compatibility."""
import pytest
from app.services.hla_service import HLAService
from app.utils.encryption import encrypt_hla_data, decrypt_hla_data


@pytest.fixture
def hla_service():
    return HLAService()


class TestSBioCalculation:
    """Tests for S_bio = (N_unique / N_total) x 100."""

    def test_spec_worked_example(self, hla_service, sample_hla_alleles_a, sample_hla_alleles_b):
        """Section 10.4: 11 unique / 12 total = 91.67."""
        alleles_a = sample_hla_alleles_a["HLA-A"] + sample_hla_alleles_a["HLA-B"] + sample_hla_alleles_a["HLA-DRB1"]
        alleles_b = sample_hla_alleles_b["HLA-A"] + sample_hla_alleles_b["HLA-B"] + sample_hla_alleles_b["HLA-DRB1"]
        s_bio = hla_service._calculate_s_bio(alleles_a, alleles_b)
        # 11 unique out of 12 total = 91.67
        assert abs(s_bio - 91.67) < 0.5

    def test_identical_alleles(self, hla_service):
        """Same alleles = 50% (6 unique / 12 total)."""
        alleles = ["A*01:01", "A*02:01", "B*08:01", "B*07:02", "DRB1*15:01", "DRB1*03:01"]
        s_bio = hla_service._calculate_s_bio(alleles, alleles)
        assert abs(s_bio - 50.0) < 0.5

    def test_completely_different(self, hla_service):
        """No shared alleles = 100% (12 unique / 12 total)."""
        alleles_a = ["A*01:01", "A*02:01", "B*08:01", "B*07:02", "DRB1*15:01", "DRB1*03:01"]
        alleles_b = ["A*03:01", "A*24:02", "B*44:02", "B*35:01", "DRB1*04:01", "DRB1*07:01"]
        s_bio = hla_service._calculate_s_bio(alleles_a, alleles_b)
        assert abs(s_bio - 100.0) < 0.5


class TestHeterozygosity:
    """Tests for Heterozygosity Index."""

    def test_spec_example(self, hla_service, sample_hla_alleles_a, sample_hla_alleles_b):
        """11/12 = 0.9167."""
        alleles_a = sample_hla_alleles_a["HLA-A"] + sample_hla_alleles_a["HLA-B"] + sample_hla_alleles_a["HLA-DRB1"]
        alleles_b = sample_hla_alleles_b["HLA-A"] + sample_hla_alleles_b["HLA-B"] + sample_hla_alleles_b["HLA-DRB1"]
        h = hla_service._calculate_heterozygosity(alleles_a, alleles_b)
        assert abs(h - 0.9167) < 0.01


class TestOlfactoryPrediction:
    """Tests for olfactory attraction prediction."""

    def test_strong_attraction(self, hla_service):
        """Heterozygosity >=0.90 -> intensity 95."""
        result = hla_service._predict_olfactory_attraction(0.92)
        assert result["intensity"] == 95

    def test_weak_attraction(self, hla_service):
        """Heterozygosity <0.60 -> intensity 25."""
        result = hla_service._predict_olfactory_attraction(0.50)
        assert result["intensity"] == 25


class TestDisplayTier:
    """Tests for display tier logic."""

    def test_strong_tier(self, hla_service):
        """S_bio >=75 shows."""
        tier = hla_service._get_display_tier(85.0)
        assert tier["show"] is True
        assert tier["tier"] == "strong"

    def test_hidden_tier(self, hla_service):
        """S_bio <25 is hidden."""
        tier = hla_service._get_display_tier(20.0)
        assert tier["show"] is False

    def test_good_tier(self, hla_service):
        """S_bio 50-74 shows 'good'."""
        tier = hla_service._get_display_tier(60.0)
        assert tier["show"] is True
        assert tier["tier"] == "good"


class TestAlleleValidation:
    """Tests for allele format validation."""

    def test_valid_format(self, hla_service):
        """Standard format A*01:01 is valid."""
        assert hla_service._validate_allele_format("A*01:01") is True
        assert hla_service._validate_allele_format("B*44:02") is True
        assert hla_service._validate_allele_format("DRB1*15:01") is True

    def test_invalid_format(self, hla_service):
        """Invalid formats rejected."""
        assert hla_service._validate_allele_format("invalid") is False
        assert hla_service._validate_allele_format("A01:01") is False


class TestEncryption:
    """Tests for HLA data encryption/decryption."""

    def test_roundtrip(self):
        """Encrypt then decrypt gives back original data."""
        from unittest.mock import patch, MagicMock
        from cryptography.fernet import Fernet

        key = Fernet.generate_key()
        with patch("app.utils.encryption.get_settings") as mock:
            mock.return_value = MagicMock(FERNET_KEY=key.decode())
            data = {"HLA-A": ["A*01:01", "A*02:01"], "HLA-B": ["B*08:01", "B*07:02"]}
            encrypted = encrypt_hla_data(data)
            assert isinstance(encrypted, bytes)
            decrypted = decrypt_hla_data(encrypted)
            assert decrypted == data
