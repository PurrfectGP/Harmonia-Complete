"""
Harmonia V3 — HLA Biological Compatibility Service

Calculates biological compatibility between two users based on their
Human Leukocyte Antigen (HLA) genotype data.  The service handles:

  - Decryption of stored HLA payloads (Fernet at rest)
  - Unique-allele diversity scoring  (S_bio)
  - Heterozygosity Index calculation
  - Olfactory attraction prediction
  - Peptide-binding analysis with disease-association references
  - Display-tier generation for the match card
  - Validated upload of new HLA data
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.hla import HLAData
from app.utils.encryption import encrypt_hla_data, decrypt_hla_data

logger = structlog.get_logger("harmonia.hla_service")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Standard loci and expected allele count per user
_STANDARD_LOCI = ("A", "B", "C")
_ALLELES_PER_LOCUS = 2
_ALLELES_PER_USER = len(_STANDARD_LOCI) * _ALLELES_PER_LOCUS  # 6
_TOTAL_ALLELE_SLOTS = _ALLELES_PER_USER * 2  # 12 for a pair

# Regex for validated allele format: e.g. A*01:01, B*44:02, C*07:01
_ALLELE_PATTERN = re.compile(
    r"^[A-Z]{1,3}\*\d{2,4}:\d{2,4}$"
)

# ──────────────────────────────────────────────────────────────────────────────
# Reference data: common HLA alleles and their known peptide-binding /
# disease associations (curated subset for MVP display purposes)
# ──────────────────────────────────────────────────────────────────────────────

_HLA_REFERENCE: dict[str, dict[str, Any]] = {
    "A*01:01": {
        "binding_characteristics": "Prefers peptides with small hydrophobic residues at P2, aromatic at P9",
        "disease_associations": [
            "Protective against certain viral infections",
            "Associated with response to EBV antigens",
        ],
        "population_frequency": 0.15,
    },
    "A*02:01": {
        "binding_characteristics": "Broad peptide repertoire; favours leucine/methionine at P2, valine/leucine at P9",
        "disease_associations": [
            "Strong CTL responses to influenza and HIV epitopes",
            "Associated with type 1 diabetes susceptibility",
        ],
        "population_frequency": 0.25,
    },
    "A*03:01": {
        "binding_characteristics": "Binds peptides with small residues at P2, positively charged at P9",
        "disease_associations": [
            "Protective in certain HIV progressions",
            "Associated with multiple sclerosis risk",
        ],
        "population_frequency": 0.13,
    },
    "A*11:01": {
        "binding_characteristics": "Favours threonine/valine at P2, lysine at P9",
        "disease_associations": [
            "Common in East Asian populations",
            "Associated with nasopharyngeal carcinoma susceptibility",
        ],
        "population_frequency": 0.10,
    },
    "A*24:02": {
        "binding_characteristics": "Prefers tyrosine/phenylalanine at P2, leucine/phenylalanine at P9",
        "disease_associations": [
            "Associated with type 1 diabetes in Japanese populations",
            "Strong CTL response to certain viral epitopes",
        ],
        "population_frequency": 0.11,
    },
    "B*07:02": {
        "binding_characteristics": "Prefers proline at P2, hydrophobic C-terminal anchor",
        "disease_associations": [
            "Associated with slower HIV progression",
            "Linked to iron overload susceptibility",
        ],
        "population_frequency": 0.12,
    },
    "B*08:01": {
        "binding_characteristics": "Favours small residues at P2 and P3, aromatic at P9",
        "disease_associations": [
            "Part of the 8.1 ancestral haplotype",
            "Associated with autoimmune conditions including coeliac disease",
            "Linked to myasthenia gravis susceptibility",
        ],
        "population_frequency": 0.09,
    },
    "B*15:01": {
        "binding_characteristics": "Binds peptides with glutamine at P2, phenylalanine/tyrosine at P9",
        "disease_associations": [
            "Associated with drug hypersensitivity reactions",
            "Linked to certain autoimmune conditions",
        ],
        "population_frequency": 0.06,
    },
    "B*35:01": {
        "binding_characteristics": "Prefers proline at P2, tyrosine at P9",
        "disease_associations": [
            "Associated with faster HIV progression",
            "Linked to cervical cancer susceptibility",
        ],
        "population_frequency": 0.08,
    },
    "B*44:02": {
        "binding_characteristics": "Favours glutamic acid at P2, hydrophobic residues at P9",
        "disease_associations": [
            "Protective in HIV infection",
            "Associated with ankylosing spondylitis (weaker than B*27)",
        ],
        "population_frequency": 0.08,
    },
    "B*44:03": {
        "binding_characteristics": "Very similar to B*44:02 but distinct peptide repertoire at position 116",
        "disease_associations": [
            "Differential T-cell recognition compared to B*44:02",
            "Relevant to transplant matching outcomes",
        ],
        "population_frequency": 0.05,
    },
    "B*27:05": {
        "binding_characteristics": "Binds arginine at P2, hydrophobic C-terminus",
        "disease_associations": [
            "Strongly associated with ankylosing spondylitis",
            "Protective against HIV progression",
        ],
        "population_frequency": 0.04,
    },
    "C*01:02": {
        "binding_characteristics": "Interacts with KIR receptors; presents limited peptide repertoire",
        "disease_associations": [
            "KIR-ligand interactions relevant to NK cell function",
        ],
        "population_frequency": 0.05,
    },
    "C*04:01": {
        "binding_characteristics": "Ligand for KIR2DL1; hydrophobic anchor preferences",
        "disease_associations": [
            "Modulates NK cell activity via KIR interactions",
            "Associated with HIV viral load set-point",
        ],
        "population_frequency": 0.09,
    },
    "C*07:01": {
        "binding_characteristics": "Broad peptide binding; interacts with KIR2DL2/3",
        "disease_associations": [
            "Associated with psoriasis susceptibility",
            "Modulates innate immune responses",
        ],
        "population_frequency": 0.14,
    },
    "C*07:02": {
        "binding_characteristics": "Similar repertoire to C*07:01 with distinct KIR interactions",
        "disease_associations": [
            "Linked to NK cell education differences",
        ],
        "population_frequency": 0.10,
    },
}


class HLAService:
    """Orchestrates HLA-based biological compatibility calculations.

    Workflow (for ``calculate_compatibility``):
      1. Decrypt both users' HLA data from the database.
      2. Count unique alleles across the combined pool.
      3. Calculate S_bio = (N_unique / N_total) * 100.
      4. Compute Heterozygosity Index = N_unique / N_total.
      5. Generate olfactory attraction prediction from heterozygosity.
      6. Produce a display tier for the match card.
    """

    # ── Public API ────────────────────────────────────────────────────────

    async def calculate_compatibility(
        self,
        user_a_id: str,
        user_b_id: str,
        db_session: AsyncSession,
    ) -> dict:
        """Calculate biological compatibility between two users.

        Parameters
        ----------
        user_a_id, user_b_id:
            UUIDs of the two users to compare.
        db_session:
            Active SQLAlchemy async session.

        Returns
        -------
        dict
            Full compatibility result including S_bio, heterozygosity,
            olfactory prediction, peptide analyses, and display tier.
        """
        log = logger.bind(user_a=user_a_id, user_b=user_b_id)
        log.info("compatibility_calculation_start")

        # Step 1 — Retrieve and decrypt HLA data for both users
        hla_a = await self._fetch_hla_data(user_a_id, db_session)
        hla_b = await self._fetch_hla_data(user_b_id, db_session)

        if hla_a is None or hla_b is None:
            missing = []
            if hla_a is None:
                missing.append(user_a_id)
            if hla_b is None:
                missing.append(user_b_id)
            log.warning("hla_data_missing", missing_users=missing)
            return {
                "compatible": False,
                "error": "hla_data_missing",
                "missing_users": missing,
            }

        alleles_a = self._extract_alleles(hla_a)
        alleles_b = self._extract_alleles(hla_b)

        log.info(
            "alleles_extracted",
            count_a=len(alleles_a),
            count_b=len(alleles_b),
        )

        # Step 2-3 — Calculate S_bio
        s_bio = self._calculate_s_bio(alleles_a, alleles_b)

        # Step 4 — Heterozygosity Index
        heterozygosity = self._calculate_heterozygosity(alleles_a, alleles_b)

        # Step 5 — Olfactory attraction prediction
        olfactory = self._predict_olfactory_attraction(heterozygosity)

        # Peptide analyses for each user
        peptide_a = self._generate_peptide_analysis(alleles_a)
        peptide_b = self._generate_peptide_analysis(alleles_b)

        # Step 6 — Display tier
        display = self._get_display_tier(s_bio)

        result = {
            "user_a_id": user_a_id,
            "user_b_id": user_b_id,
            "s_bio": round(s_bio, 4),
            "heterozygosity_index": round(heterozygosity, 4),
            "olfactory_prediction": olfactory,
            "peptide_analysis": {
                "user_a": peptide_a,
                "user_b": peptide_b,
            },
            "display": display,
            "allele_summary": {
                "user_a_count": len(alleles_a),
                "user_b_count": len(alleles_b),
                "combined_unique": len(set(alleles_a + alleles_b)),
                "total_slots": _TOTAL_ALLELE_SLOTS,
            },
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }

        log.info(
            "compatibility_calculation_complete",
            s_bio=round(s_bio, 4),
            heterozygosity=round(heterozygosity, 4),
            display_tier=display.get("tier"),
        )

        return result

    async def upload_hla_data(
        self,
        user_id: str,
        alleles: dict,
        source: str,
        db_session: AsyncSession,
    ) -> dict:
        """Validate, encrypt, and store HLA allele data for a user.

        Parameters
        ----------
        user_id:
            UUID of the user.
        alleles:
            Dict mapping locus names to allele lists, e.g.::

                {
                    "A": ["A*01:01", "A*02:01"],
                    "B": ["B*07:02", "B*44:02"],
                    "C": ["C*07:01", "C*04:01"],
                }

        source:
            Data provenance string, e.g. ``"23andMe_v5"``.
        db_session:
            Active SQLAlchemy async session.

        Returns
        -------
        dict
            Confirmation including the record ID and validation summary.
        """
        log = logger.bind(user_id=user_id, source=source)
        log.info("hla_upload_start")

        # Validate allele format
        validation_errors: list[str] = []
        all_alleles: list[str] = []
        snp_count = 0

        for locus, locus_alleles in alleles.items():
            if not isinstance(locus_alleles, list):
                validation_errors.append(
                    f"Locus '{locus}' value must be a list, got {type(locus_alleles).__name__}"
                )
                continue

            for allele in locus_alleles:
                if not self._validate_allele_format(allele):
                    validation_errors.append(
                        f"Invalid allele format: '{allele}' — expected pattern like 'A*01:01'"
                    )
                else:
                    all_alleles.append(allele)
                    snp_count += 1

        if validation_errors:
            log.warning("hla_validation_failed", errors=validation_errors)
            return {
                "success": False,
                "errors": validation_errors,
            }

        if not all_alleles:
            log.warning("hla_no_alleles")
            return {
                "success": False,
                "errors": ["No valid alleles provided"],
            }

        # Prepare payload for encryption
        hla_payload = {
            "alleles": alleles,
            "all_alleles": all_alleles,
            "loci": list(alleles.keys()),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }

        # Encrypt
        encrypted = encrypt_hla_data(hla_payload)

        # Calculate imputation confidence based on completeness
        expected_alleles = len(alleles) * _ALLELES_PER_LOCUS
        actual_alleles = len(all_alleles)
        imputation_confidence = min(1.0, actual_alleles / max(1, expected_alleles))

        # Check for existing record (upsert logic)
        stmt = select(HLAData).where(HLAData.user_id == user_id)
        result = await db_session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing is not None:
            existing.encrypted_data = encrypted
            existing.source = source
            existing.imputation_confidence = imputation_confidence
            existing.snp_count = snp_count
            existing.ancestry_model = None
            record_id = str(existing.id)
            log.info("hla_record_updated", record_id=record_id)
        else:
            new_record = HLAData(
                user_id=user_id,
                encrypted_data=encrypted,
                source=source,
                imputation_confidence=imputation_confidence,
                snp_count=snp_count,
                ancestry_model=None,
            )
            db_session.add(new_record)
            await db_session.flush()
            record_id = str(new_record.id)
            log.info("hla_record_created", record_id=record_id)

        return {
            "success": True,
            "record_id": record_id,
            "user_id": user_id,
            "source": source,
            "allele_count": len(all_alleles),
            "loci": list(alleles.keys()),
            "imputation_confidence": round(imputation_confidence, 4),
        }

    # ── Core calculation methods ──────────────────────────────────────────

    def _calculate_s_bio(
        self,
        alleles_a: list[str],
        alleles_b: list[str],
    ) -> float:
        """Calculate the biological compatibility score (S_bio).

        Formula:  ``S_bio = (N_unique / N_total) * 100``

        Where:
          - N_unique is the count of unique alleles across both users' combined pool.
          - N_total is the total number of allele slots (typically 12 for
            3 loci x 2 alleles x 2 users).

        Parameters
        ----------
        alleles_a, alleles_b:
            Flat lists of allele identifiers for each user.

        Returns
        -------
        float
            S_bio score in range [0, 100].
        """
        combined = alleles_a + alleles_b
        n_total = max(len(combined), _TOTAL_ALLELE_SLOTS)
        n_unique = len(set(combined))

        if n_total == 0:
            return 0.0

        s_bio = (n_unique / n_total) * 100.0
        return min(100.0, s_bio)

    def _calculate_heterozygosity(
        self,
        alleles_a: list[str],
        alleles_b: list[str],
    ) -> float:
        """Calculate the Heterozygosity Index for a pair.

        Formula:  ``H = N_unique / N_total``

        Higher heterozygosity correlates with broader immune repertoire
        and, per the MHC-mediated mate choice hypothesis, increased
        olfactory attraction signals.

        Parameters
        ----------
        alleles_a, alleles_b:
            Flat lists of allele identifiers for each user.

        Returns
        -------
        float
            Heterozygosity Index in range [0, 1].
        """
        combined = alleles_a + alleles_b
        n_total = max(len(combined), _TOTAL_ALLELE_SLOTS)
        n_unique = len(set(combined))

        if n_total == 0:
            return 0.0

        return min(1.0, n_unique / n_total)

    def _predict_olfactory_attraction(
        self,
        heterozygosity_index: float,
    ) -> dict:
        """Predict olfactory attraction intensity from the Heterozygosity Index.

        Based on MHC-mediated mate-choice research: greater HLA
        dissimilarity (higher heterozygosity) correlates with stronger
        body-odour attractiveness signals.

        Parameters
        ----------
        heterozygosity_index:
            The H value in [0, 1].

        Returns
        -------
        dict
            ``{"intensity": int, "assessment": str}`` where intensity
            is a score from 0-100.
        """
        if heterozygosity_index >= 0.90:
            return {
                "intensity": 95,
                "assessment": "Exceptionally strong biological attraction signal",
            }
        elif heterozygosity_index >= 0.80:
            return {
                "intensity": 80,
                "assessment": "Strong biological attraction signal",
            }
        elif heterozygosity_index >= 0.70:
            return {
                "intensity": 65,
                "assessment": "Moderate biological attraction signal",
            }
        elif heterozygosity_index >= 0.60:
            return {
                "intensity": 45,
                "assessment": "Mild biological attraction signal",
            }
        else:
            return {
                "intensity": 25,
                "assessment": "Weak biological attraction signal",
            }

    def _generate_peptide_analysis(
        self,
        alleles: list[str],
    ) -> dict:
        """Generate peptide-binding and disease-association analysis for a set
        of alleles.

        Maps each allele to its known binding characteristics and disease
        associations from the curated reference dictionary.  Alleles without
        a reference entry are reported with a generic description.

        Parameters
        ----------
        alleles:
            List of allele identifiers, e.g. ``["A*01:01", "B*07:02"]``.

        Returns
        -------
        dict
            ``{"alleles": [...], "coverage": float, "summary": str}``
            where each allele entry contains binding and disease info.
        """
        allele_details: list[dict] = []
        known_count = 0

        for allele in alleles:
            ref = _HLA_REFERENCE.get(allele)
            if ref is not None:
                known_count += 1
                allele_details.append({
                    "allele": allele,
                    "known": True,
                    "binding_characteristics": ref["binding_characteristics"],
                    "disease_associations": ref["disease_associations"],
                    "population_frequency": ref["population_frequency"],
                })
            else:
                allele_details.append({
                    "allele": allele,
                    "known": False,
                    "binding_characteristics": "Binding profile not in reference database",
                    "disease_associations": [],
                    "population_frequency": None,
                })

        coverage = known_count / max(1, len(alleles))

        if coverage >= 0.8:
            summary = "Comprehensive peptide-binding profile available"
        elif coverage >= 0.5:
            summary = "Partial peptide-binding profile available"
        else:
            summary = "Limited reference data for these alleles"

        return {
            "alleles": allele_details,
            "coverage": round(coverage, 4),
            "summary": summary,
        }

    def _get_display_tier(self, s_bio: float) -> dict:
        """Determine the display tier for the match card based on S_bio.

        Display tiers control whether and how the chemistry signal is
        shown to the user on the match card.

        Parameters
        ----------
        s_bio:
            Biological compatibility score in [0, 100].

        Returns
        -------
        dict
            Display metadata including show/hide decision, tier label,
            emoji, and human-readable label.
        """
        if s_bio >= 75.0:
            return {
                "show": True,
                "tier": "strong",
                "emoji": "\U0001f525",  # fire
                "label": "Strong chemistry signal",
            }
        elif s_bio >= 50.0:
            return {
                "show": True,
                "tier": "good",
                "emoji": "\u2728",  # sparkles
                "label": "Good chemistry",
            }
        elif s_bio >= 25.0:
            return {
                "show": True,
                "tier": "some",
                "emoji": "\U0001f4ab",  # dizzy / shooting star
                "label": "Some chemistry",
            }
        else:
            return {
                "show": False,
                "tier": "hidden",
            }

    def _validate_allele_format(self, allele: str) -> bool:
        """Validate that an allele string matches the expected HLA format.

        Expected format examples:
          - ``A*01:01``
          - ``B*44:02``
          - ``C*07:01``
          - ``DRB1*15:01``

        Parameters
        ----------
        allele:
            The allele identifier string.

        Returns
        -------
        bool
            True if the format is valid.
        """
        if not isinstance(allele, str):
            return False
        return _ALLELE_PATTERN.match(allele) is not None

    # ── Private helpers ───────────────────────────────────────────────────

    async def _fetch_hla_data(
        self,
        user_id: str,
        db_session: AsyncSession,
    ) -> dict | None:
        """Retrieve and decrypt a user's HLA data from the database.

        Parameters
        ----------
        user_id:
            UUID of the user.
        db_session:
            Active SQLAlchemy async session.

        Returns
        -------
        dict or None
            The decrypted HLA payload, or None if no record exists.
        """
        stmt = select(HLAData).where(HLAData.user_id == user_id)
        result = await db_session.execute(stmt)
        record = result.scalar_one_or_none()

        if record is None:
            return None

        try:
            decrypted = decrypt_hla_data(record.encrypted_data)
            return decrypted
        except Exception:
            logger.exception(
                "hla_decryption_failed",
                user_id=user_id,
            )
            return None

    @staticmethod
    def _extract_alleles(hla_data: dict) -> list[str]:
        """Extract a flat list of alleles from a decrypted HLA payload.

        The payload may store alleles in either of two layouts:

        1. Pre-flattened: ``{"all_alleles": ["A*01:01", ...]}``
        2. Per-locus:     ``{"alleles": {"A": ["A*01:01", ...], ...}}``

        Parameters
        ----------
        hla_data:
            Decrypted HLA payload dict.

        Returns
        -------
        list[str]
            Flat list of allele identifiers.
        """
        # Prefer the pre-flattened list if present
        if "all_alleles" in hla_data and isinstance(hla_data["all_alleles"], list):
            return hla_data["all_alleles"]

        # Fall back to per-locus extraction
        alleles: list[str] = []
        locus_data = hla_data.get("alleles", {})
        if isinstance(locus_data, dict):
            for locus_alleles in locus_data.values():
                if isinstance(locus_alleles, list):
                    alleles.extend(locus_alleles)

        return alleles
