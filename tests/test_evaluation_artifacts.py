"""#306 — Evaluation artifact completeness and consistency validation.

Proves:
  1. All required artifact files exist
  2. Criteria count is 8 (not stale 6)
  3. Hard requirements count is 3
  4. Scenario types count is 6
  5. Thresholds match ADR-014: 20+ questions, 90% citation, no silent fallback
  6. Internal consistency with ADR-014

Run: cd agentopia-protocol/bot-config-api/src
     python -m pytest tests/test_evaluation_artifacts.py -v
"""

from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).parent.parent.parent.parent / "docs" / "evaluation"
ADR_DIR = Path(__file__).parent.parent.parent.parent.parent / "docs" / "adrs"

# These tests validate agentopia-protocol/docs/evaluation/ artifacts.
# In the standalone super-rag repo those files don't exist — skip the whole module.
pytestmark = pytest.mark.skipif(
    not EVAL_DIR.exists(),
    reason="agentopia-protocol docs/evaluation/ not present (standalone repo checkout)",
)


# ── 1. All required files exist ─────────────────────────────────────────────


class TestArtifactFilesExist:
    def test_checklist_exists(self):
        assert (EVAL_DIR / "checklist.md").exists()

    def test_scenario_matrix_exists(self):
        assert (EVAL_DIR / "scenario-matrix.md").exists()

    def test_question_set_template_exists(self):
        assert (EVAL_DIR / "question-set-template.md").exists()

    def test_sample_fixtures_exists(self):
        assert (EVAL_DIR / "sample-fixtures.md").exists()

    def test_setup_guide_exists(self):
        assert (EVAL_DIR / "test-setup-guide.md").exists()

    def test_go_live_report_template_exists(self):
        assert (EVAL_DIR / "go-live-report-template.md").exists()


# ── 2. Criteria count is 8 ─────────────────────────────────────────────────


class TestCriteriaCount:
    def test_checklist_has_8_criteria(self):
        """Checklist must have exactly 8 numbered criteria rows."""
        content = (EVAL_DIR / "checklist.md").read_text()
        # Count rows matching "| N |" pattern in the criteria table
        import re
        criteria_rows = re.findall(r"^\|\s*\d+\s*\|", content, re.MULTILINE)
        assert len(criteria_rows) == 8, f"Expected 8 criteria, found {len(criteria_rows)}"

    def test_go_live_report_has_8_criteria(self):
        """Go-live report template must have 8 criteria result rows."""
        content = (EVAL_DIR / "go-live-report-template.md").read_text()
        import re
        criteria_rows = re.findall(r"^\|\s*\d+\s*\|", content, re.MULTILINE)
        assert len(criteria_rows) >= 8


# ── 3. Hard requirements count is 3 ────────────────────────────────────────


class TestHardRequirements:
    def test_checklist_has_3_hard_criteria_rows(self):
        """Checklist must mark exactly 3 criteria rows as HARD."""
        content = (EVAL_DIR / "checklist.md").read_text()
        import re
        # Count table rows that contain both a criterion number and **HARD**
        hard_rows = re.findall(r"^\|\s*\d+\s*\|\s*\*\*HARD\*\*", content, re.MULTILINE)
        assert len(hard_rows) == 3, f"Expected 3 HARD criteria rows, found {len(hard_rows)}"

    def test_hard_requirements_are_4_5_6(self):
        """Hard requirements must be criteria 4 (scope isolation), 5 (fabricated citations), 6 (unavailability)."""
        content = (EVAL_DIR / "checklist.md").read_text()
        assert "| 4 | **HARD**" in content
        assert "| 5 | **HARD**" in content
        assert "| 6 | **HARD**" in content

    def test_hard_requirements_are_100_percent(self):
        """Hard requirements must specify 100% threshold."""
        content = (EVAL_DIR / "checklist.md").read_text()
        # Each HARD row must contain "100%"
        import re
        hard_rows = [line for line in content.split("\n") if "**HARD**" in line]
        for row in hard_rows:
            assert "100%" in row, f"Hard requirement row missing 100%: {row}"


# ── 4. Scenario types count is 6 ───────────────────────────────────────────


class TestScenarioCount:
    def test_scenario_matrix_has_6_scenarios(self):
        """Scenario matrix must define exactly 6 scenario types."""
        content = (EVAL_DIR / "scenario-matrix.md").read_text()
        # Count "## Scenario N:" headers
        import re
        scenarios = re.findall(r"^## Scenario \d+:", content, re.MULTILINE)
        assert len(scenarios) == 6, f"Expected 6 scenarios, found {len(scenarios)}"

    def test_scenario_types_match_adr014(self):
        """Scenarios must cover the 6 required types from ADR-014."""
        content = (EVAL_DIR / "scenario-matrix.md").read_text().lower()
        assert "positive grounded" in content
        assert "empty retrieval" in content
        assert "retrieval timeout" in content or "timeout" in content
        assert "conflicting sources" in content
        assert "stale source" in content
        assert "scope isolation" in content


# ── 5. Thresholds match ADR-014 ────────────────────────────────────────────


class TestThresholds:
    def test_minimum_20_questions(self):
        """Checklist requires >= 20 questions."""
        content = (EVAL_DIR / "checklist.md").read_text()
        assert ">= 20" in content or "20 questions" in content

    def test_citation_accuracy_90_percent(self):
        """Citation accuracy threshold is >= 90%."""
        content = (EVAL_DIR / "checklist.md").read_text()
        assert ">= 90%" in content or "90%" in content

    def test_retrieval_relevance_80_percent(self):
        """Retrieval relevance threshold is >= 80%."""
        content = (EVAL_DIR / "checklist.md").read_text()
        assert ">= 80%" in content or "80%" in content

    def test_no_silent_fallback_principle(self):
        """Checklist references no-silent-fallback principle."""
        content = (EVAL_DIR / "checklist.md").read_text().lower()
        assert "silent fallback" in content or "no fabricated" in content

    def test_question_template_has_20_rows(self):
        """Question set template has >= 20 grounded question rows."""
        content = (EVAL_DIR / "question-set-template.md").read_text()
        import re
        grounded_rows = re.findall(r"^\|\s*\d+\s*\|.*grounded", content, re.MULTILINE)
        assert len(grounded_rows) >= 20, f"Expected >= 20 grounded rows, found {len(grounded_rows)}"


# ── 6. ADR-014 consistency ─────────────────────────────────────────────────


class TestADRConsistency:
    def test_adr014_exists(self):
        """ADR-014 source document exists (skipped in CI where monorepo parent is absent)."""
        adr = ADR_DIR / "014-sa-kb-quality-evaluation-answer-contract.md"
        if not ADR_DIR.exists():
            pytest.skip("ADR directory not available (CI checkout is repo-only)")
        assert adr.exists(), f"ADR-014 not found at {adr}"

    def test_checklist_references_adr014(self):
        """Checklist references ADR-014 as source."""
        content = (EVAL_DIR / "checklist.md").read_text()
        assert "ADR-014" in content

    def test_scenario_matrix_references_adr014(self):
        """Scenario matrix references ADR-014 or D7."""
        content = (EVAL_DIR / "scenario-matrix.md").read_text()
        assert "ADR-014" in content or "D7" in content

    def test_answer_contract_in_checklist(self):
        """Checklist includes answer contract verification section."""
        content = (EVAL_DIR / "checklist.md").read_text()
        assert "Answer Contract" in content
        assert "Citation Rules" in content
