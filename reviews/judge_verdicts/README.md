# Verification Infrastructure & Agent-as-a-Judge Artifacts

This directory contains raw, unedited evaluation logs generated during the development of `finance`. The codebase was built using an iterative, spec-driven agentic loop:

$$\text{Plan (human)} \longrightarrow \text{Spec (machine)} \longrightarrow \text{Do-While}(\text{Implement}, \text{Judge}) \longrightarrow \text{Ship}$$

### Protocol Architecture & Artifact Types

- **`F-XXX_pm_verdict_round_N.txt` (Spec Judge):** Validates implementation diffs against machine-readable specifications (`portfolio_management_spec.json`) without access to implementation reasoning traces. Specifically tests for spec drift and missing boundary requirements.
- **`F-XXX_verdict_round_N.txt` (Technical Judge):** Validates code quality, edge-case coverage, and unit/property test assumptions.
- **Multi-Round Traces (`round_1`, `round_2`, `round_3`):** Demonstrate automated re-evaluation loops following agentic remediation of advisories and boundary failures.

These artifacts serve as the machine-verifiable evidentiary packet for the library's verification harness.
