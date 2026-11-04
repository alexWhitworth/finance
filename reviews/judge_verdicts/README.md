# Verification Infrastructure & Agent-as-a-Judge Artifacts

This directory contains raw, unedited Agent-as-a-Judge evaluation logs generated during the development of `finance`. The codebase was built using an iterative, spec-driven agentic loop:

$$\text{Plan (human)} \longrightarrow \text{Spec (machine)} \longrightarrow \text{Do-While}(\text{Implement}, \text{Judge}) \longrightarrow \text{Ship}$$

These artifacts serve as the machine-verifiable evidentiary packet for the library's verification harness.

### File naming convention

- `<prefix>-F-<XXX>_verdict_round_k.txt` where
    - prefix denotes a particualr spec/plan set of features
    - XXX denotes the feature number
    - and k denotes the judging round