# DiveGuard — hydrophone_module — Documentation Index

This repo implements **Phases 1-4** of the DiveGuard propeller-detection system
(acoustic acquisition + DSP core). It is one of three repos that together make
up the full DiveGuard project — see the cross-repo index in each repo's
`DOCUMENTATION_INDEX.md`, or the consolidated copy delivered to the user.

## Architecture & Design Docs
| File | Contents |
|---|---|
| `DiveGuard_HLD.md` | 12-phase High-Level Design, expert review notes |
| `DiveGuard_TECHNICAL_SPEC.md` | Detailed technical specification |
| `CLOUD_DECISION_FRAMEWORK.md` | 32-specialist ensemble decision framework (linear spectrogram vs MFCC, DEMON vs spectral subtraction, etc.) |
| `ACOUSTIC_MASKING_SOLUTION.md` | 6-layer discrimination system (propeller vs marine mammal vs ambient noise) |
| `ACOUSTIC_DATABASES_RU.md` | Acoustic dataset reference (ShipsEar, DeepShip, MMSD) — RU |

## User-Facing Docs
| File | Contents |
|---|---|
| `README.md` | Primary English README |
| `README_RU_FULL.md` | Full Russian documentation (architecture, API, testing, limitations) |
| `PROPELLER_DETECTOR_README.md` | Propeller detector overview (EN) |
| `PROPELLER_DETECTOR_RU.md` | Propeller detector overview (RU) |
| `INSTALLATION_RU.md` | 5-minute quick start + module walkthrough (RU) |
| `INDEX_RU.md` | Legacy RU file-by-file index (superseded by this file for full coverage) |

## Code
| File | Phase | Contents |
|---|---|---|
| `src/dsp_core.hpp` | 2-4 | C++ lock-free ring buffer, LOFAR, DEMON |
| `dsp_bridge.py` | 1-4 | Python/ZMQ bridge to C++ DSP server, `ALSAHydrophoneReader` |
| `phases_1_4_implementation.py` | 1-4 | Pure-Python reference implementation (for testing without C++ build) |
| `sensor_fusion.py` | — | Extended Kalman Filter, SONAR+hydrophone fusion |
| `propeller_classifier.py` | 5-8 (consumer) | Classifier interface consumed by ML pipeline in `ml-camera-backend` |
| `threat_assessment.py` | — | Threat scoring (1-10 scale) |
| `diver_alert_controller.py` | — | Diver alert dispatch logic |
| `main_integration.py` | — | Top-level integration entrypoint |

## CI
`.github/workflows/ci.yml` — Python syntax compile check + `ruff` lint (errors/undefined-names only).

## Related Repos
- **ml-camera-backend**: Phases 5-8 (ML training/quantization) + Phase 12A/B/C (video enhancement testing)
- **underwater-ai-platform**: Phases 9-11 (Docker/FastAPI/Vue dashboard) + Phase 12A integration

## Known Gaps (tracked, being closed on the 2-hour audit cycle)
- No unit test suite (only integration-style scripts) — CI currently only validates syntax, not behavior.
- No cross-repo automated test running the full Phase 1→12 pipeline end-to-end.
