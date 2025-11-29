## Overview

Since 2024‑12‑01 the work has focused on:  
- Strengthening modem and MQTT/Mosquitto functionality and examples  
- A major mDNS refactor with new modules, fuzzing, and extensive tests  
- Upgrading and de‑customizing Asio and networking utilities  
- Broad CI/venv standardization and tooling maintenance  
- Documentation improvements and configuration clean‑ups

Below is a thematic summary.

---

## Modem & PPP Stack

**Feature work**

- Introduced richer URC (Unsolicited Result Code) handling:
  - New “enhanced” URC interfaces, state‑aware URC logic, and an enhanced observer API.
  - Examples migrated to the new URC format and APIs, with supporting tests.
- Improved sample applications:
  - Mode detection added to PPP examples, including correct PPP mode detection for LCP/conf.
  - AT‑based modem example extended for multiple simultaneous connections.
  - TCP client example updated to work with ESP‑AT–based modems.
- Expanded simulation capabilities:
  - Added an ESP‑AT–based `modem_sim` module and initial simulator documentation.
  - Improved modem_sim handling of PPPD exit to cleanly react to PPP termination.

**Stability, build, and config**

- Numerous build and dependency fixes for modem across IDF/kernel versions (e.g., driver dependencies on v6.0, release‑branch CI publish paths, removal of unused Kconfig, reduced warning noise).
- Clean‑up of TCP client internals and console dependency wiring.
- Adjustments to download tooling (replacing deprecated actions) to restore reliable modem image downloads.
- Small workflow/quality‑of‑life improvements, such as removing release‑branch‑specific pre‑commit checks.

**Documentation**

- New migration manual and updated docs for modem v2.0.
- Clarified URC handler behavior and URC enhancement, aligning examples and docs with the new URC model.
- Temporary addition of coding‑assistant integration files to keep workflows unblocked.

**Impact:**  
Modem integration is now more robust and testable, with better URC semantics, improved PPP behavior, and realistic simulation support, all backed by clearer documentation.

---

## MQTT, Mosquitto, and MQTT‑related Components

**Features**

- Mosq:
  - Added an `on-message` callback hook for received messages with corresponding API documentation and consistency checks between API docs and versions.
  - Introduced basic MQTT authentication support, including optional username/password handling in examples.
- Modem & examples:
  - Made the MQTT public broker endpoint configurable and updated examples to point to a more reliable public broker.

**Dependency and build hygiene**

- Systematically moved MQTT use into **optional dependencies**:
  - Examples, modem, mosq, and various components (`eppp`, `console`, `mqtt_cxx`) now declare MQTT/esp‑mqtt dependencies explicitly and optionally, reducing unnecessary coupling.
- Modernized Mosq internals:
  - Replaced local socket stubs with shared `sock_utils`.
  - Fixed esp_webRTC usage for new FreeRTOS APIs.
  - Corrected version checks and updated pytest to use current embedded packages.
- Removed temporary modifications to IDF files and added notes on stack size requirements so developers can size tasks correctly.

**Testing & CI**

- Added IDF MQTT stress tests into Mosquitto CI.
- Scoped expensive IDF build/version checks to master or labeled jobs to control CI cost.
- Ensured Mosquitto tests run in modern Python virtualenvs (e.g., py‑venv3.12).

**Impact:**  
MQTT‑related code is more modular, configurable, and secure, with better dependency modeling and more realistic CI coverage under load and new Python/IDF versions.

---

## mDNS Refactor, Features, and Robustness

**Major architecture refactor**

- Performed a staged refactor of mDNS:
  - Introduced new modules: responder, service, mdns‑debug, packet‑tx vs browse separation, action queue abstraction, and a querier implementation.
  - Cleaned up and split interfaces between modules, renamed and reorganized internals to improve modularity.
- Extended capabilities:
  - Implemented querier search enhancements and responder PCB management.
  - Added support to allocate memory with configurable capability flags, and CI checks to ensure allocator usage follows project rules.

**Reliability and correctness**

- Fixed critical race and lifecycle issues:
  - Forward‑ported a delete‑race fix.
  - Corrected task creation/deletion and static‑task test configurations.
  - Stabilized builds against ESP‑IDF master and newer `kconfiglib`.
- Numerous small correctness and robustness fixes:
  - Cleaner querier, PCB, sender, packet parser, and debug logging (including behavior with small buffers).
  - Host tests updated for correct include paths and pinned tooling (`idf-build-apps`).
  - Adjusted AFL mocks to match upstream ESP‑IDF changes.

**Testing, fuzzing, and coverage**

- Expanded and modernized testing:
  - New unit tests for host, receiver, delegated answers, and TXT handling (bool/NULL values).
  - Added a test template and tests around `mdns-sender`.
  - Refactored fuzzing suites and added AFL++ fuzzing into CI, now using tagged images for reproducibility.
- CI‑level hardening:
  - Added checks for forbidden std allocators in mDNS sources.
  - Integrated fuzzing jobs for early crash/edge‑case detection.

**Documentation**

- Added a data‑flow diagram and detailed refactor documentation to support reviewers.
- Updated docs to explain the refactor and new architecture.

**Impact:**  
mDNS has moved from a monolithic implementation to a modular, fuzz‑hardened subsystem with richer tooling and tests, significantly improving maintainability and resilience.

---

## Asio, WebSocket, TLS, and DNS

**Asio**

- Upgraded Asio to 1.32 and removed custom esp/asio patches in favor of the shared `sock-utils` layer.
- Ensured compatibility and coverage with ESP‑IDF v5.4 through new tests.
- Fixed networking and examples:
  - Enabled `if_nametoindex` to resolve linking issues.
  - Refreshed TLS certificates in the TLS server‑client example to fix runtime failures.
  - Corrected chat example output to print only message bodies.
  - Ensured target tests run inside Python virtualenvs for reproducibility.

**WebSocket**

- Improved runtime robustness by correctly propagating timeouts and transport‑level errors from the TCP transport.
- Switched WebSocket target tests to run in a dedicated virtualenv and fixed pytest logic to properly validate clients.

**TLS / mbedtls**

- Enabled cookie support in `mbedtls_cxx` so DTLS and related flows relying on cookies work correctly.
- Removed obsolete warning suppressions from `tls_cxx` to surface real issues.

**DNS**

- Hardened the DNS HTTP event handler to gracefully handle default/fallback event types, improving resilience to new/unknown events.

**Impact:**  
Networking and transport components are now on current libraries, share a common socket utility layer, and have more reliable examples and tests across IDF versions.

---

## EPPP & Related Networking Examples

**Features**

- Added support for:
  - Custom channels in EPPP.
  - Transport over Ethernet links.
- Improved error handling:
  - TUN netif can now optionally propagate errors instead of silently ignoring them.

**Compatibility & testing**

- Updated UART and driver dependency handling for newer ESP‑IDF versions.
- Fixed PPP‑related link issues when PPP is disabled in lwIP.
- Expanded build‑matrix coverage to test more EPPP combinations and fixed driver‑related test dependencies.

**Examples**

- Multi‑netif example updated to work with `DNS_PER_DEFAULT_NETIF`.
- SLIP netif example is now executed on actual target hardware as part of CI.

**Impact:**  
EPPP is more flexible and robust across configurations and transports, with better error visibility and CI coverage.

---

## General Examples & Component Compatibility

- Ethernet examples:
  - Switched to `eth-phy-generic` on newer IDF versions to stay aligned with supported PHYs.
  - Console Ethernet initialization workarounds added to insulate against IDF behavior changes.
- LWS:
  - Removed LWS support for IDF ≥ 6.0 to avoid unsupported configurations.
  - Fixed extensive clang‑tidy “file not found” issues and added missing license information.
- Various examples updated to align with optional MQTT dependencies and new brokers.

**Impact:**  
Examples continue to compile and run across changing IDF versions, providing up‑to‑date reference implementations with fewer surprises.

---

## CI / Tooling / Virtual Environments

**Python virtualenv standardization**

- Consolidated Python venv setup into shared common workflows.
- Migrated a wide range of *target tests* (ASIO, sock_utils, WebSocket, mDNS, Mosquitto, and others) to run inside dedicated venvs, including explicit Python 3.12 environments where relevant.
- This improves dependency isolation and reduces environment‑specific flakiness.

**Clang‑tidy and static analysis**

- Updated common clang‑tidy runners to only use supported versions.
- Resolved path and macro‑collision issues (e.g., in LWS and sock_utils) to enable clean static analysis.
- Reformatted mDNS sources to the current style and wired pre‑commit hooks to correctly call `astyle`.

**CI infrastructure and scope**

- Removed deprecated Ubuntu 20.04 runners from CI.
- Tightened scope of expensive jobs:
  - Mosq IDF build and version check run only on master or labeled PRs.
  - Docs build and deployment limited to tagged components.
- Fixed multiple docs and CONTRIBUTING links across common CI.
- Added additional publish coverage for modem release branches.

**Impact:**  
CI is leaner, more reliable, and more uniform across repositories, with consistent tooling versions and better control of heavy jobs.

---

## Documentation & Compliance

- Improved and extended documentation for:
  - Modem (migration guides, v2.0 details, URC enhancements, simulator usage).
  - Mosq APIs (on‑message callback, stack size considerations).
  - mDNS (refactor overview, data‑flow diagrams, allocator usage, removal of outdated fuzz docs).
- Licensing and compliance:
  - Added missing license headers to LWS.
- General cleanup:
  - Numerous small doc and review‑driven cleanups across mDNS and other components.

**Impact:**  
Developers now have clearer guidance on using the modem, Mosq, and mDNS subsystems, while the codebase aligns better with licensing and documentation standards.

---

## Patterns & Overall Impact

- **Convergence on shared utilities and venvs:** Transition to `sock-utils` and common Python venv workflows reduces duplication and platform‑specific issues.
- **Strong focus on mDNS and modem robustness:** Both areas have seen deep refactors, new APIs, and extensive testing/fuzzing to make them production‑ready under evolving IDF/toolchain versions.
- **Better dependency modeling:** Across MQTT, console, modem, and examples, dependencies are more explicit and often optional, improving modularity.
- **CI as first‑class citizen:** CI changes systematically address flakiness, cost, and coverage, with modern runners, targeted jobs, and better tooling integration.

Overall, the work significantly improves reliability, maintainability, and feature richness of the networking stack while keeping pace with upstream changes in ESP‑IDF, FreeRTOS, and tooling.
