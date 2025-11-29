## Networking Features & API Enhancements

- Expanded `esp_netif` capabilities:
  - Added events for SNTP time updates, netif status, and custom GOT_IP/LOST_IP flows, enabling event-driven handling instead of polling.
  - Introduced backward‑compat options to ease migration for external components.
  - Cleaned up deprecated APIs and simplified hostname reporting.
- Extended lwIP functionality:
  - Enabled POSIX `NETIF_API` unconditionally.
  - Added support for netif link status callbacks.
  - Added DHCP server support for reporting client hostnames.
  - Introduced a framework to test network performance under controlled packet loss.
- Other subsystems:
  - Added an MQTT local‑broker publish stress test.
  - Introduced a “minimum IRAM” iperf test configuration.
  - Provided hints for using external socket utilities to simplify integration.

**Impact:** Applications gain richer, event-driven network state awareness and better tools for testing performance, stress, and constrained-memory scenarios.

---

## Stability, Safety & Correctness Fixes

### Core networking components

- `esp_netif`
  - Corrected PPP input return values for accurate error reporting.
  - Fixed sequencing/consistency of GOT_IP vs LOST_IP notifications.
  - Ensured proper callback invocation for IPv6 MLD and IGMP after initialization.
  - Added status event support and compatibility options to avoid regressions during API evolution.
  - Reduced log noise by pushing tracing logs from debug to verbose.

- lwIP and DHCP
  - Fixed memory handling in the DHCP server (proper `mem_alloc`/`mem_free` pairing).
  - Ensured DHCP server callbacks are only called when configured.
  - Improved DHCP option construction (hardware ID) and clarified conflict detection, with tests added.
  - Reduced heap fragmentation by allocating some synchronization primitives once.
  - Cleaned up unused config flags and removed deprecated ping wrappers.
  - Corrected IPv6 raw-socket checksum handling in IPv6-only builds.

- Wi‑Fi, Ethernet, and OpenThread
  - Adjusted Wi‑Fi station list handling to use dynamically sized storage.
  - Fixed Ethernet test code to properly unregister events and avoid leaks or cross‑test interference.
  - Updated OpenThread netif layers to return explicit error codes, improving diagnostics and error handling.

**Impact:** Networking subsystems are more robust under edge conditions (memory pressure, IPv6-only configs, DHCP corner cases), with better error propagation and fewer potential crashes or leaks.

---

## Testing & CI Improvements

- CI and test infrastructure:
  - Refined CI `depends` patterns for netif test apps, improving selection of relevant tests.
  - Added network stress/performance tests:
    - MQTT publish stress test against a local broker.
    - lwIP packet-loss performance tests.
    - iperf “minimum IRAM” configuration for footprint-sensitive scenarios.

**Impact:** Test coverage for networking behavior under load, packet loss, and constrained resources is broader and more realistic, increasing confidence in releases.

---

## Example & Sample Application Fixes

- Networking examples:
  - Fixed Ethernet–Wi‑Fi bridge example configuration to ensure it runs as documented.
  - Switched the MTU example to use the minimal build profile for reliable builds in constrained environments.
  - Updated comments and guidance in the STA–Ethernet bridge example to be clearer and less ambiguous.

**Impact:** Example projects now build and run more reliably, and serve as more accurate references for users integrating similar functionality.

---

## Documentation & Developer Experience

- Documentation updates:
  - Clarified `sdkconfig.rename` usage.
  - Documented lwIP task name changes in migration guides.
  - Improved example comments and hints around networking utilities and bridge usage.

- Error and header maintenance:
  - Regenerated `esp_err_to_name` mappings after lwIP header cleanup.
  - Removed reliance on deprecated `sntp.h` in `esp_netif`, aligning with current time‑sync APIs.

**Impact:** Migration paths and configuration workflows are clearer, error diagnostics remain accurate, and deprecated dependencies are being systematically removed.
