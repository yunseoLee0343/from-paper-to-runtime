# GPU Runtime Batching Strategy

## Summary

`TensorRTRuntime.build_assignments()` now uses a bounded lookahead window instead of a full-table group-then-chunk pass.

## What Changed

- `TensorRTRuntime` accepts `lookahead_window` with a default of `8`.
- Requests are admitted into a small pending deque.
- The runtime counts bucket frequency inside that window and chooses the densest bucket first.
- Ties are broken by the static `BUCKETS` order so the policy stays deterministic.
- Selected requests are packed up to the bucket's `max_batch`, while non-matching requests stay in the queue for the next round.

## Why This Helps

The original implementation grouped the entire request table by bucket before chunking. That works, but it can over-separate near-term requests when a mixed-shape stream arrives in bursts.

The lookahead scheduler stays simple while improving two practical traits:

- better near-term batch density for the hottest bucket in a short request window
- more realistic runtime behavior for online serving, where the full future request set is not assumed to be visible

## Limits

This is still a lightweight host-side heuristic.

- It is not a global optimal packing algorithm.
- It does not model request deadlines or per-request latency SLAs.
- It keeps the existing bucket and stream model unchanged.

## Runtime Contract

The change is local to assignment construction.

- request normalization stays unchanged
- bucket selection legality stays unchanged
- stream dispatch and result ordering stay unchanged

That makes the patch safe to adopt without changing the TensorRT compile path or the result contract exposed by `execute()`.
