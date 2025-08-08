# Tensor API Overview

This document outlines the prototype Tensor interface for Apple-Metal Orchard. It covers dtype and device handling, unified memory storage backed by page-aligned `MTLBuffer`, and basic view/slice utilities.

## Memory Model
Tensors allocate 64-byte aligned buffers via `Storage`. On Apple platforms this wraps `MTLBuffer` with `StorageModeShared`, allowing CPU and GPU to share memory with no copies. For large buffers (>16 MiB) the implementation is prepared to upgrade to IOSurface in future revisions.

## Threading and Mutability
Each `TensorImpl` owns an `os_unfair_lock` (or `std::mutex` when not on Apple) protecting mutations. Read‑only operations are lock free and multiple views share the same underlying storage via intrusive reference counting.

## Profiling
When `ORCHARD_PROFILE_ALLOC` is defined, `ArenaAllocator` logs every allocation and free to `/tmp/orchard_tensor_profile.log`. The log includes byte size and device so developers can trace leaks and high‑water marks.
