"""cppmega_mlx CLI entrypoints.

Each module exposes a ``main()`` function callable from
``python -m cppmega_mlx.cli.<name>`` for end-user orchestration of
single-Mac, loopback, and multi-mac smoke receipts.

Currently shipping:
  - smoke_zero1 — ZeRO-1 distributed-wrapper receipt (loopback or
    multi-host) via ``mlx.launch``. V7-Q05 closure of
    ``docs/distributed_zero1_smoke_procedure.md`` "not yet implemented".
"""
