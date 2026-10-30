# Reading the IBM AML files: pandas or polars

`python benchmarks/io_engines.py ...` on Windows-11-10.0.26200-SP0 (Python 3.12.10);
`load_dataset` = read + canonical schema + validation + laundering patterns; best of
2 runs, each in a fresh process. Both engines give the same table.

| file | rows | engine | seconds | peak memory (MB) | of which interpreter + imports |
|---|---:|---|---:|---:|---:|
| HI-Small_Trans.csv | 5,078,345 | pandas | 17.3 | 2,592 | 90 |
| HI-Small_Trans.csv | 5,078,345 | polars | 12.8 | 4,227 | 90 |
| LI-Small_Trans.csv | 6,924,049 | pandas | 23.0 | 3,474 | 90 |
| LI-Small_Trans.csv | 6,924,049 | polars | 18.0 | 4,849 | 90 |
