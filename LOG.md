## Day 1 — Nov 2–3

Learned:
- Built a bigram model: a 27x27 table counting how often each letter follows another, then turned each row into probabilities.
- `chs[1:]` is slicing (the list minus its first item), and `zip(chs, chs[1:])` gives every pair of neighbors.
- `sum(1)` adds across each row; `keepdim=True` keeps it as a column so each row divides by its own total.
- `nan` means 0 ÷ 0. Mine happened because a cell hadn't been run, so N was all zeros.
