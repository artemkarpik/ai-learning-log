## Day 1 — Nov 2–3

Learned:
- Built a bigram model: a 27x27 table counting how often each letter follows another, then turned each row into probabilities.
- `chs[1:]` is slicing (the list minus its first item), and `zip(chs, chs[1:])` gives every pair of neighbors.
- `sum(1)` adds across each row; `keepdim=True` keeps it as a column so each row divides by its own total.
- `nan` means 0 ÷ 0. Mine happened because a cell hadn't been run, so N was all zeros.

## Day 2 — Nov 3

So the keepdim and efficiency vectorized normalization thing does this: It takes the original 27x27 matrix and divides it by a matrix where the rows have all been added up and squashed into one single 27x1 column, and then each of the original 27x27 entries is getting divided by the total number in that column because when the division happens the column entry get's stretched over the amount of rows

the inf is caused because there is no jq data point, and then after using smoothing and adding 1 to every data point, technically every letter is possible after every letter with extremely low likelihood. I am given 3.4 which is quite bad but possible

