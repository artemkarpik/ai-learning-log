## Day 1 — Nov 2–3

Learned:
- Built a bigram model: a 27x27 table counting how often each letter follows another, then turned each row into probabilities.
- `chs[1:]` is slicing (the list minus its first item), and `zip(chs, chs[1:])` gives every pair of neighbors.
- `sum(1)` adds across each row; `keepdim=True` keeps it as a column so each row divides by its own total.
- `nan` means 0 ÷ 0. Mine happened because a cell hadn't been run, so N was all zeros.

## Day 2 — Nov 3

So the keepdim and efficiency vectorized normalization thing does this: It takes the original 27x27 matrix and divides it by a matrix where the rows have all been added up and squashed into one single 27x1 column, and then each of the original 27x27 entries is getting divided by the total number in that column because when the division happens the column entry get's stretched over the amount of rows

the inf is caused because there is no jq data point, and then after using smoothing and adding 1 to every data point, technically every letter is possible after every letter with extremely low likelihood. I am given 3.4 which is quite bad but possible

## Day 3 Nov 4
Explain how this probs variable works and then why exactly they all equal 1 in the end to resemble probabilities. Is what we are doing basically just making a 27x27 grid of random numbers and then tuning them through exponents and divisions to get probabilities where all the rows equal 1

Explaining how softmax turns raw scores into probabilities.

Your summary is close. One correction matters a lot: the exponent and division don’t tune anything. They’re a fixed conversion step. The tuning is done by gradient descent changing W, which is tomorrow. Here’s how it all fits.

Started rebuilding the bigram model as a neural network. Built the training set
  myself: `xs` holds each current letter and `ys` holds the correct next letter,
  as indices. For "emma": xs = [0, 5, 13, 13, 1], ys = [5, 13, 13, 1, 0].
- Collect data in Python lists with `.append()`, then convert once with
  `torch.tensor()`. Lowercase `torch.tensor` keeps whole numbers as int64;
  uppercase `torch.Tensor` always makes floats.
- A tensor is like a Java multidimensional array (`int[][]`), but it can do math
  on the whole grid at once, runs fast, and can compute its own gradients.
- One-hot encoding turns index 5 into 27 numbers: all 0 except a 1 at position 5.
  Needs `.float()` before multiplying with `W`.
- `xenc @ W` is matrix multiplication. Shapes: [5, 27] @ [27, 27] = [5, 27]. The
  inner numbers must match. My bug: `torch.randn(27)` made a single neuron with
  1 score per input instead of 27.
- Multiplying a one-hot vector by `W` just selects one row of `W`. So the input 'e'
  gives row 5 of `W`: 27 scores (logits) for what comes next.
- There is one `W` shared by every bigram, like one `N` for the whole dataset.
  `W` plays the role of `N`: `N` is filled by counting, `W` starts random and gets
  tuned by gradient descent. After training, `W` should be close to log(N).
- Softmax: `logits.exp()` makes every score positive ("fake counts"), then dividing
  each row by its sum makes each row add up to 1. This always gives valid
  probabilities, even with a random `W`, just bad ones. The loss measures how bad,
  and training fixes `W`.

  ## Day 4
  

