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
  The pulls cancel out exactly when the probabilities match the real frequencies. If the network says m comes after e more often than it really does, the downward pulls win and the score drops. If it says less often, the upward pulls win and the score rises. The only stable point, where W.grad becomes about zero and nothing moves anymore, is where the network’s probability for each letter equals how often that letter actually followed e.

That’s why the trained network ends up matching your counting table: counting gives the real frequencies directly, and gradient descent gets there by tug-of-war. And it’s why the loss stops at about 2.45 instead of going to 0. The data itself is uncertain, since many different letters can follow e, and no model with one letter of context can predict it perfectly.

and this loss.backward pushed them to their respective spots, but still we have to think about probabilities right like just because it happened doesn't mean it always happens

guess, measure, adjust, repeat. W is a grid of 729 dials. You start with the dials set randomly, measure how bad the predictions are (the loss), figure out which way to turn each dial to make the loss smaller (backward), turn every dial a little in that direction (the update), and repeat. After enough rounds, the dials end up in a good position. That’s all training is, here and in GPT.

Ok finished the video. So from what I understand, the neural network part of this works by having a dataset of names, then putting each of those names in a bigram tensor kind of list by their integers, having . as the first and last letters. Then we create a new W matrix of size 27/27 with random values, create a new list xenc which makes a list of  lists size 27 of the input xs with a 1 in the position where the original xs stoi would be, then we multiply that list by the matrix W which in turn gives us a matrix of [5,27], then we manipulate that data by exponentioating it and taking the probabilities of every row to add up to 1 in order to "fake" that data looking like probabilities even thought they ar ebad ones, and then calculate a loss from that using negative log likelihood, then using.backward to go back through all the steps in order to tune the gradients either up or down depending on how they shift the loss up or down. I know some of this doesn't make sense probably, so with the parts that I didn't put clearly I didn't put clearly because I wasn't sure or didn't know so help explain, mostly with the matrix parts

You’ve got almost all of it. Here’s your summary with the fuzzy parts fixed, followed by the matrix flow in one picture.

Your summary, corrected step by step
Dataset to pairs. ✅ Right. One fix: it’s two separate 1D tensors, xs (the current letter of each pair) and ys (the correct next letter), lined up so xs[i] and ys[i] form one bigram.
Random W, 27×27. ✅ Right. It’s one shared grid for every bigram, just a starting guess.
One-hot xenc. ✅ Exactly right: one row of 27 per example, with a 1 at that letter’s position. Shape [number of examples, 27].
xenc @ W. ✅ Right, with one fix: [5, 27] was only because “emma” has 5 bigrams. On the full dataset it’s [228146, 27]. The key idea: each one-hot row picks out one row of W. So row i of the result is just “W’s row for the letter xs[i]”: 27 scores for what comes next.
exp and normalize. Mostly right. Fix the word “fake”: the probabilities are completely real and valid; each row truly sums to 1. What’s “fake” is the step in between: exp makes the scores act like counts (“fake counts”), similar to N. And the probabilities are only bad at the start; training makes them good.
The loss. ✅ Right, and here’s the missing detail: for each row, you pick out only the probability given to the correct answer (probs[torch.arange(num), ys]), take the log, average, and flip the sign. The 26 wrong letters aren’t looked at directly.
backward(). One important fix: backward() doesn’t tune anything. It calculates a gradient for every number in W (stored in W.grad): which way, and how much, that number should move to lower the loss. The update (W.data += -50 * W.grad) is what tunes W. The gradients are the directions, and W is what changes.
The step you left out: repeat. Reset the gradients, run the forward pass again, compute the loss, backward(), update, and do that 100 or more times. Each loop, the loss drops a bit.

## Day 5

Each pass does four things:

ix = stoi[ch] turns the current character into its number.
X.append(context) saves the 3 letters before this character, as the input.
Y.append(ix) saves this character as the answer.
context = context[1:] + [ix] slides the window: drop the oldest letter, add this one. The current character becomes part of the context for the next pass.

