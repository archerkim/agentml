**Direction: give the model the ability to WEIGHT history by relevance to the candidate item.**

This run exists because of a mechanism two previous runs derived from their own
measurements, not because a deep model is fashionable:

> "In a standard FM the interaction sum is **position-symmetric**, so adding more history
> slots cannot concentrate signal on the most relevant ones."

> "In this symmetric-FM setup, sweeping K between 5 and 10 for long_view history is a
> **null experiment**; the binding constraint on history signal is the architecture."

That explains every history result recorded here: last-3-videos + prev_author merged
together scored 0.6026 — no better than one field alone; K-sweeps moved nothing; and
FM+history plateaus at valid ~0.6045 / test ~0.5981 regardless of how many recency
fields are added. Adding a 4th and 5th history field cannot help, because FM's
second-order term sums all field pairs symmetrically and has no way to say "this
historical item matters for THIS candidate video and that one does not".

**Your job is to add that missing capability.** Concretely:

a. Keep an FM component (it is a strong baseline) and add a deep tower that SHARES one
   embedding table with it, summing both outputs — DeepFM. `findings.jsonl` records
   DeepFM at test 0.5982, above everything the automated search has reached, and no run
   has ever attempted it.
b. The deep tower is what can represent target-dependent weighting: feed it the
   concatenated candidate embedding and the history embeddings so it can learn
   interactions between them that the symmetric FM term cannot express.
c. If that works and budget remains, make the weighting explicit — score each historical
   item by similarity to the target embedding and pool with those weights (DIN-style
   attention). Note the recorded caveat: naive shared-embedding attention previously came
   out at parity (test 0.5949), so prefer the deep tower first and attention second.
d. Sweep COORDINATE-WISE, not as a full grid, and REPORT valid primary for every setting
   tried: start from hidden=[128,64], dropout=0.2, k=16, then vary ONE axis at a time -
   hidden ∈ {[64,32], [128,64], [256,128]}, then dropout ∈ {0.0, 0.2}, then k ∈ {16, 32}.
   A full grid is 12 configs and, with 3-seed averaging, 36 trainings - a previous attempt
   tried exactly that and was killed by the time limit. Run the sweep single-seed to pick
   the best config, then apply 3-seed averaging only to that winner.
   Do not sweep negative sampling counts — measured null this session (1 vs 16 negatives:
   0.0003 apart at 16x the cost).
e. Keep the mandatory 3-seed averaging.

⚠️ **USE PYTORCH FOR THE DEEP TOWER. Do NOT hand-write backpropagation in numpy.**
The solution you are given is pure numpy because it descends from the numpy FM baseline,
and the first DeepFM attempt inherited that and tried to hand-derive the MLP backward
pass. It crashed with two independent gradient bugs: `dh = (dh @ self.DW[i].T) * (acts[i+1] > 0)`
applied the ReLU mask after the weight transpose instead of before
(`ValueError: operands could not be broadcast together with shapes (8192,128) (8192,64)`),
plus a shape mismatch accumulating embedding gradients. Hand-deriving backprop for a
multi-layer tower is an unnecessary source of exactly this failure. `torch` 2.13 is
installed. Define the model as an `nn.Module` and let autograd compute the gradients -
you may keep the numpy FM as-is and add a torch tower alongside it, or port the whole
scorer to torch, whichever is cleaner.

⚠️ **PUT THE TORCH MODEL ON THE GPU.** Two NVIDIA A5000s are present with roughly 10 GB
free on each (`torch.cuda.is_available()` is True, `device_count()` is 2). Use
`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` and move the model
and each batch to it. A previous attempt trained this model on CPU over 1.1M rows and was
killed by the time limit; on GPU the same work is orders of magnitude faster. Guard with
the `is_available()` check so the script still runs if no GPU is present.

⚠️ **KEEP THE EXISTING DATA PIPELINE. Change only the model and its training loop.**
The solution you are given already loads the CSVs, builds the causal split-aware history
features, encodes fields with a train-only vocabulary, averages 3 seeds, writes an
int-cast `submission.csv` and saves `valid_scores.npy`/`test_scores.npy`. Four previous
full-rewrite attempts in this project all crashed: three (TimeoutError, KeyError,
ValueError) in re-derived data-loading/feature code, and the fourth in a hand-written
gradient. Reuse the data scaffolding verbatim and spend your effort on the architecture. Also note the recorded vocab trap: any history value referencing an entity
not seen in train silently becomes UNK and destroys the interaction signal.
