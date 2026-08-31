**Direction: exhaustive hyperparameter sweep on the best known configuration.**

Every previous run changed the model's STRUCTURE and left its hyperparameters at the
baseline's defaults (k=16, lr=0.001, epochs=40, batch 8192, patience=4). Nobody has
actually tuned them on top of the features that work. That is this run's only job — do
not invent new features.

- Start from the best known configuration: FM + causal history field(s) + 3-seed averaging.
- Sweep, and report a table of valid primary for every setting tried:
  embedding dim k **∈ {8, 16, 32, 64}**, learning rate **∈ {3e-4, 1e-3, 3e-3}**,
  L2 **∈ {1e-6, 1e-5, 1e-4}**, batch size **∈ {4096, 8192, 16384}**, early-stopping
  patience **∈ {4, 8}**.
- Do not sweep all combinations exhaustively — that will not fit the time limit. Vary one
  axis at a time from the best-so-far setting (coordinate descent), and say which axis you
  were varying at each point.
- Note: organizers already tested k ∈ {8,16,32} on the PLAIN baseline and found no effect.
  The open question is whether that still holds once history features are present.
