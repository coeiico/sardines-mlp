"""
The model for sardines-mlp.

x → ReLU(WᵀWx + b), with n features (5–20) squeezed into m dimensions (2–5), varying
feature sparsity and importance.

forward() accepts return_cache=True and returns a dict of internal activations, and a
patch argument (or forward hooks) overwrites a named activation for interventions.
"""

# TODO: implement. See README.md for the plan.
