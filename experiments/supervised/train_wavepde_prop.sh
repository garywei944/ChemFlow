#!/bin/bash

for pde in wave hj; do
  for prop in 1err 2iik; do
    python experiments/supervised/train_wavepde_prop.py \
      --prop $prop \
      --model.learning_rate 1e-3 \
      --model.pde_function $pde \
      --data.n 10_000
  done
done
