#!/bin/bash

for pde in wave hj; do
  for prop in plogp qed sa jnk3 drd2 gsk3b uplogp; do
    #  for prop in 1err 2iik; do
    python experiments/supervised/train_wavepde_prop.py \
      --prop $prop \
      --model.learning_rate 1e-3 \
      --model.pde_function $pde \
      --data.n 100_000
  done
done
