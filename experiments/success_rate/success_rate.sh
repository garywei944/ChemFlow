#!/bin/bash

python experiments/success_rate/success_rate.py --prop plogp &
python experiments/success_rate/success_rate.py --prop qed &
python experiments/success_rate/success_rate.py --prop sa &
python experiments/success_rate/success_rate.py --prop drd2 &
python experiments/success_rate/success_rate.py --prop jnk3 &
python experiments/success_rate/success_rate.py --prop gsk3b &