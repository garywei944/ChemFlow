#!/bin/bash

prop=qed

## pf
#python experiments/optimization/optimization.py --prop $prop --method fp --step_size 0.1 --relative
#id1=$!
#
## limo
#python experiments/optimization/optimization.py --prop $prop --method limo --step_size 0.1 --relative
#id2=$!
#
## random
#python experiments/optimization/optimization.py --prop $prop --method random --step_size 0.1
#id3=$!
## pde
#python experiments/optimization/optimization.py --prop $prop --method wave_sup --step_size 0.1 --relative
#id5=$!
#
#wait $id1 $id2 $id3 $id5
#
## random 1d
#python experiments/optimization/optimization.py --prop $prop --method random_1d --step_size 0.1
#id4=$!
## chemspace
#python experiments/optimization/optimization.py --prop $prop --method chemspace --step_size 0.1
#
python experiments/optimization/optimization.py --prop $prop --method wave_unsup --step_size 0.1 --relative
#id6=$!
#python experiments/optimization/optimization.py --prop $prop --method hj_sup --step_size 0.1 --relative
#id7=$!
python experiments/optimization/optimization.py --prop $prop --method hj_unsup --step_size 0.1 --relative
#id8=$!
#
#wait $id4 $id6 $id7 $id8
