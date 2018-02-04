#!/usr/bin/env bash

N=10000
base_lr=0.001
v=100
lr_div=100

ks=(20 25 30 35 40 45 50 55 60 65 70)

for k in ${ks[*]}
do
    time python3 smoe_test.py -i images/peppers.png -r final_div100/batch_ref_peppers_128_lr0-001-_div100_r0/${k} -k ${k} -n ${N} -lr ${base_lr} -v ${v} --lr_div ${lr_div} -b 32
done