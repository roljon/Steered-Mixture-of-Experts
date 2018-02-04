#!/usr/bin/env bash

mu_bits=(2 3 4 5 6 7 8)
m_bits=(2 3 4 5 6 7 8)
a_bits=(2 3 4 5 6 7 8)

results_base="blocksmoe_lena_np"

for mu_bit in ${mu_bits[*]}
do
    for m_bit in ${m_bits[*]}
    do
        for a_bit in ${a_bits[*]}
        do
            results_path="${results_base}/quant_mu_m_a_${mu_bit}_${m_bit}_${a_bit}"
            python block_smoe.py --results_path ${results_path} --mu_bits ${mu_bit} --m_bits ${m_bit} --a_bits ${a_bit} --iterations 20000
        done
    done
done
