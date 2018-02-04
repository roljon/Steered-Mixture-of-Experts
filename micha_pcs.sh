#!/usr/bin/env bash

N=100000
base_lr=0.0001
v=1000
lr_div=100
k=7

train_pis=(0 1)
train_gammas=(0 1)
radial=(0 1)
for tp in ${train_pis[*]}
do
    for tg in ${train_gammas[*]}
    do
        for rad in ${radial[*]}
        do
            if ((${tp} == 0))
            then
                pis_arg="--disable_train_pis True"
            else
                pis_arg=""
            fi

            if ((${tg} == 0))
            then
                gammas_arg="--disable_train_gammas True"
            else
                gammas_arg=""
            fi

            if ((${rad} == 0))
            then
                rad_arg="--lr_mult 100 --radial_as True "
            else
                rad_arg=""
            fi

            time python3 smoe_test.py -i Tok1.png -r micha_pcs_full_7x7/${tp}_${tg}_${rad} -k ${k} -n ${N} -lr ${base_lr} -v ${v} --lr_div ${lr_div} ${pis_arg} ${gammas_arg} ${rad_arg}

        done
    done
done



#time python3 smoe_test.py -i Tok1.png -r micha_pcs/tok1_${k}_lr${base_lr}-_div${lr_div} -k ${k} -n ${N} -lr ${base_lr} -v ${v} --lr_div ${lr_div}
#time python3 smoe_test.py -i Tok1.png -r micha_pcs/tok1_${k}_lr${base_lr}-_div${lr_div}_dp_rad -k ${k} -n ${N} -lr ${base_lr} -v ${v} --lr_div ${lr_div} --lr_mult 100 --radial_as True --disable_train_pis True
#time python3 smoe_test.py -i Tok1.png -r micha_pcs/tok1_${k}_lr${base_lr}-_div${lr_div}_dp_dg_rad -k ${k} -n ${N} -lr ${base_lr} -v ${v} --lr_div ${lr_div} --lr_mult 100 --radial_as True --disable_train_pis True --disable_train_gammas True
#time python3 smoe_test.py -i Tok1.png -r micha_pcs/tok1_${k}_lr${base_lr}-_div${lr_div}_dp -k ${k} -n ${N} -lr ${base_lr} -v ${v} --lr_div ${lr_div} --disable_train_pis True
#time python3 smoe_test.py -i Tok1.png -r micha_pcs/tok1_${k}_lr${base_lr}-_div${lr_div}_dp_dg -k ${k} -n ${N} -lr ${base_lr} -v ${v} --lr_div ${lr_div} --disable_train_pis True --disable_train_gammas True
