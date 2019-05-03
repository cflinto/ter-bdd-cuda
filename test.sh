#!/bin/bash

#echo '1thread;2thread;4threads;8threads' > result.csv
#for i in {1..30}
#do
#    ./bdd-cuda -n 20000000 -t -m 1-100000 -l compare_exchange -x 100000 -s dynamic -p 1 >> result.csv
#    sleep .5
#    echo -n ';' >> result.csv
#    ./bdd-cuda -n 20000000 -t -m 1-100000 -l compare_exchange -x 100000 -s dynamic -p 2 >> result.csv
#    sleep .5
#    echo -n ';' >> result.csv
#    ./bdd-cuda -n 20000000 -t -m 1-100000 -l compare_exchange -x 100000 -s dynamic -p 4 >> result.csv
#    sleep .5
#    echo -n ';' >> result.csv
#    ./bdd-cuda -n 20000000 -t -m 1-100000 -l compare_exchange -x 100000 -s dynamic -p 8 >> result.csv
#    sleep .5
#    echo '' >> result.csv
#done

filename='result_constraint_lock.csv'

echo 'low_none;low_omp_critical;low_compare_exchange;medium_none;medium_omp_critical;medium_compare_exchange;high_none;high_omp_critical;high_compare_exchange' > $filename
for i in {1..60}
do
    ./bdd-cuda -n 10000000 -t -m constaint_low -l none >> $filename
    sleep .1
    echo -n ';' >> $filename
    ./bdd-cuda -n 10000000 -t -m constaint_low -l omp_critical >> $filename
    sleep .1
    echo -n ';' >> $filename
    ./bdd-cuda -n 10000000 -t -m constaint_low -l compare_exchange >> $filename
    sleep .1
    echo -n ';' >> $filename
    ./bdd-cuda -n 10000000 -t -m constaint_medium -l none >> $filename
    sleep .1
    echo -n ';' >> $filename
    ./bdd-cuda -n 10000000 -t -m constaint_medium -l omp_critical >> $filename
    sleep .1
    echo -n ';' >> $filename
    ./bdd-cuda -n 10000000 -t -m constaint_medium -l compare_exchange >> $filename
    sleep .1
    echo -n ';' >> $filename
    ./bdd-cuda -n 10000000 -t -m constaint_high -l none >> $filename
    sleep .1
    echo -n ';' >> $filename
    ./bdd-cuda -n 10000000 -t -m constaint_high -l omp_critical >> $filename
    sleep .1
    echo -n ';' >> $filename
    ./bdd-cuda -n 10000000 -t -m constaint_high -l compare_exchange >> $filename
    sleep .1
    echo '' >> $filename
done

#filename='result_repartition.csv'
#
#echo '1/100_static;1-100000_static;1/100_dynamic;1-100000_dynamic;1/100_guided;1-100000_guided' > $filename
#for i in {1..60}
#do
#    ./bdd-cuda -n 10000100 -s static -x 100000 -m 1/100 -t -l compare_exchange >> $filename
#    sleep .1
#    echo -n ';' >> $filename
#    ./bdd-cuda -n 10000100 -s static -x 100000 -t -m 1-100000 -l compare_exchange >> $filename
#    sleep .1
#    echo -n ';' >> $filename
#    ./bdd-cuda -n 10000100 -s dynamic -x 100000 -m 1/100 -t -l compare_exchange >> $filename
#    sleep .1
#    echo -n ';' >> $filename
#    ./bdd-cuda -n 10000100 -s dynamic -x 100000 -t -m 1-100000 -l compare_exchange >> $filename
#    sleep .1
#    echo -n ';' >> $filename
#    ./bdd-cuda -n 10000100 -s guided -x 100000 -m 1/100 -t -l compare_exchange >> $filename
#    sleep .1
#    echo -n ';' >> $filename
#    ./bdd-cuda -n 10000100 -s guided -x 100000 -t -m 1-100000 -l compare_exchange >> $filename
#    sleep .1
#    echo '' >> $filename
#done

#filename='result_chunk_size_influence.csv'
#
#echo '100000;50000;20000;10000' > $filename
#for i in {1..60}
#do
#    ./bdd-cuda -n 10000100 -s static -x 100000 -m 1-100000 -t -l compare_exchange >> $filename
#    sleep .1
#    echo -n ';' >> $filename
#    ./bdd-cuda -n 10000100 -s static -x 50000 -m 1-100000 -t -l compare_exchange >> $filename
#    sleep .1
#    echo -n ';' >> $filename
#    ./bdd-cuda -n 10000100 -s static -x 20000 -m 1-100000 -t -l compare_exchange >> $filename
#    sleep .1
#    echo -n ';' >> $filename
#    ./bdd-cuda -n 10000100 -s static -x 10000 -m 1-100000 -t -l compare_exchange >> $filename
#    sleep .1
#    echo '' >> $filename
#done

#filename='result_thread_num_influence.csv'
#
#echo '1;2;4;8' > $filename
#for i in {1..60}
#do
#    ./bdd-cuda -n 10000100 -t 1 -m 1/100 -t -l compare_exchange >> $filename
#    sleep .1
#    echo -n ';' >> $filename
#    ./bdd-cuda -n 10000100 -t 2 -m 1/100 -t -l compare_exchange >> $filename
#    sleep .1
#    echo -n ';' >> $filename
#    ./bdd-cuda -n 10000100 -t 4 -m 1/100 -t -l compare_exchange >> $filename
#    sleep .1
#    echo -n ';' >> $filename
#    ./bdd-cuda -n 10000100 -t 8 -m 1/100 -t -l compare_exchange >> $filename
#    sleep .1
#    echo '' >> $filename
#done

#filename='result_column_vs_row_first.csv'

#echo 'column_first;row_first' > $filename
#for i in {1..60}
#do
#    ./bdd-cuda -n 30000100 -m 1/100 -t -l compare_exchange >> $filename
#    sleep .1
#    echo -n ';' >> $filename
#    ./bdd-cuda -n 30000100 -r -m 1/100 -t -l compare_exchange >> $filename
#    sleep .1
#    echo '' >> $filename
#done
