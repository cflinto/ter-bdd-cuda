#!/bin/bash

filename='result_cuda.csv'

echo 'CUDA' > $filename
for i in {1..60}
do
    bin/test_cuda >> $filename
    sleep .1
    echo '' >> $filename
done


