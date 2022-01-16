#!/bin/sh
clear
echo "Running benchmark "

ST_ExecutablePath="builds/tsp_stdThreads" # std threads implementation
FF_ExecutablePath="builds/tsp_ff" # fastflow implementation
filenameRoot="plotting-and-data/data/benchmark"
maxThreads=8
nCities=500
iterations=2
crossoverRate=0.1
mutationRate=0.01

for popSize in 10000 20000 100000; do # number of chormosomes
  echo pop size: ${popSize}
  for exec in ${ST_ExecutablePath} ${FF_ExecutablePath}; do # for each version
    if [ "${exec}" = "${ST_ExecutablePath}" ]; then echo \"stdThreads\">> ${filenameRoot}-${popSize}.txt; fi #print the heading
    if [ "${exec}" = "${FF_ExecutablePath}" ]; then echo \"FastFlow\">> ${filenameRoot}-${popSize}.txt; fi #print the heading
    : $((w = 1))
    while [ $((w <= maxThreads)) -ne 0 ]; do #for each thread configuration
      echo ${exec} ${nCities} ${popSize} ${iterations} ${mutationRate} ${crossoverRate} ${w}
      ./${exec} ${nCities} ${popSize} ${iterations} ${mutationRate} ${crossoverRate} ${w}>>${filenameRoot}-${popSize}.txt
      : $((w = w + 1))
      sleep 0.5s
    done
    echo >>${filenameRoot}-${popSize}.txt # separate the two datasets in the file
    echo >>${filenameRoot}-${popSize}.txt
  done
  echo
done

echo "Benchmark ended"
