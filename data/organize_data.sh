#!/bin/bash

mkdir -p others

for file in *.npz; do
    if [[ $file =~ ^X_(train|test)_TN_[0-9]+_time_[0-9]+_zone=(.+)\.npz$ 
]]; then
        type=${BASH_REMATCH[1]}      # train o test
        zone=${BASH_REMATCH[2]}      # nombre de zona

        mkdir -p "$zone/$type"
        mv "$file" "$zone/$type/"
    else
        echo "Archivo no coincide con el patrón esperado: $file"
        mv "$file" others/
    fi
done

