#!/usr/bin/env bash

for file in sprites/mario_orig/*.png; do
    filename=$(basename $file .png)
    stripped=$(echo $filename | tr -d '0-9')
    number=$(echo $filename | tr -dc '0-9')
    out=${stripped}left_${number}.png
    echo $file
    echo $stripped
    echo $number
    echo $out
    echo -----
    convert -interpolate Nearest -filter point -resize 300% -flop ${file} out/${out}
    
done
