#!/bin/bash
for folder in ./*; do
 sed -n '50,60p' $folder/namelist.input
done

