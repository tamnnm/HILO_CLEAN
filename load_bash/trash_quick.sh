#!/bin/bash
## Use only for extreme cases when you need to erase all the cases
for folder in ./*; do
  if [ -d "$folder" ]; then
    cd $folder
    find . -maxdepth 1 ! -name '.' ! -name '..' ! -name 'wrfout*' ! -name 'wrfrst*' ! -name 'namelist.input*' -exec mv {} "/work/users/tamnnm/trashbin/" \; 
    echo $folder
    cd ..  
fi
done
