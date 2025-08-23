module load pyngl/1.4.0
#exp='0109'
exp='1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010'
#exp='0206 0213 0218 0305 0312 0319 0326 0402 0409 0416 0423 0514 0521 0528 0625'
mkdir PDF
mkdir TXT
mkdir KQ
mkdir PNG
for year in $exp
 do
  cp Auto_Post_proc.py temp.py 
  sed 's/year/'"$year"'/g' -i temp.py
  ./temp.py > Track_${year}
  mv TCs.pdf PDF/"$year".pdf
 ########## V ?
  mv Output.V4."$year" TXT/ 
  mv Track_$year KQ/
convert TCs.PS ${year}.png
  mv ${year}-1.png PNG/${year}.png
 rm ${year}-0.png TCs.PS
done
