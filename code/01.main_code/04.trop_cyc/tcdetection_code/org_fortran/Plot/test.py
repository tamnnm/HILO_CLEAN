#!/usr/bin/python
#
# This python script read the output from TCdetect, filter it and do some statistic analyics and plot the tracks
#

import Numeric, Ngl, math


# Read TC file
### automatic running 
input_file="test.txt"
#input_file="TCs.txt"
output_file="output"
min_TC_length = 8  # Minimum number TC obs for processing
output_type="x11"    # PS or x11 or PDF

file=open(input_file, 'r')
ofile=open(output_file,'w')
temp = file.readline().split(';')
num_Tcs = int (temp[1])
print num_Tcs
TCs_per_mon=[0,0,0,0,0,0,0,0,0,0,0,0]
TCs = []
for itc in range(num_Tcs):
   #print itc
   temp = file.readline()
   temp = file.readline()
   temp = file.readline().split(':')
   num_obs = int (temp[1])
   TC_obs = []
   for iobs in range(num_obs):
      temp=file.readline().split(';')
      obs = {'yyyy':int(temp[0]),
             'mm':int(temp[1]),
	     'dd':int(temp[2]),
	     'hh':int(temp[3]),
	     'lon': float(temp[4]),
	     'lat': float(temp[5]),
	     'Tano': float(temp[6]),
	     'Pano':float(temp[7]),
	     'OCS':float(temp[8])} 
      TC_obs.append(obs)
   
# Filter only TCs that lives at least 24 hours.
   if (len(TC_obs)>=min_TC_length):
      TCs.append(TC_obs)
      for m in range(len(TCs_per_mon)):
         if ((m+1 == TC_obs[0]['mm']) or (m+1 == TC_obs[len(TC_obs)-1]['mm'])) :
            TCs_per_mon[m]=TCs_per_mon[m]+1

print "Number of TCS",len(TCs)
print "TCs per mon",TCs_per_mon



#Plot tracks
wkres = Ngl.Resources()
wks_type = output_type
wks = Ngl.open_wks(wks_type,"TCs",wkres)

res=Ngl.Resources()
#
res.mpDataBaseVersion="Ncarg4_1"
res.mpDataSetName = "Earth..3"

# Limit the map
res.mpProjection='CylindricalEquidistant'
res.mpCenterLonF=105.
res.mpCenterLatF=0.
res.mpLimitMode = 'LatLon'
res.mpMinLatF=0.
res.mpMaxLatF=45.
res.mpMinLonF=90.
res.mpMaxLonF=180.
res.mpGridSpacingF=5.
res.mpGridLineColor="green"
res.mpGridMaskMode = "MaskLand"
res.mpLabelsOn=True
res.mpGridAndLimbDrawOrder="Draw" 
# 
res.mpFillOn=True
res.mpFillBoundarySets = "National"

res.mpOutlineBoundarySets = "National"
res.mpNationalLineColor = "yellow"
res.mpOutlineDrawOrder = "Draw"

# Fillthe area manually
res.mpFillColor=True
#res.mpOceanFillColor ="blue"
#res.mpOceanFillColor = Ngl.new_color(wks,.4,0.3,.8)
res.mpOceanFillColor = Ngl.new_color(wks,1.,1.,1.)
res.mpFillAreaSpecifiers = ['Land','Vietnam']
#land_color=Ngl.new_color(wks,0.3, 0.3, 0.3)
land_color = Ngl.new_color(wks,0.7, 0.7, 0.7)
vn_color   = Ngl.new_color(wks,0.5, 0.5, 0.5)
cmap = [land_color,vn_color]

#res.mpSpecifiedFillColors = ['DarkOrange','DeepPink']
res.mpSpecifiedFillColors = cmap
res.mpGeophysicalLineColor="gray" # Chi la duong bo bien?
res.mpGeophysicalLineThicknessF=2.0     # double

tcmap = Ngl.map(wks,res)

lineres=Ngl.Resources()
lineres.gsLineThicknessF=1.5
#lineres.gsLineColor="black"

tres=Ngl.Resources()
#tres.txFont='weather2' 
tres.txFontHeightF=0.01 
tres.txFontColor="Brown"

# Now use PyNGL to plot!
itc=1
for TC in TCs:
   
   lons=[]
   lats=[]
   
   print "TC---",itc
   for i in range(len(TC)):	   
      lons.append(TC[i]['lon'])
      lats.append(TC[i]['lat'])
      print TC[i]['dd'],TC[i]['mm'],TC[i]['yyyy'],TC[i]['hh'],TC[i]['lon'],TC[i]['lat'],TC[i]['Tano'],TC[i]['Pano'],TC[i]['OCS']

# tc
   pline = Ngl.add_polyline(wks, tcmap, lons, lats, lineres)
   Ngl.add_text(wks, tcmap, `TC[i]['dd']`+'/'+`TC[i]['mm']`, lons[0], lats[0],tres)	
   itc=itc+1
   
   #Ngl.draw(tcmap)

Ngl.add_text(wks, tcmap, "~F35~z", 105.51, 21.2,tres)
Ngl.draw(tcmap)
Ngl.frame(wks)
Ngl.end()

print "\nTCs per mon",TCs_per_mon

# Output to file

ofile.write("No of TCs: %(notc)i" % {'notc':len(TCs)})
ofile.write("\nTCs per mon")
for m in range(len(TCs_per_mon)):
   ofile.write("\n %02i    %i" % (m+1,TCs_per_mon[m]))

for TC in TCs:
   ofile.write("\n%(yyyy1)04i/%(mm1)02i/%(dd1)02i - %(yyyy2)04i/%(mm2)02i/%(dd2)02i " % 
                {'yyyy1':TC[0]['yyyy'],
                'mm1':TC[0]['mm'],
                'dd1':TC[0]['dd'],
                'yyyy2':TC[len(TC)-1]['yyyy'],
                'mm2':TC[len(TC)-1]['mm'],
                'dd2':TC[len(TC)-1]['dd']})
   max_OCS=0.
   min_Pano=0.
   for obs in TC:
     if (max_OCS < obs['OCS']):
        max_OCS=obs['OCS']
     if (min_Pano > obs['Pano']):
        min_Pano=obs['Pano']
   
   ofile.write("  %(OCS)3.1f"%{"OCS":max_OCS})
   ofile.write("  %(Pano)3.1f"%{"Pano":min_Pano})
   
