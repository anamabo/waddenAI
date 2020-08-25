# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:21:59 2020

@author: marti_cn
"""

import numpy as np
import pandas as  pd
import rasterio
import os

#####Get pixels and geof location ################

path = os.path.abspath( os.path.join('..', '..', '2_data','grid'))
raster = rasterio.open(os.path.join(path,"wadden.tif"))
band= raster.read(1)
band = np.where(band!=0, band, np.nan)

X= []
Y= []
Xm = []
Ym= []
value = []
counts = []
k = 0
for x in range(0, raster.width):
    for y in range(0, raster.height):
        xm, ym = raster.transform*(x,y)
        X.append(x)
        Y.append(y)
        Xm.append(xm)
        Ym.append(ym)
        value.append(band[y,x])
        counts.append(k)
        k+=1

df = pd.DataFrame({'pixelID': counts,
              'x_pixel': X,
              'y_pixel': Y,
              'x_meters': Xm,
              'y_meters': Ym,
              'value': value})

df.to_csv(os.path.join(path,'wadden_pixel2meter.csv'), index= False)
# Get a table with filtered pixels for future joins:
df1 = df[df.value== 1]
df1[['pixelID']].to_csv(os.path.join(path,'filtered_pixels.csv'), index= False)
