from __future__ import division
import numpy as np

def Gen_Surf_Mesh(mx,my,XL,XR,XB,XT,YL,YR,YB,YT):
    # 2D transfinite interpolation mesh genenerator
    # given the boundary curves:
    # coordinates of the left edge: XL, YL.
    # coordinates of the right edge: XR, YR.
    # coordinates of the bottom edge: XB, YB.
    # coordinates of the top edge: XT, YT.
       
       hq = 1.0/(mx-1)
       hr = 1.0/(my-1)

       X = np.zeros((mx, my))
       Y = np.zeros((mx, my))

       for j in xrange(my):
          for i in xrange(mx):
            q = (i-1)*hq
            r = (j-1)*hr

            X[i,j] = (1.0-q)*XL[j]+q*XR[j]+(1.0-r)*XB[i]+r*XT[i]-\
                   (1.0-q)*(1-r)*XL[0]-q*(1.0-r)*XR[0]-r*(1.0-q)*XT[0]-\
                   (r*q)*XT[mx-1]

            Y[i,j] = (1.0-q)*YL[j]+q*YR[j]+(1.0-r)*YB[i]+r*YT[i]-\
                   (1.0-q)*(1.0-r)*YL[0]-q*(1.0-r)*YR[0]-r*(1.0-q)*YT[0]-\
                   (r*q)*YT[mx-1]              

       return [np.transpose(X), np.transpose(Y)]
       
