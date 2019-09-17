'''
Copyright (c) 2019 Mikhail Milchenko, Washington University in Saint Louis
Comments: mmilch@wustl.edu

All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    - Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name of the Washington University in Saint Louis nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import synth_training_set as sts
import tensorflow as tf
import numpy as np
import cv2 
import os
import collections
import copy
import time
import sys
import argparse
import csv

__all__ = (
    'run_detection',
)

TSX=sts.TEXT_SHAPE[1]
TSY=sts.TEXT_SHAPE[0]
"""
Rectangle manipulation
"""        
class Rect:
    def overlaps(self,other):
        a, b = self, other
        xlt = max(min(a.xlt, a.xrb), min(b.xlt, b.xrb))
        ylt = max(min(a.ylt, a.yrb), min(b.ylt, b.yrb))
        xrb = min(max(a.xlt, a.xrb), max(b.xlt, b.xrb))
        yrb = min(max(a.ylt, a.yrb), max(b.ylt, b.yrb))
        return xlt<xrb and ylt<yrb
    
    def intersection(self, other):
        a, b = self, other
        xlt = max(min(a.xlt, a.xrb), min(b.xlt, b.xrb))
        ylt = max(min(a.ylt, a.yrb), min(b.ylt, b.yrb))
        xrb = min(max(a.xlt, a.xrb), max(b.xlt, b.xrb))
        yrb = min(max(a.ylt, a.yrb), max(b.ylt, b.yrb))
        if xlt<=xrb and ylt<=yrb:
            return type(self)(xlt, ylt, xrb, yrb)   
    def in_line(self,other):
        a,b=self,other
        if a.ylt==b.ylt and a.yrb==b.yrb:
            r=a.intersection(b)
            if not (r is None):
                return True
        else:
            return False
    def __str__(self):
        return"Rectangle, l,t,r,b=({},{},{},{})".format(self.xlt,self.ylt,self.xrb,self.yrb)
    
    @staticmethod
    def union_list(rects):
        if len(rects)<1: return None
        out=rects[0]
        for i in range(len(rects)):
            out=out.union(rects[i])
        return out        
            
    def union(self, other):
        a,b=self,other
        return type(self) (min(a.xlt,b.xlt),min(a.ylt,b.ylt),max(a.xrb,b.xrb),max(a.yrb,b.yrb))
        
    def __init__(self, xlt, ylt, xrb, yrb):
        if xlt>xrb or ylt>yrb:
            raise ValueError("Rectangle coordinates are invalid")
        self.xlt, self.ylt, self.xrb, self.yrb = xlt, ylt, xrb, yrb
        
    def area(self):
        return float(self.xrb-self.xlt)*(self.yrb-self.ylt)
    
    def significant_intersection(self,other,ratio=0.5):
        a,b=self,other
        c=a.intersection(b)
        if c is None: 
            return False
        s1,s2,s3=a.area(),b.area(),c.area()
        if s3!=0: 
            return (min(s1,s2)/s3 >= ratio)
        else:
            return False
#end class Rect            

"""
Resample a 2D image to the specified maximum dimension
"""
def resize_im(im,maxdim):    
    
    rx,ry=float(maxdim)/im.shape[1],float(maxdim)/im.shape[0]
    #print(rx,ry)
    r=min(rx,ry)
    return cv2.resize(im,dsize=(0,0),fx=r,fy=r),1./r
    #im1.shape
    #pyplot.imshow(im1)

def detect(im_gray_orig,maxdim,model,pthresh,verbose):
    #first, resample the image.    
    im_gray,scale=resize_im(im_gray_orig,maxdim)        
    #sliding window step.
    sx=8; sy=4;
    iw=im_gray.shape[1]; ih=im_gray.shape[0]
    found_rects=[]
    #print ("image dimensions: {}".format(str(iw)+","+str(ih)))
    t=0; b=t+TSY  
    t0=time.time()
    maxp=0
    detected_rects=[]
    tsx,tsy=TSX*scale,TSY*scale
    
    while b<=ih:
        l=0; r=l+TSX
        rects=[]
        coords=[]
        while r<=iw:
            rect=im_gray[t:b,l:r]

            rects+=[rect]
            coords+=[np.array([l,t])]
            if r==iw: 
                #print("last column: iw={}, l={}, r={}, t={}, b={}".format(iw, l,r,t,b))
                break            
            r+=sx; l+=sx
            if (l<iw and r>iw):
                l=iw-TSX; r=iw

        rects=np.array(rects)
        ps=model.predict(rects.reshape(rects.shape[0],TSY,TSX,1))

        #detect for each scan line.
        for i in range(rects.shape[0]):
            maxp=max(ps[i][0],maxp)
            if ps[i][0]>pthresh:                
                detected_rects+=[ Rect( int(coords[i][0]*scale),
                                       int(coords[i][1]*scale),
                                       int(coords[i][0]*scale+tsx),
                                       int(coords[i][1]*scale+tsy)) ]
        if b==ih: 
            break
        t+=sy; b+=sy
        if t<ih and b>ih:
            #print ("last row: ih={}, l={}, r={}, t={}, b={}".format(ih, l,r, t, b))
            t=ih-TSY; b=ih

    #detected_rects=np.array(detected_rects)    
    if verbose:
        if len(detected_rects)>0:
            print("dims=({},{}), p={:1.2f}, rects: {}, time: {:1.1f} s".format(iw,ih,maxp,len(detected_rects),time.time()-t0))
        else:
            print("dims=({},{}), p={:1.2f}, time: {:1.1f} s".format(iw,ih,maxp,time.time()-t0))
    return detected_rects,maxp
    
"""
Reduce detected rectangles to a non-intersecting set.
"""   
def prune_rects(detected_rects):
    n=0
    match={}
    for i in range(len(detected_rects)):
        for j in range(i):
            r1=detected_rects[i]; r2=detected_rects[j]
            if r1.overlaps(r2):
            #if r1.in_line(r2):
            #    match[i]=match[j]
            #    break
            #if r1.significant_intersection(r2):
                match[i]=match[j]
                break                
        else:
            match[i]=n
            n+=1
            
    groups = collections.defaultdict(list)
    for i, group in match.items():
        groups[group].append(i)
    out,n=[],0
    for group, rects in groups.items():
        #print(rects)
        grects=[ detected_rects[i] for i in rects ]
        r=Rect.union_list(grects)
        out+=[r]
    return out

"""
Get an image with detected rectangles.
"""    
def get_detect_im(im, detected_rects, prune=True):
    l1,l,n=0,len(detected_rects),0

    #iteratively reduce the number of rects.
    if prune:
        while (l1!=l):
            l=len(detected_rects)
            #print("iteration {}, rects={}".format(n,l))    
            detected_rects=prune_rects(detected_rects)
            l1=len(detected_rects)
            n+=1
    #output
    im_out=copy.copy(im)
    for r in detected_rects:
        cv2.rectangle(im_out, (r.xlt,r.ylt),(r.xrb,r.yrb), (0,255,0))
    return im_out

"""
Run detection on a directory of png images.
Write detected images and a .csv summary to dir_out.
"""
def run_detection(model,pthresh,dir_in,dir_out,verbose=False,monitor=None):
    dir=dir_in
    if not os.path.exists(dir): 
        print('path '+dir+' does not exist')
        return []        
    files = list()
    for (dirpath, dirnames, filenames) in os.walk(dir):
        files += [os.path.join(dirpath, file) for file in filenames]            
    #files=os.listdir(dir)
    if len(files)<1: 
        print('directory '+dir+' is empty')
        return []
    
    csvfile=dir_out+"/text_detect.csv"
    os.system('mkdir -p '+dir_out)    
    ntot,ndetect=0,0
    r1=384
    r2,r3=int((2.**0.25)*r1),int((2.**-0.25)*r1)
    mdl=model
    res_csv=[]       
    
    for file in files:
        #file=dir+'/'+f
        if not os.path.isfile(file):
            print ("{} is not a file.".format(file))
            continue      
        if verbose:
            print("Detecting text in: "+file)
        try:        
            im=cv2.imread(file)
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.
        except cv2.error as e:
            continue        
        #pyplot.imshow(im)
        found_rects,p=detect(im_gray,r1,mdl,pthresh,verbose)
        found_rects1,p1=detect(im_gray,r2,mdl,pthresh,verbose)
        found_rects+=found_rects1; p=max(p,p1)
        found_rects1,p1=detect(im_gray,r3,mdl,pthresh,verbose)
        found_rects+=found_rects1; p=max(p,p1)
        ntot+=1
        if len(found_rects)>0:
            im_out=get_detect_im(im,found_rects)
            outfil=file.replace('.png','')+'.dtext.png'
            if verbose:
                print('writing '+outfil)
            cv2.imwrite(outfil,im_out)
            res_csv+=[{'infile':file,'outfile':outfil,'text_present':1,"maxp":"{:1.2f}".format(p)}]
            ndetect+=1
        else:
            #cv2.imwrite(dir+"out/"+f,im)
            res_csv+=[{'infile':file,'outfile':file, 'text_present':0,"maxp":"{:1.2f}".format(p)}]
        if monitor is not None:
            if monitor['status']=='aborting': break
            
    if len(res_csv)>0:
        print ("Writing "+csvfile)
        with open(csvfile,'w') as f:
            dw=csv.DictWriter(f,res_csv[0].keys())
            dw.writeheader()
            dw.writerows(res_csv)
    print('Detected text in {} out of {} files.'.format(ndetect,ntot))
    return res_csv
  
if __name__ == "__main__":
    p=argparse.ArgumentParser(description='Detect text in a directory medical images in PNG format')
    p.add_argument('path_in',type=str,
        help='directory with input .png images')
    p.add_argument('path_out', type=str,
        help='output directory')
    p.add_argument('-p',metavar='<float>',type=float,help='threshold probability [0.99]')
    p.add_argument('-m',metavar='<model_path>',type=str,help='.h5 model file to use')
    p.add_argument('-v',action='store_true',help='verbose output')
    a=p.parse_args()
    model_file=a.m
    if model_file is None: 
        model_file=os.path.dirname(sys.argv[0])+'./models/09.10.2019.on_5M.hd5'
    try:
        model=tf.keras.models.load_model(model_file)
    except:
        print('Cannot load model from {}, exiting!'.format(model_file))
        sys.exit(-1)
    pmin=a.p
    if pmin is None:
        pmin=0.99 
    verbose=a.v
    print ('run_detection({},{},{},{},{})'.
        format('model',pmin,a.path_in,a.path_out,verbose))
    run_detection(model,pmin,a.path_in,a.path_out,verbose) 
        