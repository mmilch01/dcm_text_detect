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

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from skimage.filters.rank import entropy

import cv2
import numpy as np
import itertools
import math
import os
import random
import sys

__all__ = (
    'gen_training_im',
    'TEXT_SHAPE'
)

FONT_HEIGHT=14
DIGITS = ['0','1','2','3','4','5','6','7','8','9']
#LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#CHARS = LETTERS + DIGITS

VOWS_L = ['a','e','i','o','u','y']
VOWS_U = ['A', 'E', 'I', 'O', 'U', 'Y']
VOWS_P = [ 0.2025 ,  0.3175 ,  0.17425,  0.1875 ,  0.069  ,  0.04925]
CONS_L = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z']
CONS_U = ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Z']
CONS_P = [ 0.02150877,  0.0401497 ,  0.06165847,  0.03298011,  0.02882175,
           0.08746899,  0.00219389,  0.01106985,  0.0576435 ,  0.03441403,
           0.09678946,  0.02767462,  0.01362222,  0.08589168,  0.09062361,
           0.12991296,  0.14052395,  0.03384046,  0.00215088,  0.0010611 ]
#DIGITS = ['0','1','2','3','4','5','6','7','8','9']
CHARS = VOWS_L+VOWS_U+CONS_U+CONS_L+DIGITS
LETTERS = VOWS_L+VOWS_U+CONS_U+CONS_L
LETTERS_L = VOWS_L + CONS_L
LETTERS_U = VOWS_U + CONS_U
CHARS_L = VOWS_L + CONS_L + DIGITS
CHARS_U = VOWS_U + CONS_U + DIGITS
TEXT_SHAPE = (16, 32)

def strcode(s1,s2,s3):
    return "{0}{1}{2}".format(random.choice(s1),random.choice(s2),random.choice(s3))
#generate character sequence
def generate_char_seq():
    letters_digits=random.randint(0,4)
    if letters_digits == 4:
        return strcode(DIGITS,DIGITS,DIGITS)
    else:
        code=random.randint(0,7)
        if code == 0:
            return strcode(CHARS_L,CHARS_L,CHARS_L)
        elif code == 1: 
            return strcode(CHARS_U,CHARS_U,CHARS_U)
        elif code <= 3: 
            return strcode(LETTERS_U,LETTERS_U,LETTERS_U)
        elif code <= 5: 
            return strcode(LETTERS_U,LETTERS_L,LETTERS_L)
        else: 
            return strcode(LETTERS_L,LETTERS_L,LETTERS_L)

def generate_random_text_image(char_ims, font_height, bg):
    top_pad = random.uniform(0.0, 0.2) * font_height
    bottom_pad = 0
    
    left_pad = random.uniform(0, 0.15) * font_height
    right_pad = random.uniform(0, 0.15) * font_height
    
    spacing=font_height * random.uniform(-0.05,0.05)
    chseq = generate_char_seq()                                    
                                
    text_width=sum(char_ims[c].shape[1] for c in chseq)
    #print(text_width)
    text_width+=(len(chseq)-1)*spacing+left_pad+right_pad
    #print(text_width)
    out_shape=( int(font_height+top_pad+bottom_pad), int(text_width) )
    left_justify=(np.random.random(1)>0.5)   
    top_justify=(np.random.random(1)>0.5)
    if np.random.random(1)>0.5: 
        left_pad1=max(0,TEXT_SHAPE[1]-out_shape[1])
    else:
        left_pad1=0
    if np.random.random(1)>0.5:
        top_pad1=max(0,TEXT_SHAPE[0]-out_shape[0])
    else:
        top_pad1=0
    #print("left_pad1:{}, top_pad1:{}".format(left_pad1,top_pad1))                       
    #print("font_height: {}, text_width: {}, top_pad: {}, bottom_pad: {}, spacing: {}, out_shape:{}".format(font_height,text_width,top_pad,bottom_pad,spacing,out_shape))
    
    #bright text on dark bg
    p5 = np.percentile(bg,50)
    if p5 > 0.5:
        text_int = random.random()*random.random()*0.2
        text_inv = 1
    #dark text on bright bg
    else:
        text_int = 1. - random.random()*random.random()*0.2
        text_inv = 0
    text_mask=np.zeros(out_shape)
    x=left_pad
    y=top_pad
    for c in chseq:
        ci=char_ims[c]
        ix,iy=int(x),int(y)
        #print ("ix:{}, iy:{}, ci.shape0:{}, ci.shape1:{}".format(ix,iy,ci.shape[0],ci.shape[1]))
        text_mask[iy:iy+ci.shape[0],ix:ix+ci.shape[1]]=ci
        x += ci.shape[1]+spacing
    out_mask=np.zeros((TEXT_SHAPE[0],TEXT_SHAPE[1]))
    wx=min(out_shape[1],TEXT_SHAPE[1])
    wy=min(out_shape[0],TEXT_SHAPE[0])
    out_mask[top_pad1:wy+top_pad1,left_pad1:wx+left_pad1]=text_mask[0:wy,0:wx]
    
    return out_mask,text_int,chseq

#generate random background.
def generate_bg(bgs,nbgs):
    found = False
    while not found:
        fname = "bgs/{0}".format(bgs[random.randint(0, nbgs - 1)])
        bg = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) / 255.
        if (bg.shape[1] >= TEXT_SHAPE[1] and
            bg.shape[0] >= TEXT_SHAPE[0]):
            found = True
	
    x = random.randint(0, bg.shape[1] - TEXT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - TEXT_SHAPE[0])
    bg = bg[y:y + TEXT_SHAPE[0], x:x + TEXT_SHAPE[1]]

	#randomly flip.
    flipx = random.randint(0,1);
    if flipx == 1:
        bg = cv2.flip(bg,0);
		
    flipy = random.randint(0,1);
    if flipy == 1:
        bg = cv2.flip(bg,1);		
    return bg
        
def make_random_text_im(bgs, nbgs, fonts, font_char_ims):
    bg = generate_bg(bgs,nbgs)
    font=fonts[np.random.randint(len(fonts))]
    text_mask, text_int, code = generate_random_text_image(font_char_ims[font], FONT_HEIGHT, bg)    
    if np.random.random(1) > 0.2:
        out_of_bounds = True
    else:
        out_of_bounds = False
        
    if out_of_bounds:
        if np.random.random(1)>0.5:
            out = bg
        else:
        
            #randomly shift the text by at least half of width and height.
            wx=TEXT_SHAPE[1];wy=TEXT_SHAPE[0]
            # randomly generate |pad|>0.4
            r=np.random.uniform(-0.6,0.6); padx=int(wx*np.where(r<0,1.+r,-1.+r))
            r=np.random.uniform(-0.6,0.6); pady=int(wy*np.where(r<0,1.+r,-1.+r))
            sy00=max(0,pady);  sy01=min(wy,wy+pady)
            sy10=max(0,-pady); sy11=min(wy-pady,wy)
            sx00=max(0,padx);  sx01=min(wx,wx+padx)
            sx10=max(0,-padx); sx11=min(wx-padx,wx)
            out=bg
            #print("sy0: {}-{}, sy1: {}-{}, sx0: {}-{},sx1:{}-{}".format(sy00,sy01,sy10,sy11,sx00,sx01,sx10,sx11))
            out[sy00:sy01,sx00:sx01]=bg[sy00:sy01,sx00:sx01]*(1-text_mask[sy10:sy11,sx10:sx11]) \
                + text_int*text_mask[sy10:sy11,sx10:sx11]
        
    else:        
        out = bg * (1-text_mask) + text_int * text_mask
    out = cv2.resize(out, (TEXT_SHAPE[1], TEXT_SHAPE[0]))
    out = np.clip(out, 0., 1.)
    return out, code, not out_of_bounds


def make_char_ims(font_path, output_height):
    font_size = output_height * 4    
    font = ImageFont.truetype(font_path, font_size)
    height = max(font.getsize(c)[1] for c in CHARS)

    for c in CHARS:
        width = font.getsize(c)[0]
        im = Image.new("RGBA", (width, height), (0, 0, 0))

        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, np.array(im)[:, :, 0].astype(np.float32) / 255.

def gen_training_im():
    bgs=[f for f in os.listdir("./bgs") if f.endswith('.png')]
    nbgs=len(bgs)
    fonts=[f for f in os.listdir("./fonts") if f.endswith('.ttf')]
    fonts_char_ims={}
    for font in fonts:
        fonts_char_ims[font] = dict(make_char_ims(os.path.join('./fonts', font), FONT_HEIGHT))
    while True:
        yield make_random_text_im(bgs,nbgs,fonts,fonts_char_ims)

def entropy (im):
    bins=50; nv=1e-10
    h=np.histogram(im, bins=bins)
    s=sum(h[0]); ls=np.log2(s)
    e=(1.0/s)*sum([h[0][i]*(np.log2(h[0][i]+nv)-ls)  for i in range(len(h[0])) ])
    return e
def max_contrast(im):
    av=np.average(im)
    if av: 
        return (np.max(im)-np.min(im))/av
    else:
        return 0.
                
if __name__ == "__main__":        
    l=len(sys.argv)
    if (l<2):
        print("usage: synth_training_set.py <training set size>")
    train_size=int(sys.argv[1])
    e=[]
    c=[]
    smb=[]        
    print("generating images")
    ims=itertools.islice(gen_training_im(),train_size)
    #im,code,detected=gen_training_im()
    print("writing images to disk")
    for i, (im,code,detected) in enumerate(ims):
        #e+=[max_contrast(im)]
        #c+=[detected]
        #smb+=[code]
        name="./train/{:08d}_{}_{}.png".format(i,code,"1" if detected else "0")
        cv2.imwrite(name,im*255.)
        #print(name)
    print ("written {} files to ./train directory".format(i+1))
    """   
    minc=0.4
    et=[e[i] for i in range(len(e)) if c[i] ]
    ent=[e[i] for i in range(len(e)) if not c[i] ]    
    smbt=[smb[i] for i in range(len(e)) if c[i] ]
    smbnt=[smb[i] for i in range(len(e)) if not c[i] ]
    cm=[ 1 for i in range(len(ent)) if ent[i]<minc ]

    print ("minimum contrast in text: {}, in no text: {}".format(np.min(et), np.min(ent)))
    print ("average contrast in text: {}, in no text: {}".format(np.average(et), np.average(ent)))

    print ("proportion of cases with less than minimal contrast in no-text cases: {}".format(len(cm)/len(ent)))
    """
    
    
    
