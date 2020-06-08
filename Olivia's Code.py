#running instructions: python full_code.py

#packages
from PIL import Image
from numpy import asarray
from scipy.fft import fft, ifft
import numpy as np
import os, glob

#lines 9 - 12: remove one line from .ps file that creates borders around desired plot. This gets rid of tick marks that intrude on face of sub-band plot.

for ps_file in (glob.glob("/home/ory001/green_bank_work/data/*ps")): #loops through .ps files

    file = open(ps_file, 'r+') #opens file
    lines = file.readlines()  #stores lines of file
    i = 0  #starts counter
    for num, line in enumerate(lines): #loops through lines in file
        if "gsave newpath" in line:  #checks for occurrences of line that creates borders
           i += 1 #adds to counter when line occurs
           if i == 3: #when the third occurrence (the one associated with the sub_band plot) of the line is found:
              del lines[num]        #remove that line
    file.close() #close file

    new_file = open(ps_file, "w+") #lines 22 - 27 reopen file and update file with line removed

    for line in lines:
        new_file.write(line)

    new_file.close()

  #lines 30-33 create .png file from updated .ps
    root = ps_file[:-2]
    pngfile = root + 'png'
    os.system('convert ' + ps_file + ' ' + pngfile)



    #lines 37-45 open png file using pillow, crop the image to only encompass sub-band plot, and convert image into python array
    image = Image.open(pngfile).convert("L") #opens png
    box = (158, 252, 374, 413) #defines where sub-band plot is
    #box = (340, 178, 513, 410) #uncomment for .png files from GBNCC website

    cropped_image = image.crop(box) #crop image

    #cropped_image.save("cropped_image.png")

    new = asarray(cropped_image) #converts image into array


    #lines 49 - 55 perform mathmatical operations on array
    fft = np.fft.fft2(new) #takes fft of both dimensions of array (need to talk about this, as I am not sure how the double transpose is working in the MATLAB code)
    abs_value = abs(fft) #takes absolute value of all components of fft
    unit_vec = np.divide(fft,abs_value) #creates unit vectors
    half = int((len(unit_vec)-1)/2) #finds half the length of the array (-1 because the array is an odd number)
    omegas = sum(unit_vec[2:half:2]) #takes the sum of every other omega starting at unit_vec[2] and going to half of the length ofthe array
    nw = len(omegas) #finds length of omegas
    Pow = abs(omegas)**2 #finds rms of omegas
    SNR = (sum(Pow) - (new.shape[1])*nw)/((new.shape[1])*((new.shape[1]) - 1)*nw)**0.5 #finds SNR, where new.shape[1] is the nchannels

    print(SNR, ps_file) #print SNR