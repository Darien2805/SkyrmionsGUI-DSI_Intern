# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:31:09 2017

@author: NPStudent
"""
import csv
import scipy.misc
import math
import skimage
import numpy as np
import tkinter
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
from PIL import Image, ImageDraw
from PIL import Image
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu, threshold_local
from skimage.measure import label, regionprops
from scipy.optimize import curve_fit

#finding directory of file   
def file():
    global file_name
    global imgraw
    global img8
    global img8_1
    file_name = filedialog.askopenfilename()
    image_show = Image.open(file_name)
    imgraw = plt.imread(file_name)
    if imgraw.dtype == ('uint16'):
        img8=(imgraw/256).astype('uint8')
    if imgraw.dtype == ('uint8') :
        imgraw=imgraw[:,:,0]
        img8=(imgraw).astype('uint8')
    img8_1 = img8
    scipy.misc.imsave('Rawimg.jpg',imgraw)    
    photo = ImageTk.PhotoImage(Image.open("Rawimg.jpg"))
    label.config(image = photo)
    label.image =photo

    return

def filtering():
    
    global img8
    
    img8 = scipy.ndimage.filters.gaussian_filter(img8, 1.5)
    
    scipy.misc.imsave('Filimg.jpg',img8)    
    filphoto = ImageTk.PhotoImage(Image.open("Filimg.jpg"))
    label.config(image = filphoto)
    label.image = filphoto
    
def globalthresh():
    
    global img8
    global global_thresh
    global binary_global
    
    if tv.get() == 0:
        global_thresh = threshold_otsu(img8)
        e3.delete(0,END)
        e3.insert([0],global_thresh)
    else :
        global_thresh = tv.get()
    binary_global = img8 > global_thresh
    scipy.misc.imsave('Threshold.jpg',binary_global) 
    globalimg = ImageTk.PhotoImage(Image.open("Threshold.jpg"))
    label.config(image = globalimg)
    label.image =globalimg
    binary_global = binary_global*1
    return

def adapthresh():
    
    global img8
    global adaptive_thresh
    global binary_adaptive
    block_size = bs.get()
    offsetval = os.get()
    adaptive_thresh = threshold_local(img8, block_size, offset = offsetval)
    binary_adaptive = img8 > adaptive_thresh
    scipy.misc.imsave('Threshold.jpg',binary_adaptive) 
    adapimg = ImageTk.PhotoImage(Image.open("Threshold.jpg"))
    label.config(image = adapimg)
    label.image = adapimg
    binary_adaptive = binary_adaptive*1
    return

#command to return which def when button is clicked               
def threshold(): 
    
    if v.get() == '2' :
        return globalthresh()
        
    if v.get() == '3' :
        return adapthresh()

#Regioning
def region():
    global binary_global
    global binary_adaptive
    global holding_area
    global holding_cent
    global holding_orient
    global holding_peri
    global holding_rad
    global n
    global im
    global img
    global rightclick
    global leftclick
    
    sizeArr=np.shape(img8_1)
    r=sizeArr[0]
    c=sizeArr[1]
    img = np.zeros([r,c,3], np.uint8)
    for x in range(0,r):
        for y in range(0,c):     
            img[x,y] = img8_1[x,y]
    scipy.misc.imsave('Rgb.jpg',img) 
   
    if v.get() == '2':
        binary = binary_global
    
    if v.get() == '3':  
        binary = binary_adaptive
   
    n = 0
    g = 0
    #convertion to rgb format
    im = Image.open("Rgb.jpg")
    draw = ImageDraw.Draw(im)
    labelimg = skimage.measure.label(binary, connectivity=binary.ndim)
    regions = regionprops(labelimg)
      
    if ar.get() ==0:
        '''
        for reg in regions:
            n = n + 1
            ycent, xcent = reg.centroid
            draw.ellipse((xcent-2,ycent-2,xcent+2,ycent+2), fill = 'red')
        holding_area = np.zeros(n)
        holding_orient = np.zeros(n)
        holding_cent = np.zeros([n,2])
        holding_peri = np.zeros(n)
        holding_rad = np.zeros(n)
        for cell in range(0,n):
            holding_area[cell] = regions[cell].area
            holding_orient[cell] = regions[cell].orientation
            holding_cent[cell]= regions[cell].centroid
            holding_peri[cell] = regions[cell].perimeter
            #finding radius
            radius = math.sqrt(holding_area[cell]/math.pi)
            holding_rad[cell] = radius
        max_rad = np.max(holding_rad)
        '''
        citeria = 15
        for reg in regions:
            if citeria < reg.centroid[0] < np.size(img8_1[0]) - citeria:
                if citeria < reg.centroid[1] < np.size(img8_1[1]) - citeria:
                    n=n+1
                    y0, x0 = reg.centroid
                    draw.ellipse((x0-2,y0-2,x0+2,y0+2), fill = 'red')
        holding_area = np.zeros(n)
        holding_orient = np.zeros(n)
        holding_cent = np.zeros([n,2])
        holding_peri = np.zeros(n)
        holding_rad = np.zeros(n)
        for props in regions:
            if citeria < props.centroid[0] < np.size(img8_1[0]) - citeria:
                if citeria < props.centroid[1] < np.size(img8_1[1]) - citeria:
                    holding_area[g] = props.area
                    holding_orient[g] = props.orientation
                    holding_cent[g]= props.centroid
                    holding_peri[g] = props.perimeter
                    radius = math.sqrt(holding_area[g]/math.pi)
                    holding_rad[g] = radius
                    g= g+1
        max_rad = np.max(holding_rad)
    
    else :
        max_rad = ar.get()
        max_area = (max_rad**2) * math.pi        
        for reg in regions:
            if max_area > reg.area:
                n=n+1
                y0, x0 = reg.centroid
                draw.ellipse((x0-2,y0-2,x0+2,y0+2), fill = 'red')
        holding_area = np.zeros(n)
        holding_orient = np.zeros(n)
        holding_cent = np.zeros([n,2])
        holding_peri = np.zeros(n)
        holding_rad = np.zeros(n)
        for props in regions:
            if max_area > props.area:
                holding_area[g] = props.area
                holding_orient[g] = props.orientation
                holding_cent[g]= props.centroid
                holding_peri[g] = props.perimeter
                radius = math.sqrt(holding_area[g]/math.pi)
                holding_rad[g] = radius
                g= g+1
        
    mean_area = np.mean(holding_area)
    mean_peri = np.mean(holding_peri)
    avg_rad = math.sqrt(mean_area/math.pi)
    e5.delete(0,END)
    e5.insert([0],"%0.2f" %max_rad)   
    #image   
    scipy.misc.imsave('Regioning.jpg',im) 
    regimg = ImageTk.PhotoImage(Image.open("Regioning.jpg"))
    label.config(image = regimg)
    label.image = regimg
    #listing of coordinates
    LB1.delete(3,END)
    LB1.insert(4,"Number of clusters : %0.f" %n)
    LB1.insert(5,"Mean Value of Area : %0.2f"  %mean_area)
    LB1.insert(6,"Mean Value of Perimeter: %0.2f" %mean_peri)
    LB1.insert(7,"Average Radius: %0.2f" %avg_rad)
    
    label.bind('<Motion>',motion)
    label.bind('<Button-3>', right_click)
    rightclick = 0
    label.bind('<Button-1>', left_click)
    leftclick = 0

    return

def motion(event):
    LB2.delete(1,END)  
    LB2.insert(1,"( %s %s )" % (event.x, event.y))
    return

def right_click(event):
    global coor_yx 
    global rightclick

    rightclick = rightclick + 1
    coor_yx = [event.y, event.x]

    LB1.delete(3, END)
    LB1.insert(3, "                    Right Click [Remove]")
    LB1.insert(4, "---------------------------------------------------------")
    LB1.insert(5, "Number of right clicks :  %0.f" %rightclick)
    LB1.insert(6, "Mouse Position clicked :  ( %s %s )"% (event.x, event.y))

    return remove()

def remove():  
    global holding_area
    global holding_cent
    global holding_orient
    global holding_peri
    global holding_rad
    global coor_yx
    global n

    #finding difference between clicked and actual coordinates
    diff = abs(holding_cent - coor_yx)
    #puting in a single array
    #np.size(holding_cent[:,1])
    difference = np.zeros([n,1])
    for ndiff in range (0,n):
        difference[ndiff][0] = diff[ndiff,0] + diff[ndiff,1]
    #finding the smallest value & locate position
    smallest = np.min(difference)

    position = np.where(difference == smallest)
    if np.size(difference[position]) > 1:
        comparison = np.zeros([np.size(difference[position]),1])
        for compare in range(0,np.size(difference[position])):
            comparison[compare][0] = diff[position[compare]][0][0]
        checksmall = np.min(comparison)
        position = np.where (holding_cent == checksmall)
    
    #deleting position
    new_holdingcent = np.delete(holding_cent, position[0],0)
    holding_cent = new_holdingcent
    new_holdingarea = np.delete(holding_area, position[0],0)
    holding_area = new_holdingarea
    new_meanarea = np.mean(new_holdingarea)
    new_holdingorient = np.delete(holding_orient, position[0],0)
    holding_orient = new_holdingorient
    new_holdingperi = np.delete(holding_peri, position[0],0)
    holding_peri = new_holdingperi
    new_meanperi = np.mean(holding_peri)
    new_holdingrad = np.delete(holding_rad, position [0],0)
    holding_rad = new_holdingrad
    newavg_rad = math.sqrt(new_meanarea/math.pi)
    #recreate image
    new_im = Image.open("Rgb.jpg")
    draw = ImageDraw.Draw(new_im)
    for new_cell in range(0,n-1):
        y1,x1 = new_holdingcent[new_cell]
        draw.ellipse((x1-2,y1-2,x1+2,y1+2), fill = 'red')
    scipy.misc.imsave('Regioning.jpg',new_im) 
    new_regimg = ImageTk.PhotoImage(Image.open("Regioning.jpg"))
    label.config(image = new_regimg)
    label.image = new_regimg
    #update n
    n=n-1
    #label
    LB1.insert(7, "New Number of clusters :  %.0f" %n)
    LB1.insert(8, "New Mean Value of Area :  %.2f" %new_meanarea)
    LB1.insert(9, "New Mean Value of Perimeter: %.2f" %new_meanperi)
    LB1.insert(10,"New Mean Value of Radius:  %.2f" %newavg_rad)


def left_click(event):
    global coor_yx 
    global leftclick
    
    leftclick = leftclick + 1
    coor_yx = [event.y, event.x]
    
    LB1.delete(3, END)
    LB1.insert(3, "                         Left Click [Add]")
    LB1.insert(4, "----------------------------------------------------------")
    LB1.insert(5, "Number of left clicks :  %0.f" %leftclick)
    LB1.insert(6,"Mouse Position clicked :( %s %s )" % (event.y, event.x))

    return add()

def add():
    global holding_area
    global holding_cent
    global holding_orient
    global holding_peri
    global holding_rad
    global coor_yx
    global n

    #finding difference between clicked and actual coordinates
    diff = abs(holding_cent - coor_yx)
    #puting in a single array
    difference = np.zeros([n,1])
    for ndiff in range (0,n):
        difference[ndiff] = diff[ndiff,0] + diff[ndiff,1]
    #finding the smallest value & locate position
    smallest = np.min(difference)
    position = np.where(difference == smallest)
    #adding position
    new_holdingcent = np.append(holding_cent, [coor_yx], axis =0)
    holding_cent = new_holdingcent
    new_holdingarea = np.append(holding_area, 0)
    holding_area = new_holdingarea
    new_meanarea = np.mean(new_holdingarea)
    new_holdingorient = np.append(holding_orient, 0)
    holding_orient = new_holdingorient
    new_holdingperi = np.append(holding_peri, 0)
    holding_peri = new_holdingperi
    new_meanperi = np.mean(holding_peri)
    new_holdingrad = np.append(holding_rad,0)
    holding_rad = new_holdingrad
    newavg_rad = math.sqrt(new_meanarea/math.pi)
    #recreate image
    new_im = Image.open("Rgb.jpg")
    draw = ImageDraw.Draw(new_im)
    for new_cell in range(0,n+1):
        y1,x1 = new_holdingcent[new_cell]
        draw.ellipse((x1-2,y1-2,x1+2,y1+2), fill = 'red')
    scipy.misc.imsave('Regioning.jpg',new_im) 
    new_regimg = ImageTk.PhotoImage(Image.open("Regioning.jpg"))
    label.config(image = new_regimg)
    label.image = new_regimg
    #update n
    n=n+1
    #label
    LB1.insert(7, "New Number of clusters :  %.0f" %n)
    LB1.insert(8, "New Mean Value of Area :  %.2f" %new_meanarea)
    LB1.insert(9, "New Mean Value of Perimeter: %.2f" %new_meanperi)
    LB1.insert(10, "New Mean Value of Radius:  %.2f" %newavg_rad)
    return

def next_fit():
    global next_n
    global window
    global holding_loss
    global holding_sig
    global saving
    global last
    
    next_n = next_n + 1
    saving = -1
    last = 10
    
    holding_loss = np.zeros([totalrefitnum,2])
    holding_sig = np.zeros([totalrefitnum,3])
    
    window.destroy()
    
    return fitting(0,0)

def prev_fit():
    global next_n
    global window
    global holding_loss
    global holding_sig
    global saving
    global last
    
    next_n = next_n-1
    saving = -1
    last = 10
    holding_loss = np.zeros([totalrefitnum,2])
    holding_sig = np.zeros([totalrefitnum,3])
    
    window.destroy()
    
    return fitting(0,0)

def fit():
    fitnum = 0
    fitsize = 0

    return fitting(fitnum, fitsize)

def fitwork(fitnum, fitsize):
    
    global r_dist
    global hold_radint

    y0, x0 = holding_cent[next_n]
    img8_1 = img8
    win = 2
    if holding_rad[next_n] < 2 :
        x1 = x0 - 10.0*holding_rad[next_n]
        x2 = x0 + 10.0*holding_rad[next_n]
        y1 = y0 - 10.0*holding_rad[next_n]
        y2 = y0 + 10.0*holding_rad[next_n]
    
    else :
        x1 = x0 - win*holding_rad[next_n]
        x2 = x0 + win*holding_rad[next_n]
        y1 = y0 - win*holding_rad[next_n]
        y2 = y0 + win*holding_rad[next_n]
    
    if fitnum > 0:

        x0 = raw_sig[0][0]
        y0 = raw_sig[1][0]
        x1 = x0- fitsize #- 2*raw_sig[2][0] - fitsize
        y1 = y0- fitsize #- 2*raw_sig[3][0] - fitsize
        x2 = x0+ fitsize #+ 2*raw_sig[2][0] +  fitsize
        y2 = y0+ fitsize #+ 2*raw_sig[3][0] + fitsize
        sigmax = raw_sig[2][0]
        sigmay = raw_sig[3][0]

    else:
        sigmax = holding_rad[next_n]
        sigmay = holding_rad[next_n]

    if x1 < 0 :
            x1 = np.float64(0)
       
    if x2 > im.size[0] :
            x2 = np.float64(im.size[0]-1)
                
    if y1 < 0 :
            y1 = np.float64(0)
            
    if y2 > im.size[1]:
            y2 = np.float64(im.size[1]-1)
    
    holding_points[0][0] = x0
    holding_points[1][0] = y0
    holding_points[2][0] = x1
    holding_points[3][0] = y1
    holding_points[4][0] = x2
    holding_points[5][0] = y2
    #Extracting values from line
    x,y = np.arange(x1.astype(np.int),(x2+1).astype(np.int)),np.arange(y1.astype(np.int),(y2+1).astype(np.int))
    x,y = np.meshgrid(x,y)
    
    z=img8_1[y.astype(np.int),x.astype(np.int)]
    #image
    if fitnum > 0 :
        a_low= 0
        a_high = np.inf
        mux_low = x0-holding_rad[next_n]#np.size(z[:,0])/2 - 2*holding_rad[next_n]
        mux_high = x0+holding_rad[next_n] #np.size(z[:,0])/2 + 2*holding_rad[next_n]
        muy_low = y0-holding_rad[next_n]#np.size(z[0,:])/2 - holding_rad[next_n]
        muy_high = y0+holding_rad[next_n]#np.size(z[0,:])/2 + holding_rad[next_n]
        sigma_low = 0
        sigma_high = 2*holding_rad[next_n]#(2*(2*np.sqrt(2*(math.log(2)))))#np.sqrt(np.size(z)) /2
        off_low = np.min(np.min(z,axis=0))
        off_high =np.max(np.min(z,axis=0))
    else : 
        a_low= 0
        a_high = np.inf
        mux_low = 0
        mux_high =  np.inf
        muy_low = 0
        muy_high = np.inf
        sigma_low = 0
        sigma_high = np.inf
        off_low = 0
        off_high =np.inf
        
    #param_bounds=([0,-np.inf, -np.inf, -np.inf, -np.inf, off_low],[255, np.inf, np.inf, np.inf, np.inf, off_high])
    param_bounds=([a_low, mux_low, muy_low, sigma_low, sigma_low, off_low],[a_high, mux_high, muy_high, sigma_high, sigma_high, off_high])
    #a, mux, muy, sigmax, sigmay, off
    #try:
        
    popt,pcov = curve_fit(twoD_gaussian, (x, y), z.reshape((len(x))*(len(y[0]))), p0=[np.max(z),x0,y0,sigmax,sigmay, np.min(z)], maxfev = 1000, bounds = param_bounds)
    #popt,pcov = curve_fit(twoD_gaussian, (x, y), z.reshape((len(x))*(len(y[0]))), p0=[np.max(z),x0,y0,sigmax,sigmay, np.min(z)])
    #param_bounds=([a_low, mux_low, muy_low, sigma_low, sigma_low, off_low],[a_high, mux_high, muy_high, sigma_high, sigma_high, off_high])
    #except:
 
        #pass
    
    #sigmax = sigmay
    popt[4]=popt[3]
    data_fitted = twoD_gaussian((x, y), *popt)
    #line cuts
    
    #middle
    zm = z[(y0-y1).astype(np.int)]
    zm = zm.astype('uint16')
    
    #btm
    zbn = (y0+raw_sig[3][0]-y1).astype(np.int)
    if zbn < 0:
        zb = z[0]
        
    elif zbn >= np.size(z[:,0]):
        #zb = z[(y0+0.3*raw_sig[3][0]-y1).astype(np.int)]
        zb = z[np.size(z[:,0])-1]
    else:
        zb = z[(zbn).astype(np.int)]
    zb = zb.astype('uint16')

    #top
    ztn = (y0-raw_sig[3][0]-y1).astype(np.int)
    
    if ztn < 0:
        
        zt = z[0]
        
    elif ztn >= np.size(z[:,0]):
        #zt = z[(y0-0.3*raw_sig[3][0]-y1).astype(np.int)]
        zt = z[np.size(z[:,0])-1]
    
    else:
        zt = z[(ztn).astype(np.int)]
    zt = zt.astype('uint16')
    
    lossfit = loss(z,data_fitted.reshape(np.size(z[:,1]),np.size(z[0,:])))
    #lossfitting = lossfit[0]/np.size(z)
    lossfitting = (np.sum(np.abs(z - data_fitted.reshape(np.size(z[:,1]),np.size(z[0,:])))))/ np.size(z)
    fwhmx = fullwhm(popt[3])
    fwhmy = fullwhm(popt[4])
    vert_z1 = z[:,0]
    horiz_z1 = z[0,:]
    vert_z2 = z[:,np.size(z[0,:])-1]
    horiz_z2 =z[np.size(z[:,0])-1,:]
    repeated = z[0,0]+z[np.size(z[:,0])-1,0]+z[np.size(z[:,0])-1,np.size(z[0,:])-1]+z[0,np.size(z[0,:])-1]
    #intensity = np.sum(z)/np.size(z)
    max1=np.max(vert_z1)
    max2=np.max(horiz_z1)
    max3=np.max(vert_z2)
    max4=np.max(horiz_z2)
    maxI=np.mean(np.array([max1,max2,max3,max4]))
    intensity = (np.sum(vert_z1)+np.sum(horiz_z1)+np.sum(vert_z2)+np.sum(horiz_z2)-repeated) / (np.size(vert_z1)+np.size(horiz_z1)+np.size(vert_z2)+np.size(horiz_z2)-4)#maxI
    '''
    if last == saving + 2:
        print('~saving image~')  
        circle_rad = round(len(z[0,:])/2)
        hold_radint = np.zeros([circle_rad, 3])
        r_dist = np.sqrt((x-popt[1])**2 + (y-popt[2])**2)
        for circle_n in range (0,circle_rad):
            r_i=(r_dist <= circle_n+1) & (r_dist>circle_n)
            #plt.imshow(r_i)
            circle_img = z[(r_dist <= circle_n+1) & (r_dist>circle_n)]
            hold_radint[circle_n] = [circle_n,np.sum(circle_img)/np.size(circle_img), np.max(circle_img)]
        
        r_th = np.min(hold_radint[:,2])
        r_n = [np.where(hold_radint[:,2] == r_th)][0][0][0]
        circle_img = z*(r_dist<r_n)
        plt.imshow(circle_img)
        plt.colorbar()
        plt.savefig("FinalRawImage_sk_" + str(next_n)+ "_fit_" + str (len(z[0,:])) +".jpg", bbox_inches='tight')
        plt.close()
        
        
        plt.plot(hold_radint[:,0],hold_radint[:,2])
        plt.savefig("RadialIntensity(Max)_sk_" + str(next_n)+ '.jpg', bbox_inches='tight')
        plt.close()
        plt.plot(hold_radint[:,0],hold_radint[:,1])
        plt.savefig("RadialIntensity_sk_" + str(next_n)+ '.jpg', bbox_inches='tight')
        plt.close()
        
    
    plt.close()
    plt.imshow(lossfit[1], vmin = np.min(z), vmax = np.max(z))
    plt.colorbar()
    plt.savefig("LossImage_sk_" + str(next_n)+ "_fit_" + str (len(z[0,:])) +".jpg", bbox_inches='tight')
    plt.close()
    plt.imshow(z)
    plt.colorbar()
    plt.savefig("RawImage_sk_" + str(next_n)+ "_fit_" + str (len(z[0,:])) +".jpg", bbox_inches='tight')
    plt.close()
    plt.imshow(data_fitted.reshape(np.size(z[:,1]),np.size(z[0,:])),vmin = np.min(z), vmax = np.max(z))
    plt.colorbar()
    plt.savefig("FittedImage_sk_" + str(next_n)+ "_fit_" + str (len(z[0,:])) +".jpg", bbox_inches='tight')
    plt.close()
    
    '''
    if  saving+2 > last:
        
        plt.imshow(z)
        plt.colorbar()
        plt.savefig("FinalRawImage_sk_" + str(next_n)+ "_fit_" + str (len(z[0,:])) +".jpg", bbox_inches='tight')
        plt.close()
    
    if saving + 2 < last:
        saving_cell[saving][0] = len(z[0,:])
        saving_cell[saving][1] = popt[4]
        saving_cell[saving][2] = popt[5]
        saving_cell[saving][3] = x0
        saving_cell[saving][4] = y0
        saving_cell[saving][5] = lossfitting
     
        with open("Data_sk_"+ str(next_n)+ ".txt", "w+") as text_file:
            text_file.write('width')
            text_file.write('\t')
            text_file.write('sigma')
            text_file.write('\t')
            text_file.write('offset')
            text_file.write('\t')
            text_file.write('x0')
            text_file.write('\t')
            text_file.write('y0')
            text_file.write('\t')
            text_file.write('Loss per pix')
            text_file.write('\n')
            for reading in range (0,countrefitnum):
                text_file.write('%.6s' %saving_cell[reading][0])
                text_file.write('\t')
                text_file.write('%.6s' %saving_cell[reading][1])
                text_file.write('\t')
                text_file.write('%.6s' %saving_cell[reading][2])
                text_file.write('\t')
                text_file.write('%.6s' %saving_cell[reading][3])
                text_file.write('\t')
                text_file.write('%.6s' %saving_cell[reading][4])
                text_file.write('\t')
                text_file.write('%.6s' %saving_cell[reading][5])
                text_file.write('\n')
    
    return (z,data_fitted, zb, zbn, zm, zt, ztn, lossfit, lossfitting, fwhmx, fwhmy, popt, intensity)

def fitting(fitnum, fitsize):
    
    global window
    global holding_cent
    global holding_rad
    global holding_points
    global raw_sig
    global next_n

    window = Toplevel(master = root)
    plt.close()
    fit_img = Image.open("Regioning.jpg")
    draw1 = ImageDraw.Draw(fit_img)
    holding_points = np.zeros([6,1])

    if next_n in range (0,n):
        y_0, x_0 = holding_cent[next_n]
       
        draw1.ellipse((x_0-2,y_0-2,x_0+2,y_0+2), fill = 'green')
        
        z,data_fitted, zb, zbn, zm, zt, ztn, lossfit, lossfitting , fwhmx, fwhmy, popt, intensity= fitwork(fitnum, fitsize)
        y1 = holding_points[3][0]
        if fitnum == 0:
            raw_sig[0] = popt[1]
            raw_sig[1] = popt[2]
            raw_sig[2] = popt[3]
            raw_sig[3] = popt[4]
            zbn = raw_sig[1]-y1+raw_sig[3]
            ztn = raw_sig[1]-y1-raw_sig[3]
        #marking green dot
        scipy.misc.imsave('Fitting.jpg',fit_img) 
        fit = ImageTk.PhotoImage(Image.open("Fitting.jpg"))
        label.config(image = fit)
        label.image = fit
        
        #NEW WINDOW
        #frame for line cut graph
        frame4 = LabelFrame(window,text = 'Gaussian Fit')
        frame4.grid(row = 4, column = 0, pady = 10, padx = 10)
        #frame for refitting
        refitframe = LabelFrame(window,text = 'Gaussian Fit')
        refitframe.grid(row = 4, column = 1, pady = 10, padx = 10)
        #Label :Graph
        Lregion = Label(window, text = 'Region', fg= "dark blue", font = "Verdaba 12 bold")
        Lregion.grid(row = 0, column = 0)
        Lgraph = Label(window, text = 'Graph', fg = "dark blue", font = "Verdaba 12 bold")
        Lgraph.grid(row = 0, column = 3, padx = 200)
        Lgaureg = Listbox(window, relief = RAISED, bd = 5, height = 14, width = 40, font = 'Helvetica 10 italic')
        Lgaureg.grid(row = 3, column = 1, padx = 10)
        Lgaureg.insert(0, "========================================")
        Lgaureg.insert(1, "                                     FIT")
        Lgaureg.insert(2, "========================================")

        #imageraw
        labelimgraw = Label(window)
        labelimgraw.grid(row = 1, column = 0)
        #seperator
        sep = Listbox(window, relief = RAISED, bd = 5, height = 1, width = 100, bg = 'light blue')
        sep.grid(row=2, column = 0, columnspan = 3)
        sep1 = Listbox(window, relief = RAISED, bd=5, height =14, width = 2, bg = 'light blue')
        sep1.grid(row = 1, column = 2, padx = 10)
        sep2 = Listbox(window, relief = RAISED, bd=5, height =14, width = 2, bg = 'light blue')
        sep2.grid(row = 3, column = 2, padx = 10)
        #imagefit
        labelimgthr = Label(window)
        labelimgthr.grid(row = 3, column = 0)
        #Next fitting
        Nbt = Button(frame4, text = 'Next Fit', command = next_fit, width = 10,font = 'Times 12 bold', bg = 'light green')
        Nbt.grid(row = 1, column = 2, padx = 10)
        #Previous fitting
        Nbt1 = Button(frame4, text = 'Previous Fit', command = prev_fit, width = 10, font = 'Times 12 bold', bg = 'light green')
        Nbt1.grid(row = 1, column = 1, padx = 10)
        
        #Refit button
        Refitbt = Button(refitframe, text = "Accurate Refit", command = refit, width = 15,font = 'Times 12 bold', bg = 'cyan') 
        Refitbt.grid(row = 1, column = 1, padx = 5)
        
        #Refit button
        Refitbt = Button(refitframe, text = "All Refit", command = allrefit, width = 15,font = 'Times 12 bold', bg = 'cyan') 
        Refitbt.grid(row = 1, column = 2, padx = 5)
        
        #lossgraphimg
        lossgraph = Label(window)
        lossgraph.grid(row = 1, column = 1)        
        #graphimg
        labelgraph = Label(window)
        labelgraph.grid(row = 1, column = 3, rowspan = 4, sticky = W+E+N+S , padx = 10)
        
        #lisbox insert
        Lgaureg.insert(3, "Skyrmion : %0.0f" %next_n)
        Lgaureg.insert(4, "Amplitude : %0.4f" %popt[0])
        Lgaureg.insert(5, "Sigma X : %0.4f" %popt[3])
        Lgaureg.insert(6, "Sigma Y : %0.4f" %popt[4])
        Lgaureg.insert(7, "Offset : %0.4f" %popt[5])
        Lgaureg.insert(8, "Full Width Half Max X: %0.4f" %fwhmx)
        Lgaureg.insert(9, "Full Width Half Max Y: %0.4f" %fwhmy)
        Lgaureg.insert(10, "Loss between Raw data and Fitting : %0.4f" %lossfit[0])
        Lgaureg.insert(11, "Loss (per pixel) : %0.4f" %lossfitting)
        
        #plot of raw singular region
        plt.figure(figsize=(4,4))
        plt.imshow(z)
        plt.colorbar()
        plt.plot([0, np.size(z[1])-1], [zbn ,zbn], 'red')
        plt.plot([0, np.size(z[1])-1], [raw_sig[1]-y1 , raw_sig[1]-y1], 'orange')
        plt.plot([0, np.size(z[1])-1], [ztn , ztn], 'blue')
        plt.title("Data")
        rawreg = plt.savefig('Rawreg.jpg', bbox_inches='tight')
        rawreg = ImageTk.PhotoImage(Image.open("Rawreg.jpg"))
        labelimgraw.config(image = rawreg)
        labelimgraw.image = rawreg
        plt.close()
        
        #plotting of fitted singular region
        plt.figure(figsize=(4,4))
        data_refit = data_fitted.reshape(np.size(z[:,1]),np.size(z[0,:]))
        plt.imshow(data_refit, vmin = np.min(z), vmax = np.max(z))
        plt.colorbar()
        plt.plot([0, np.size(z[1])-1], [zbn , zbn], 'red')
        plt.plot([0, np.size(z[1])-1], [raw_sig[1]-y1, raw_sig[1]-y1], 'orange')
        plt.plot([0, np.size(z[1])-1], [ztn, ztn], 'blue')
        plt.title("Fit")
        gaureg = plt.savefig('Gaureg.jpg', bbox_inches='tight')
        gaureg = ImageTk.PhotoImage(Image.open("Gaureg.jpg"))
        labelimgthr.config(image = gaureg)
        labelimgthr.image = gaureg
        plt.close()
        
        #plotting of graph
        plt.figure(figsize=(8,10))
        #topline
        plt.plot((zt + np.max(zb) + np.max(zm)) , 'g*', label = 'raw')
        plt.plot(((data_refit[(raw_sig[1][0]-raw_sig[3][0]-y1).astype(np.int)]) + np.max(zb)+ np.max(zm)), 'blue', label = 'Fit')
        #midline
        plt.plot(zm + np.max(zb), 'g*', label = 'raw')
        plt.plot(((data_refit[(raw_sig[1][0]-y1).astype(np.int)])+ np.max(zb)), 'orange', label = 'Fit')
        #btmline
        plt.plot(zb, 'g*', label = 'raw')
        plt.plot((data_refit[(raw_sig[1][0]+raw_sig[3][0]-y1).astype(np.int)]), 'red', label = 'Fit')
        
        plt.legend(bbox_to_anchor=(1, 1))
        
        graph = plt.savefig('Graph.jpg', bbox_inches='tight')
        graph = ImageTk.PhotoImage(Image.open("Graph.jpg"))
        labelgraph.config(image = graph)
        labelgraph.image = graph
       
        plt.close()
        
        plt.figure(figsize=(4,4))
        plt.imshow(lossfit[1], vmin = 0, vmax = 255)
        plt.colorbar()
        plt.title("Loss between data and fit")
        losgraph = plt.savefig('LossGraph.jpg', bbox_inches='tight')
        #plt.plot(holding_loss[:,0], holding_loss[:,1], 'r*')
        #plt.plot(holding_loss[:,0],holding_loss[:,1], 'r*')
        #plt.plot(holding_sig[:,0], holding_sig[:,2], 'g*')
        plt.savefig('LossGraph'+ str(next_n)+ ".jpg", bbox_inches='tight')
        plt.savefig('LossGraph.jpg', bbox_inches='tight')
        losgraph = ImageTk.PhotoImage(Image.open("LossGraph.jpg"))
        lossgraph.config(image = losgraph)
        lossgraph.image = losgraph
        plt.close()
        plt.plot(holding_loss[:,0], holding_sig[:,1], 'g*')
        #plt.plot(holding_sig[:,0], holding_sig[:,1], 'g*')
        plt.savefig('SigmaX'+ str(next_n) + '.jpg', bbox_inches='tight')
        plt.close()
        plt.plot(holding_loss[:,1], holding_sig[:,1], 'b*')
        plt.savefig('SigmaX&Loss'+ str(next_n) + '.jpg', bbox_inches='tight')
        plt.close()
        plt.plot(holding_int[:,0], holding_int[:,1], 'y*')
        plt.savefig('Intensity'+ str(next_n) + '.jpg', bbox_inches='tight')
        plt.close()
        

        window.mainloop()
        
    else:
        return error()

def twoD_gaussian(xdata_tuple, a, mux, muy, sigmax, sigmay, off):

    (x,y) = xdata_tuple
    gau = a*np.exp(-(((x-mux)**2/(2.0*sigmax**2))+((y-muy)**2/(2.0*sigmay**2)))) + off
    
    return gau.ravel()

def fullwhm(sigma):
    
    fwhm = 2*np.sqrt(2*(math.log(2)))*(sigma)
    
    return(fwhm)

def loss(z, datafitted):
    global diff_loss
    global submissionj
    diff_loss = (z - datafitted)
    division = (diff_loss **2)/(z+1)
    loss = np.sum(division)
    
    return(loss,diff_loss)

def refit():
    global holding_loss
    global holding_sig
    global holding_int
    global saving_cell
    global countrefitnum
    global saving
    global raw_sig
    global z_axis
    global allrefitn
    global last
    
    sigma_0=raw_sig[3][0];
    window.destroy()
    refitnum = 0
    countrefitnum = 0
    end = (3*holding_rad[next_n]).astype(np.int)#(6*sigma_0-sigma_0).astype(np.int)
    start =(0.5*holding_rad[next_n]).astype(np.int)#(0.5*sigma_0).astype(np.int)#-3#((sigma_0/4)-sigma_0).astype(np.int)
    
    last = end - start
    
    z_axis = np.arange(start,end)
    
    for next_refit in range(start,end):
        countrefitnum = countrefitnum + 1
    holding_loss = np.zeros([countrefitnum,2])
    holding_sig = np.zeros([countrefitnum,3])
    saving_cell = np.zeros([countrefitnum,6])
    holding_int = np.zeros([countrefitnum,3])
    
    for next_refit in range(start,end):
        
        z,data_fitted, zb, zbn, zm, zt, ztn, lossfit, lossfitting , fwhmx, fwhmy, popt, intensity = fitwork(1, next_refit)
        holding_loss[refitnum] = [len(z[0,:]),lossfitting]
        holding_sig[refitnum] = [len(z[0,:]), popt[3], popt[4]]
        holding_int[refitnum] = [len(z[0,:]), intensity, next_refit]
        refitnum = refitnum + 1
        saving = saving + 1

    if allrefitn == 0:
        min_int = np.min(np.min(holding_int[:,1]))
        loc_minint = holding_int[np.where(holding_int[:,1] == min_int)]
        #fitwork(1, loc_minint[0][3]-1)
        fitting(1, loc_minint[0][2])
        allrefitn = 0
        saving = -1
        last = 10


    if saving + 1 >= last:
        min_int = np.min(np.min(holding_int[:,1]))
        loc_minint = holding_int[np.where(holding_int[:,1] == min_int)]
        #fitwork(1, loc_minint[0][3]-1)
        fitwork(1, loc_minint[0][2])

 
def allrefit():
    global next_n
    global allrefitn
    global saving
    global data
    
    allrefitn = allrefitn + 1
    n=3
    for alln in range (0,n):
        next_n = alln
        fitnum = 0
        z,data_fitted, zb, zbn, zm, zt, ztn, lossfit, lossfitting , fwhmx, fwhmy, popt, intensity= fitwork(fitnum, 0)
        if fitnum == 0:
            raw_sig[0] = popt[1]
            raw_sig[1] = popt[2]
            raw_sig[2] = popt[3]
            raw_sig[3] = popt[4]
        
        try:
            refit()
        except:
            pass
        saving = -1
        plt.plot(holding_loss[:,0],holding_loss[:,1], 'r*')
        plt.savefig('LossGraph'+ str(next_n)+ ".jpg", bbox_inches='tight')
        plt.close()
        plt.plot(holding_loss[:,0], holding_sig[:,1], 'g*')
        plt.savefig('SigmaX'+ str(next_n) + '.jpg', bbox_inches='tight')
        plt.close()
        plt.plot(holding_loss[:,1], holding_sig[:,1], 'b*')
        plt.savefig('SigmaX&Loss'+ str(next_n) + '.jpg', bbox_inches='tight')
        plt.close()
        plt.plot(holding_int[:,0], holding_int[:,1], 'y*')
        plt.savefig('Intensity'+ str(next_n) + '.jpg', bbox_inches='tight')
        plt.close()
    
    for readingfiles in range (0,n):
        with open("Data_sk_"+ str(readingfiles)+ ".txt", "r") as file:
            data = []
            firstline = file.readline()
            reader = csv.reader(file, delimiter='\t')
            for line in reader:
                data.append(line)
            data = np.float64(data)
            data = data[np.argsort(data[:,0])]
            norm = (data[:,5] - np.min(data[:,5]))/(np.max(data[:,5])-np.min(data[:,5]))
            plt.plot(data[:,0], norm)
            plt.savefig('AllLoss.jpg', bbox_inches = 'tight')
    plt.close()
    for readingfiles in range (0,n):
        with open("Data_sk_"+ str(readingfiles)+ ".txt", "r") as file:
            data = []
            firstline = file.readline()
            reader = csv.reader(file, delimiter='\t')
            for line in reader:
                data.append(line)
            data = np.float64(data)
            data = data[np.argsort(data[:,0])]
            normsig = (data[:,1] - np.min(data[:,1]))/(np.max(data[:,1])-np.min(data[:,1]))
            plt.plot(data[:,0], normsig)
            plt.savefig('Allsigma.jpg', bbox_inches = 'tight')
            
    if alln == n-1:
        messagebox.showinfo("Information", "Completed")
        
        if True:
            allrefitn=0
    
    next_n = 0
        
def error():
    global next_n
    if next_n<0:
        next_n = next_n + 1
    if next_n>0:
        next_n = next_n - 1
    messagebox.showerror("Error", "Number not in range")
    window.destroy()
    return 

global file_name
global imgraw
global img8
global next_n
global holding_loss
global holding_sig
global holding_int
global raw_sig
global saving
global saving_cell
global allrefitn
global countrefitnum
global last

root = Tk()
root.geometry()
bs = IntVar()
os = IntVar()
tv = IntVar()
ma = IntVar()
ar = IntVar()
next_n = 0

saving = -1
totalrefitnum = 21
holding_loss = np.zeros([totalrefitnum,2])
holding_sig = np.zeros([totalrefitnum,3])
saving_cell = np.zeros([totalrefitnum,6])
holding_int = np.zeros([totalrefitnum,2])
raw_sig = np.zeros([4,1])
allrefitn = 0
countrefitnum = 0
last = 10
#frame for picture & parameters
frame1 = LabelFrame(root, width=100, height = 300, text = 'Image Processing')
frame1.pack(fill= 'both', expand= 'yes')
#frame1.grid(ipadx=200, ipady=200)
#frame for buttons/entries
frame2 = Frame(root, width = 50, height = 50)
frame2.pack(fill = 'both', expand = 'yes')
#frame for mouse events
frame3 = LabelFrame(frame1, text = 'Moust Events')
frame3.grid(row=3, column =1, padx = 30)

#Image Label
label = Label(frame1)  
label.grid(row=1, column=1)

#Label : Picture
L = Label(frame1, text = 'Picture', fg= "dark blue", font = "Verdaba 14 bold")
L.grid(row=0, column=1)
#labels for mouse events
L2 = Label(frame3, text = "Left click to add region", fg = "red", font = "Verdaba 10 bold")
L2.grid(row=3, column =2, padx = 50)
L3 = Label(frame3, text = "Right click to remove region", fg = "red", font = "Verdaba 10 bold")
L3.grid(row=4, column = 2, padx = 50)

#Listbox
LB1 = Listbox(frame1, relief = RAISED, bd = 5, height = 13, width = 38)
LB1.grid(row = 1, column = 2, padx = 15, pady = 10)
LB1.insert(0, "=====================================")
LB1.insert(1, "                           PARAMETERS")
LB1.insert(2, "=====================================")
#mouse position
LB2 = Listbox(frame1, relief = RAISED, bd = 5, height = 2, width = 24)
LB2.grid(row = 0, column = 2, padx = 5, pady = 5)
LB2.insert(0,"Mouse Position( x , y ):")


#opening file button
Bt = Button (frame2, text='File Open', command= file, width = 12, font = "Times 12 bold", bg = 'orange')
Bt.grid(row = 0, column = 0)

#Filter Button
Filtbt = Button(frame2, text = 'Filter Threshold', command = filtering, width = 12, font = 'Times 12 bold', bg = 'cyan')
Filtbt.grid(stick = W)

#listing of radio button
MODES = [("Global", "2"), ("Adaptive", "3")]
v = StringVar()
v.set("0") # initialize
for text, mode in MODES:
    b = Radiobutton(frame2, text=text, variable=v, value=mode)
    b.grid(sticky = W)

#Threshold Button    
Tbt = Button(frame2, text = 'Threshold', command = threshold, width = 12, font = 'Times 12 bold', bg = 'light blue')
Tbt.grid(stick = W)

#Regioning
Rbt = Button(frame2, text = 'Regioning', command = region, width = 12, font = 'Times 12 bold', bg = 'red')
Rbt.grid(sticky =W)

#Fitting
Fbt = Button(frame2, text = 'Fitting' , command = fit, width = 12, font = 'Times 12 bold', bg = 'yellow')
Fbt.grid(sticky = W)

#Adaptive Labels & Entries
adapl1 = Label(frame2, text = 'Blocksize number (odd) =', font = 'Times 10 bold', fg = 'dark blue')
adapl1.grid(row = 3, column = 2, padx = 10) 
e1 = Entry(frame2, textvariable = bs, relief = RAISED, bd = 5)
e1.grid(row = 3, column = 3)
#Each threshold value is the weighted mean of the local neighborhood minus an offset value.
adapl2 = Label(frame2, text = 'Offset value =', font = 'Times 10 bold', fg = 'dark blue')
adapl2.grid(row = 3, column = 4)
e2 = Entry(frame2, textvariable = os, relief = RAISED, bd = 5)
e2.grid(row = 3, column = 5, padx = 10)
#Global Labels & Entries
globl1 = Label(frame2, text = 'Threshold value =', font = 'Times 10 bold', fg = 'dark blue')
globl1.grid(row = 2, column = 2, padx = 10)
e3 = Entry(frame2, textvariable = tv, relief = RAISED, bd = 5)
e3.grid(row =2, column =3)

#Max Radius
avgrl = Label(frame2, text = "Maximum Radius =", font = 'Times 8 bold', fg = 'dark blue')
avgrl.grid(row = 5, column = 2, padx = 10)
e5 = Entry(frame2, textvariable = ar, relief = RAISED, bd = 5)
e5.grid(row= 5, column = 3)

root.mainloop()