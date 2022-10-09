# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 16:52:55 2022

@author: Lukas Pezenka
"""

import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
#from datetime import datetime
import math
import statistics
import csv

def file_len(filename):
    with open(filename) as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

def column(matrix, i):
    return [row[i] for row in matrix]

def findNextDistinct(a, i, y):
    j = 0
    for x in range(i, len(a)):
        if x != y:
            return x,j
        j = j + 1

def FindPeak(a, time_data, axis):
    """
    Find peak acceleration

    Find peak acceleration across all axes and return the corresponding index 

    Parameters
    ----------
    a : float[]
        Acceleration data.
    time_data : float[]
        Time data.
    axis : int
        Relevant axis 0=x, 1=y, 2=z.

    Returns
    -------
    index_max : int
        Index of max acceleration along relevant axis.

    """
    
    ##### find Peak Acceleration
    xAxis = column(a,0)
    yAxis = column(a,1)
    zAxis = column(a,2)
    aAxis = column(a,3)
    
    dominant_axis = column(a, axis)
    
    #index_max_x = max(range(len(xAxis)), key=xAxis.__getitem__)
    #index_max_y = max(range(len(yAxis)), key=yAxis.__getitem__)
    #index_max_z = max(range(len(zAxis)), key=zAxis.__getitem__)
    index_max_d = np.argmax(dominant_axis) #max(range(len(dominant_axis)), key=dominant_axis.__getitem__)
    
    #index_max = max(index_max_x, index_max_y)
    #index_max = max(index_max, index_max_z)
    #index_max = index_max_x
    #index_max = index_max_y
    index_max = index_max_d
    
    
    #xAxis = column(a[index_max - 30: index_max + 30],0)
    #yAxis = column(a[index_max - 30: index_max + 30],1)
    #zAxis = column(a[index_max - 30: index_max + 30],2)
    #aAxis = column(a[index_max - 30: index_max + 30],3)
    #td = time_data[index_max - 30: index_max + 30]
    
    #plt.plot(td, xAxis, 'b', label="ax")
    #plt.plot(td, yAxis, 'g', label="ay")
    #plt.plot(td, zAxis, 'r', label="az")
    #plt.plot(td, aAxis, 'k', label="aa")

    plt.plot(time_data[index_max - 30: index_max + 30], a[index_max - 30: index_max + 30, 0], 'b', label='x')
    plt.plot(time_data[index_max - 30: index_max + 30], a[index_max - 30: index_max + 30, 1], 'g', label='y')
    plt.plot(time_data[index_max - 30: index_max + 30], a[index_max - 30: index_max + 30, 2], 'r', label='z')
    plt.plot(time_data[index_max - 30: index_max + 30], a[index_max - 30: index_max + 30, 3], 'k', label='absolute')
    
    plt.plot(time_data[index_max], a[index_max, 0], 'b*')
    plt.plot(time_data[index_max], a[index_max, 1], 'g*')
    plt.plot(time_data[index_max], a[index_max, 2], 'r*')
    plt.plot(time_data[index_max], a[index_max, 3], 'k*')
    
    plt.title('Peak Area')
    plt.xlabel('time')
    plt.ylabel(yaxis_label)
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    return index_max

########################
#### Find beginning of punch; i.e., first moment when dominant axis acceleration surpasses user-defined threshold
########################
def FindStart(a, time_data, threshold, dominant_axis):
    ##### Find movement initiation
    #xm = a[index_max, 0]
    #ym = a[index_max, 1]
    #zm = a[index_max, 2]
    #threshold = 0.00
    i = 0
    
    #slope_threshold = 0.03
    #slope = 1
    #while (slope > slope_threshold):
    #    i = i + 1
        
        
    #    a1 = a[index_max - i, 0]
    #    a0 = a[index_max - i - 1, 0]
        
    #    t1 = time_data[index_max - i]
    #    t0 = time_data[index_max - i - 1]
        
    #    slope = (t1 - t0)/(a1 - a0)
        
    #    print("a1: " + str(a1) + " a0: " + str(a0) + " t1: " + str(t1)+" t0: " + str(t0) + " slope: "+str(slope))
        
    
    #while ((xm > threshold or ym > threshold or zm > threshold) and (index_max - i > 0.0)):
    #dominant_axis = 1
    dominant = a[index_max, dominant_axis]
    while (dominant > threshold):
        i = i + 1
        #xm = a[index_max - i, 0]
        #ym = a[index_max - i, 1]
        #zm = a[index_max - i, 2]
        dominant = a[index_max - i, dominant_axis]
        
    xAxis = column(a[index_max - i: index_max],0)
    yAxis = column(a[index_max - i: index_max],1)
    zAxis = column(a[index_max - i: index_max],2)
    aAxis = column(a[index_max - i: index_max],3)
    
    td = time_data[index_max - i: index_max]
    
    plt.plot(td, xAxis, 'b', label="ax")
    plt.plot(td, yAxis, 'g', label="ay")
    plt.plot(td, zAxis, 'r', label="az")
    plt.plot(td, aAxis, 'k', label="aa")
    
    
    plt.title('Intitiation to peak')
    plt.xlabel('time')
    plt.ylabel(yaxis_label)
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    
    initiation = index_max - i
    return initiation

########################
#### Find end of punch; i.e., first sample where the acceleration drops below threshold
########################
def FindEnd(a, time_data, threshold, dominant_axis):
    #xm = a[index_max, 0]
    #ym = a[index_max, 1]
    #zm = a[index_max, 2]
    threshold = 0.0
    i = 0
    
    dominant = a[index_max, dominant_axis]
    while (dominant > threshold and index_max + i < len(a)):
        #xm = a[index_max + i, 0]
        #ym = a[index_max + i, 1]
        #zm = a[index_max + i, 2]
        dominant = a[index_max + i, dominant_axis]
        i = i + 1
        
    xAxis = column(a[index_max: index_max + i],0)
    yAxis = column(a[index_max: index_max + i],1)
    zAxis = column(a[index_max: index_max + i],2)
    aAxis = column(a[index_max: index_max + i],3)
    
    td = time_data[index_max: index_max + i]
    
    plt.plot(td, xAxis, 'b', label="ax")
    plt.plot(td, yAxis, 'g', label="ay")
    plt.plot(td, zAxis, 'r', label="az")
    plt.plot(td, aAxis, 'k', label="aa")
    
    
    plt.title('Peak to Baseline')
    plt.xlabel('time')
    plt.ylabel(yaxis_label)
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    
    end = index_max + i
    return end

########################
#### Plot punch from initiation to end
########################
def PlotPunch(a, time_data, initiation, end):
    xAxis = column(a[initiation: end],0)
    yAxis = column(a[initiation: end],1)
    zAxis = column(a[initiation: end],2)
    aAxis = column(a[initiation: end],3)
    
    td = time_data[initiation: end]
    
    plt.plot(td, xAxis, 'b', label="ax")
    plt.plot(td, yAxis, 'g', label="ay")
    plt.plot(td, zAxis, 'r', label="az")
    plt.plot(td, aAxis, 'k', label="aa")
    
    plt.title('Punch')
    plt.xlabel('time')
    plt.ylabel(yaxis_label)
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()

def FindDominantAxis(a, time_data):
    maxes=[max(column(a,0)), max(column(a,1)), max(column(a,2))]
    
    dominant = maxes[0]
    idx = 0
    for i in range(0,3):
        if (maxes[i]>dominant):
            dominant = maxes[i]
            idx = i
    
    return idx

def CalcVelocity(a, time_data, initiation, end):
    """
    Calculate velocity by numerically integrating acceleration data
    
    Trapezoid rule is implemented for numerical integration

    Parameters
    ----------
    a : float[]
        acceleration data.
    time_data : float[]
        time stamps.
    initiation : int
        index of sample that was identified as punch initiation.
    end : int
        index of sample that was identified as punch end.

    Returns
    -------
    vrmsm : float
        Root Mean Square Velocity from punch start to punch end.

    """
    
    xAxis = column(a[initiation: end],0)
    yAxis = column(a[initiation: end],1)
    zAxis = column(a[initiation: end],2)
    td = time_data[initiation: end]
    
    vx = 0.0
    vy = 0.0
    vz = 0.0
    
    velx = [0]
    vely = [0]
    velz = [0]
    
    last_t = td[0]
    
    rms_vel = 0.0
    rmss = [0]
    
    velx.append(0)
    vely.append(0)
    velz.append(0)
    rmss.append(0)
    
    for i in range(1,len(xAxis)-1):
        dt = td[i] - last_t
        
        #apply trapezoid rule integration
        vx = vx + (xAxis[i] + xAxis[i-1]) * 0.5 * dt
        vy = vy + (yAxis[i] + yAxis[i-1]) * 0.5 * dt
        vz = vz + (zAxis[i] + zAxis[i-1]) * 0.5 * dt
        
        #calculate root mean square
        rms_vel = rms_vel + math.sqrt((xAxis[i]*xAxis[i] + yAxis[i]*yAxis[i] + zAxis[i]*zAxis[i]) /  3.0) * dt
        
        last_t = td[i]
        
        velx.append(vx)
        vely.append(vy)
        velz.append(vz)
        rmss.append(rms_vel)
        
    
    plt.plot(td, velx, 'b', label="vx")
    plt.plot(td, vely, 'g', label="vy")
    plt.plot(td, velz, 'r', label="vz")
    plt.plot(td, rmss, 'y', label="vrms")
    
    plt.title('Velocity')
    plt.xlabel('time')
    plt.ylabel('m/s')
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()   
    
    vxm = max(velx)
    print ("Max x velocity: " + str(vx))
    
    vym = max(vely)
    print ("Max y velocity: " + str(vym))
    
    vzm = max(velz)
    print ("Max z velocity: " + str(vzm))
    
    vrmsm = max(rmss)
    print ("Max rms velocity: " + str(vrmsm))
    
    return vrmsm

########################
### Plot remainder of acceleration curve after deleting previously analysed samples
########################
def PlotRemaining(a, time_data):
    plt.plot(time_data, a[:,0], 'r', label="ax")
    plt.plot(time_data, a[:,1], 'g', label="ay")
    plt.plot(time_data, a[:,2], 'b', label="az")
    plt.plot(time_data, a[:,3], 'k', label="absolute")
    
    plt.title('Acceleration')
    plt.xlabel('time')
    plt.ylabel(yaxis_label)
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()    

########################
###
########################
def FindLeftBoundary(a, time_data, threshold, start):
    value = 10.0
    i = 0
    while(value > threshold):
        i = i + 1
        value = abs(a[start - i, 0])
    return i

########################
### Plot dominant axis in black.
########################
def PlotDominant(a, axis, time_data):
    d = column(a, axis)
    plt.plot(time_data, d, 'k', label="Dominant Axis")

    plt.title('Acceleration along dominant axis')
    plt.xlabel('time')
    plt.ylabel(yaxis_label)
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()
    
def FindSubPeak(a, time_data, start, end, dominant_axis):
    acc = column(a, dominant_axis)
    acc = acc[start:end]
    subpeak_idx = np.argmax(acc)
    subpeak_idx += start
    return subpeak_idx
    
########################
### Find Baseline, i.e., first sample where absolute acceleration
### along all three axes falls below the threshold.
### This makes the RMS Velocity results more comparable to those of Kimm and Thiel
### increment specifies search direction (-1... towards start of punch; +1... towards end of punch)    
########################
def FindBaseline(a, time_data, start, threshold, increment):
    i = start
    xaxis = column(a, 0)
    zaxis = column(a, 1)
    yaxis = column(a, 2)
    
    while abs(xaxis[i]) > threshold and abs(yaxis[i]) > threshold and abs(zaxis[i]) > threshold:
        i = i + increment
    return i
    
#### Main Code ####
root = tk.Tk()
root.wm_attributes('-topmost', 1)
root.withdraw()

file_path = filedialog.askopenfilename(parent=root)


num_samples = file_len(file_path)

g = np.zeros((num_samples, 3))
a = np.zeros((num_samples, 4))
m = np.zeros((num_samples, 3))

time_data = [0]
#time_accum = float(y[0])
lsttimestamp = 0
i = 0

t1 = 0
time_accum = 0.0

start_time = -1

#yaxis_label = "g"
yaxis_label = "m/s2"
with open(file_path, newline='') as f:    
    #reader = csv.DictReader(f, delimiter=';')
    #for row in reader:
    #    print(row['Time (s)']) 
    
    reader = csv.reader(f, delimiter=';')
    headers = next(reader)
    npdata = np.array(list(reader)).astype(float)

    time_data = npdata[:,0]
    a = npdata[:,1:5]

# =============================================================================
# f = open(file_path, "r")
# for x in f:
#     if i != 0:
#         y = x.split(";")
#         
#         if len(y) == 5:
#             time_accum = float(y[0])
#             g[i-1] = [0,0,0]
#             a[i-1] = [y[1], y[2], y[3], y[4]]
#             m[i-1] = [0,0,0]
#             time_data.append(time_accum)
#             yaxis_label = "m/s2"
#             
#         elif len(y) < 10:
#             continue
#         
#         else :
#             if start_time == -1:
#                 start_time = float(y[1])
#             
#             #if i == 1:
#             #    t1 = datetime.strptime(y[1].replace(" ", ""), '%H:%M:%S.%f')
#             
#             #tn = datetime.strptime(y[1].replace(" ", ""), '%H:%M:%S.%f')
#             #td = tn - t1
# 
#             #t1 = tn
#             #tds = td.total_seconds()
#             
#             #if tds == 0.0:
#             #    tds = 0.0001
#             
#             #time_accum = time_accum + tds
#             
#             time_accum = float(y[1]) - start_time
#             
#             g[i-1] = [y[5], y[6], y[7]]
#             a[i-1] = [y[2], y[3], y[4]]
#             m[i-1] = [y[8], y[9], y[10]]
#             
#             #sec = datetime.strptime(y[1], '%H:%M:%S.%f').second
#             #ms = datetime.strptime(y[1], '%H:%M:%S.%f').microsecond / 10000
#             
#             #timestamp = datetime.strptime(y[1], '%H:%M:%S.%f').second
#             #timestamp = datetime.strptime(y[1], '%H:%M:%S.%f').microsecond / 10000
#             
#             #if (timestamp - lsttimestamp < 0.0):
#                 
#             time_data.append(time_accum)
#             #lsttimestamp = timestamp       
#     i = i+1
# f.close()
# =============================================================================

#### Configure analysis ####

print("Please specify velocity threshold (default = 0.0)")
threshold = float(input())

print("Please specify number of samples to analyse")
num_samples = int(input())

print("Please specify dominant axis (i.e., direction of punch; 0... x, 1... y, 2... z)")
dominant_axis = int(input())

PlotDominant(a, dominant_axis, time_data)


vrmss = []
for i in range (0,num_samples):
    #xAxis = column(a,0)
    #yAxis = column(a,1)
    #zAxis = column(a,2)
    PlotRemaining(a, time_data)
    
    #dominant_axis = 1#ssFindDominantAxis(a, time_data)
    index_max = FindPeak(a, time_data, 3)
    start = FindStart(a, time_data, threshold, 3)
    end = index_max
    
    
    
    #start = FindStart(a, time_data, threshold, 1)# dominant_axis)
    #start = FindBaseline(a, time_data, start, threshold, -1)
    #second_peak = FindSubPeak(a, time_data, start, index_max + 20, 0)
    
    #z = index_max
    #index_max=second_peak
    index_end = FindEnd(a, time_data, threshold, dominant_axis) 
    #end = FindEnd(a, time_data, threshold, 0)
    #end = FindBaseline(a, time_data, end, threshold, 1)
    #index_max = z
    
    PlotPunch(a, time_data, start, end)
    vrms = CalcVelocity(a, time_data, start, end)
    vrmss.append(vrms)
    
    #start = FindLeftBoundary(a, time_data, 4, start)
    
    #xAxis = column(a,0)
    #yAxis = column(a,1)
    #zAxis = column(a,2)
    
    for i in range(start, index_end):
        a[i, 0] = 0.0
        a[i, 1] = 0.0
        a[i, 2] = 0.0
        a[i, 3] = 0.0
    
    #xAxis = xAxis[:start] + xAxis[end:]
    #yAxis = yAxis[:start] + yAxis[end:]
    #zAxis = zAxis[:start] + zAxis[end:]
    #a = np.zeros((len(xAxis), 3))
    #for j in range(0,len(xAxis)):
    #    a[j] = [xAxis[j], yAxis[j], zAxis[j]]
    
    #a = [[xAxis], [yAxis], [zAxis]]

    #n = time_data[:start] + time_data[end:]
    #time_data = n
    
m = statistics.mean(vrmss)
d = statistics.stdev(vrmss)
v = statistics.variance(vrmss)

print("#################################")
print("sample mean: " + str(m) + " m/s")
print("sample deviation: " + str(d))
print("sample variance: " + str(v))
print("#################################")