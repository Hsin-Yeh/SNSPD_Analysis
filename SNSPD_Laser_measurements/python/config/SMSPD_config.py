#!/usr/bin/env python3
DEBUG = False
DISPLAY = False

########## 20240223 Prima data ##########
NpulsePerTrigger=1
# Define signal region
Pulse_startT     =  308 #312 #67 #200 #215
Pulse_endT       =  380 #125 #215 #245
# Define rising and falling separation
Pulse_rise_endT     =  313 #75 #200 #215
Pulse_fall_endT     =  342 #75 #200 #215
prePulse_startT  =  1 #100
prePulse_endT    =  20 #160
# Define post-pulse (sideband) region
postPulse_startT =  230 #230
postPulse_endT   =  250 #250

# Sideband pre-selection
cut_preRange = 0.11
cut_posRange = 10
cut_preStd = 0.021
cut_posStd = 10

# Pulse selection
cut_pulseRange = 0.05

# FFT
freq_steps = 10000

totalTreeEvents = 10000
avgMaxCount = 3000
threshold = 0.1
