#!/usr/bin/env python3
DEBUG = False
DISPLAY = False


########## 20240223 Prima data ##########
NpulsePerTrigger=1
# Define signal region
Pulse_startT     =  0 #312 #67 #200 #215
Pulse_endT       =  24 #125 #215 #245
# Define rising and falling separation
Pulse_rise_endT     =  7 #75 #200 #215
Pulse_fall_endT     =  24 #75 #200 #215
# Define pre-pulse (sideband) region
prePulse_startT  =  0 #100
prePulse_endT    =  2 #160
# Define post-pulse (sideband) region
postPulse_startT =  24 #230
postPulse_endT   =  25 #250

# Sideband pre-selection
cut_preRange = 0.03
cut_posRange = 100
cut_preStd = 0.001
cut_posStd = 100

# Pulse selection
cut_pulseRange = 0.03

# FFT
freq_steps = 10000

totalTreeEvents = 10000
avgMaxCount = 3000
threshold = 0.1