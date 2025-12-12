#!/usr/bin/env python3

DEBUG = False
DISPLAY = False


########## 20240223 Prima data ##########
NpulsePerTrigger=1
# Define signal region
Pulse_startT     =  67 #200 #215
Pulse_endT       =  125 #215 #245
# Define rising and falling separation
Pulse_rise_endT     =  75 #200 #215
# Define pre-pulse (sideband) region
prePulse_startT  =  5 #100
prePulse_endT    =  60 #160
# Define post-pulse (sideband) region
postPulse_startT =  180 #230
postPulse_endT   =  199 #250

# Sideband pre-selection
cut_preRange = 0.07
cut_posRange = 0.12
cut_preStd = 0.025
cut_posStd = 0.03

totalTreeEvents = 10000
avg_buffer = 100
threshold = 0.015
