#!/usr/bin/env python3

import math

def alt_expo(x,p):
    y = math.exp(p[0]+p[1]*x[0])
    if (y>1):
        return 1
    else:
        return y

def fitFunc_pulse_params(x, p):
    t = x[0]
    A = p[0]
    tau_s = p[1]
    tau = p[2]
    t0 = p[3]
    offset = p[4]

    norm = A/(tau-tau_s)
    rise_term = math.exp(-(t-t0)/tau_s)
    fall_term = math.exp(-(t-t0)/tau)
    pulse_shape = norm*(rise_term-fall_term)+offset
    if (t < 310):
        return 0
    else:
        return pulse_shape

def fitFunc_rise_plateau_params(x,p):
    t = x[0]
    A = p[0]
    tau = p[1]
    t0 = p[2]
    offset = p[3]
    total = A*(1-math.exp(-(t-t0)/tau))+offset
    return total

# fitFunc_rise.SetParameters(0.3, 0.3, 313, 0);
# fitFunc_rise.SetParNames("A", "tau", "t0", "offset");
def fitFunc_rise_params(x,p):
    t = x[0]
    A = p[0]
    tau = p[1]
    t0 = p[2]
    offset = p[3]
    total = A*(math.exp((t-t0)/tau))+offset
    return total

def fitFunc_rise_fall_params(x,p):
    t = x[0]
    Ar = p[0]
    tau_r = p[1]
    tr = p[2]
    offset_r = p[3]
    Af = p[4]
    tau_f = p[5]
    tf = p[6]
    offset_f = p[7]
    tswitch = p[8]

    rise = Ar*(math.exp((t-tr)/tau_r))+offset_r
    fall = Af*(math.exp(-(t-tf)/tau_f))+offset_f
    if (t < tswitch):
        return rise
    else:
        return fall

def Fit_time_constant_full_pulse(graph,fitmin,fitmax,options="SQR",goptions=""):
    fitFunc = ROOT.TF1("fitFunc",fitFunc_pulse_params,fitmin,fitmax,5)
    fitFunc.SetParameters(30, 2, 4, 313, 0);
    fitFunc.SetParNames("A", "tau_s", "tau", "t0", "offset");
    fitResult = graph.Fit(fitFunc,options,goptions); # "Q" option suppresses fit statistics output
    return fitFunc, fitResult

def Fit_time_constant_fall(graph,fitmin,fitmax,options="SQR",goptions="",color=ROOT.kRed):
    # Falling Edge
    fitFunc = ROOT.TF1("fitFunc", f"[0]*exp(-(x-[2])/[1])+[3]",fitmin, fitmax);
    fitFunc.SetLineColor(color)
    fitFunc.SetLineWidth(2)
    fitFunc.SetParameters(0.3, 4, 313, 0);
    fitFunc.SetParLimits(0,0,10)
    fitFunc.SetParLimits(1,0,10)
    fitFunc.SetParNames("A", "#tau", "t_{0}", "offset");
    fitResult = graph.Fit("fitFunc",options,goptions);
    return fitFunc, fitResult

def Fit_time_constant_rise(graph,fitmin,fitmax,options="SQR",goptions=""):
    # Rising Edge
    fitFunc = ROOT.TF1("fitFunc", fitFunc_rise_params, fitmin, fitmax, 4);
    fitFunc.SetLineWidth(2)
    fitFunc.SetParameters(0.3, 1, 308, 0);
    fitFunc.SetParNames("A", "#tau", "t_{0}", "offset");
    fitFunc.SetParLimits(1,0,10)
    fitFunc.SetParLimits(2,fitmin,fitmax)
    fitResult = graph.Fit("fitFunc",options,goptions);
    return fitFunc, fitResult

def Fit_time_constant_combined(graph,fitmin,fitmax,options="SQR",goptions=""):
    # Rise + Fall
    fitFunc = ROOT.TF1("fitFunc", fitFunc_rise_fall_params,fitmin,fitmax,9);
    fitFunc.SetParameters(0.3, 4, 308, 0, 0.3, 4, 313, 0, 313);
    fitFunc.SetParNames("Ar", "tau_r", "tr", "offset_r", "Af", "tau_f", "tf", "offset_f", "tswitch");
    fitResult = graph.Fit("fitFunc",options,goptions); # "Q" option suppresses fit statistics output
    return fitFunc, fitResult
