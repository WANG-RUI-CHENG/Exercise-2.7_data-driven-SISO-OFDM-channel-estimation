#!/usr/bin/python
from __future__ import division
import numpy as np
import math
import os
import time
import sys
import scipy.interpolate
import numpy.linalg as la

sqrt=np.sqrt
pi = math.pi

K = 64
CP = K//4
P = 64
allCarriers = np.arange(K)
pilotCarriers = allCarriers[::K//P]
dataCarriers = np.delete(allCarriers, pilotCarriers)
mu = 2
payloadBits_per_OFDM = K*mu
CR = 1

SNRdb = 25
Clipping_Flag = False
CP_flag = False
NoCP = True

_QPSK_mapping_table = {
    (0,1) : (-1+1j,), (1,1) : (1+1j,),
    (0,0) : (-1-1j,), (1,0) : (1-1j,)
}
_QPSK_demapping_table = {v : k for k, v in _QPSK_mapping_table.items()}
_QPSK_Constellation = np.array([[-1+1j], [1+1j], [-1-1j], [1-1j]])

_16QAM_mapping_table = {
    (0,0,1,0) : (-3+3j,), (0,1,1,0) : (-1+3j,), (1,1,1,0) : (1+3j,), (1,0,1,0) : (3+3j,),
    (0,0,1,1) : (-3+1j,), (0,1,1,1) : (-1+1j,), (1,1,1,1) : (1+1j,), (1,0,1,1) : (3+1j,),
    (0,0,0,1) : (-3-1j,), (0,1,0,1) : (-1-1j,), (1,1,0,1) : (1-1j,), (1,0,0,1) : (3-1j,),
    (0,0,0,0) : (-3-3j,), (0,1,0,0) : (-1-3j,), (1,1,0,0) : (1-3j,), (1,0,0,0) : (3-3j,)
}
_16QAM_demapping_table = {v : k for k, v in _16QAM_mapping_table.items()}
_16QAM_Constellation = np.array([[-3+3j], [-1+3j], [1+3j], [3+3j],
                                 [-3+1j], [-1+1j], [1+1j], [3+1j],
                                 [-3-1j], [-1-1j], [1-1j], [3-1j],
                                 [-3-3j], [-1-3j], [1-3j], [3-3j]])

_64QAM_mapping_table = {
    (0,0,0,1,0,0) : (-7+7j,), (0,0,1,1,0,0) : (-5+7j,), (0,1,1,1,0,0) : (-3+7j,), (0,1,0,1,0,0) : (-1+7j,), (1,1,0,1,0,0) : (1+7j,), (1,1,1,1,0,0) : (3+7j,), (1,0,1,1,0,0) : (5+7j,), (1,0,0,1,0,0) : (7+7j,),
    (0,0,0,1,0,1) : (-7+5j,), (0,0,1,1,0,1) : (-5+5j,), (0,1,1,1,0,1) : (-3+5j,), (0,1,0,1,0,1) : (-1+5j,), (1,1,0,1,0,1) : (1+5j,), (1,1,1,1,0,1) : (3+5j,), (1,0,1,1,0,1) : (5+5j,), (1,0,0,1,0,1) : (7+5j,),
    (0,0,0,1,1,1) : (-7+3j,), (0,0,1,1,1,1) : (-5+3j,), (0,1,1,1,1,1) : (-3+3j,), (0,1,0,1,1,1) : (-1+3j,), (1,1,0,1,1,1) : (1+3j,), (1,1,1,1,1,1) : (3+3j,), (1,0,1,1,1,1) : (5+3j,), (1,0,0,1,1,1) : (7+3j,),
    (0,0,0,1,1,0) : (-7+1j,), (0,0,1,1,1,0) : (-5+1j,), (0,1,1,1,1,0) : (-3+1j,), (0,1,0,1,1,0) : (-1+1j,), (1,1,0,1,1,0) : (1+1j,), (1,1,1,1,1,0) : (3+1j,), (1,0,1,1,1,0) : (5+1j,), (1,0,0,1,1,0) : (7+1j,),
    (0,0,0,0,1,0) : (-7-1j,), (0,0,1,0,1,0) : (-5-1j,), (0,1,1,0,1,0) : (-3-1j,), (0,1,0,0,1,0) : (-1-1j,), (1,1,0,0,1,0) : (1-1j,), (1,1,1,0,1,0) : (3-1j,), (1,0,1,0,1,0) : (5-1j,), (1,0,0,0,1,0) : (7-1j,),
    (0,0,0,0,1,1) : (-7-3j,), (0,0,1,0,1,1) : (-5-3j,), (0,1,1,0,1,1) : (-3-3j,), (0,1,0,0,1,1) : (-1-3j,), (1,1,0,0,1,1) : (1-3j,), (1,1,1,0,1,1) : (3-3j,), (1,0,1,0,1,1) : (5-3j,), (1,0,0,0,1,1) : (7-3j,),
    (0,0,0,0,0,1) : (-7-5j,), (0,0,1,0,0,1) : (-5-5j,), (0,1,1,0,0,1) : (-3-5j,), (0,1,0,0,0,1) : (-1-5j,), (1,1,0,0,0,1) : (1-5j,), (1,1,1,0,0,1) : (3-5j,), (1,0,1,0,0,1) : (5-5j,), (1,0,0,0,0,1) : (7-5j,),
    (0,0,0,0,0,0) : (-7-7j,), (0,0,1,0,0,0) : (-5-7j,), (0,1,1,0,0,0) : (-3-7j,), (0,1,0,0,0,0) : (-1-7j,), (1,1,0,0,0,0) : (1-7j,), (1,1,1,0,0,0) : (3-7j,), (1,0,1,0,0,0) : (5-7j,), (1,0,0,0,0,0) : (7-7j,)
}
_64QAM_demapping_table = {v : k for k, v in _64QAM_mapping_table.items()}
_64QAM_Constellation = np.array([[-7+7j], [-5+7j], [-3+7j], [-1+7j], [1+7j], [3+7j], [5+7j], [7+7j],
                                [-7+5j], [-5+5j], [-3+5j], [-1+5j], [1+5j], [3+5j], [5+5j], [7+5j],
                                [-7+3j], [-5+3j], [-3+3j], [-1+3j], [1+3j], [3+3j], [5+3j], [7+3j],
                                [-7+1j], [-5+1j], [-3+1j], [-1+1j], [1+1j], [3+1j], [5+1j], [7+1j],
                                [-7-1j], [-5-1j], [-3-1j], [-1-1j], [1-1j], [3-1j], [5-1j], [7-1j],
                                [-7-3j], [-5-3j], [-3-3j], [-1-3j], [1-3j], [3-3j], [5-3j], [7-3j],
                                [-7-5j], [-5-5j], [-3-5j], [-1-5j], [1-5j], [3-5j], [5-5j], [7-5j],
                                [-7-7j], [-5-7j], [-3-7j], [-1-7j], [1-7j], [3-7j], [5-7j], [7-7j]
                                  ])

def Clipping (x, CL):
    sigma = np.sqrt(np.mean(np.square(np.abs(x))))
    CL = CL*sigma
    x_clipped = x
    clipped_idx = abs(x_clipped) > CL
    x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx]*CL),abs(x_clipped[clipped_idx]))
    return x_clipped

def PAPR (x):
    Power = np.abs(x)**2
    PeakP = np.max(Power)
    AvgP = np.mean(Power)
    PAPR_dB = 10*np.log10(PeakP/AvgP)
    return PAPR_dB

def Modulation(bits):
    bit_r = bits.reshape((int(len(bits)/2), 2))
    return (2*bit_r[:,0]-1)+1j*(2*bit_r[:,1]-1)

def Modulation_16(bits):
    bit_r = bits.reshape((int(len(bits)/4), 4))
    bit_mod = []
    for i in range(int(len(bits)/4)):
        bit_mod.append(list(_16QAM_mapping_table.get(tuple(bit_r[i]))))
    return np.asarray(bit_mod).reshape((-1,))

def Modulation_64(bits):
    bit_r = bits.reshape((int(len(bits)/6), 6))
    bit_mod = []
    for i in range(int(len(bits)/6)):
        bit_mod.append(list(_64QAM_mapping_table.get(tuple(bit_r[i]))))
    return np.asarray(bit_mod).reshape((-1,))

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def addCP(OFDM_time, CP, CP_flag, mu, K):
    if CP_flag == False:
        bits_noise = np.random.binomial(n=1, p=0.5, size=(K*mu, ))
        if mu==2:
            codeword_noise = Modulation(bits_noise)
        elif mu==4:
            codeword_noise = Modulation_16(bits_noise)
        else:
            codeword_noise = Modulation_64(bits_noise)
        OFDM_data_nosie = codeword_noise
        OFDM_time_noise = np.fft.ifft(OFDM_data_nosie)
        cp = OFDM_time_noise[-CP:]
    else:
        cp = OFDM_time[-CP:]
    return np.hstack([cp, OFDM_time])

def channel(signal,channelResponse,SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise,sigma2

def removeCP(signal, CP, K):
    return signal[CP:(CP+K)]

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)

def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest

def Demodulation(bits_mod):
    X_pred = np.array([])
    for i in range(len(bits_mod)):
        tmp = bits_mod[i] * np.ones((4,1))
        min_distance_index = np.argmin(abs(tmp - _QPSK_Constellation))
        X_pred = np.concatenate((X_pred,np.array(_QPSK_demapping_table[tuple(_QPSK_Constellation[min_distance_index])])))
    return X_pred

def Demodulation_16(bits_mod):
    X_pred = np.array([])
    for i in range(len(bits_mod)):
        tmp = bits_mod[i] * np.ones((16,1))
        min_distance_index = np.argmin(abs(tmp - _16QAM_Constellation))
        X_pred = np.concatenate((X_pred,np.array(_16QAM_demapping_table[tuple(_16QAM_Constellation[min_distance_index])])))
    return X_pred

def Demodulation_64(bits_mod):
    X_pred = np.array([])
    for i in range(len(bits_mod)):
        tmp = bits_mod[i] * np.ones((64,1))
        min_distance_index = np.argmin(abs(tmp - _64QAM_Constellation))
        X_pred = np.concatenate((X_pred,np.array(_64QAM_demapping_table[tuple(_64QAM_Constellation[min_distance_index])])))
    return X_pred

def get_payload(equalized):
    return equalized[dataCarriers]

def PS(bits):
    return bits.reshape((-1,))

def ofdm_simulate(codeword, channelResponse,SNRdb, mu, CP_flag, K, P, CP,     pilotValue,pilotCarriers, dataCarriers,Clipping_Flag,ce_flag=False):
    payloadBits_per_OFDM = mu*len(dataCarriers)
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
        if mu == 2:
            QAM = Modulation(bits)
        elif mu == 4:
            QAM = Modulation_16(bits)
        else:
            QAM = Modulation_64(bits)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue

    OFDM_time = (FH @ OFDM_data.reshape((K,1))).reshape(-1,)
    OFDM_withCP = addCP(OFDM_time, CP, CP_flag, mu, K)
    OFDM_TX = OFDM_withCP
    if Clipping_Flag:
        OFDM_TX = Clipping(OFDM_TX,CR)
    OFDM_RX,_ = channel(OFDM_TX, channelResponse,SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX, CP,K)

    if ce_flag:
        return np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP)))

    if mu == 2:
        codeword_qam = Modulation(codeword)
    elif mu == 4:
        codeword_qam = Modulation_16(codeword)
    else:
        codeword_qam = Modulation_64(codeword)
    symbol = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = (FH @ OFDM_data_codeword.reshape((K,1))).reshape(-1,)
    OFDM_withCP_codeword = addCP(OFDM_time_codeword, CP, CP_flag, mu, K)
    if Clipping_Flag:
        OFDM_withCP_codeword = Clipping(OFDM_withCP_codeword,CR)
    OFDM_RX_codeword,sigma2 = channel(OFDM_withCP_codeword, channelResponse,SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword,CP,K)
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))),
                           np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), sigma2

ISI = np.zeros(K,dtype=complex)
estimated_ISI = np.zeros((K,1),dtype=complex)
def ofdm_simulate_cp_free(codeword, H, A, FH, SNR, mu, K, P,     pilotValue,pilotCarriers, dataCarriers, CE_flag=False):
    global ISI
    payloadBits_per_OFDM = mu*len(dataCarriers)
    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
        if mu == 2:
            QAM = Modulation(bits)
        elif mu == 4:
            QAM = Modulation_16(bits)
        else:
            QAM = Modulation_64(bits)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue
    yp = (H-A) @ FH @ OFDM_data
    signal_power = np.mean(abs(yp**2))
    sigma2 = signal_power * 10**(-SNR/10)
    noise = np.sqrt(sigma2/2) * (np.random.randn(*yp.shape)+1j*np.random.randn(*yp.shape))
    yp = yp + noise
    yp = yp + ISI
    ISI = A @ FH @ OFDM_data

    if CE_flag:
        return np.concatenate((np.real(yp),np.imag(yp)))

    if mu == 2:
        codeword_qam = Modulation(codeword)
    elif mu == 4:
        codeword_qam = Modulation_16(codeword)
    else:
        codeword_qam = Modulation_64(codeword)
    symbol = codeword_qam
    ys = (H-A) @ FH @ symbol
    signal_power = np.mean(abs(ys**2))
    sigma2 = signal_power * 10**(-SNR/10)
    noise = np.sqrt(sigma2/2) * (np.random.randn(*ys.shape)+1j*np.random.randn(*ys.shape))
    ys = ys + noise
    ys = ys + ISI
    ISI = A @ FH @ symbol

    return np.concatenate((np.concatenate((np.real(yp),np.imag(yp))), np.concatenate((np.real(ys),np.imag(ys))))), sigma2, codeword_qam

def LS_CE(Y,pilotValue,pilotCarriers,K,P,int_opt):
    index = np.arange(P)
    LS_est = np.zeros(P, dtype=complex)
    LS_est[index] = Y[pilotCarriers] / pilotValue[index]
    if int_opt == 0:
        H_LS = interpolate(LS_est,pilotCarriers,K,0)
    if int_opt == 1:
        H_LS = interpolate(LS_est,pilotCarriers,K,1)
    return H_LS

def MMSE_CE(Y,pilotValue,pilotCarriers,K,P,h,SNR):
    snr = 10 ** (SNR * 0.1)
    index = np.arange(P)
    H_tilde = np.zeros(P, dtype=complex)
    H_tilde[index] = Y[pilotCarriers] / pilotValue[index]

    index = np.arange(len(h))
    hh = h.dot(np.conj(h).T)
    tmp = h * np.conj(h) * index
    r = np.sum(tmp) / hh
    r2 = tmp.dot(index.T) / hh
    tau_rms = (r2 - r ** 2) ** 0.5

    df = 1 / K
    j2pi_tau_df = 1j * 2 * math.pi * tau_rms * df

    K1 = np.reshape(np.repeat(np.arange(K).T, P), (K, P))
    K2 = np.arange(P)
    for _ in range(K - 1):
        K2 = np.concatenate((K2, np.arange(P)))
    K2 = np.reshape(K2, (K, P))
    rf = np.ones((K, P), dtype=complex) / (1 + j2pi_tau_df * (K1 - K2 * (K // P)))

    K3 = np.reshape(np.repeat(np.arange(P).T, P), (P, P))
    K4 = np.arange(P)
    for _ in range(P - 1):
        K4 = np.concatenate((K4, np.arange(P)))
    K4 = np.reshape(K4, (P, P))
    rf2 = np.ones((P, P), dtype=complex) / (1 + j2pi_tau_df * (K // P) * (K3 - K4))

    Rhp = rf
    Rpp = rf2 + np.eye(len(H_tilde)) / snr
    W_MMSE = Rhp.dot(np.linalg.inv(Rpp))
    H_MMSE = (W_MMSE.dot(H_tilde.T)).T
    return H_MMSE, W_MMSE

interpolate_method = 1
def interpolate(H_est,pilotCarriers,K,method):
    if pilotCarriers[0] > 0 :
        slope = (H_est[1]-H_est[0])/(K//P)
        H_est = np.insert(H_est,0, H_est[0]-slope*(K//P))
        pilotCarriers = np.insert(pilotCarriers,0,0)
    if pilotCarriers[len(pilotCarriers) - 1] < (K-1):
        slope = (H_est[len(H_est)-1]-H_est[len(H_est)-2])/(K//P)
        H_est = np.append(H_est,H_est[len(H_est)-1]+slope*(K//P))
        pilotCarriers = np.append(pilotCarriers,(K-1))
    if method == 0:
        H_interpolated = scipy.interpolate.interp1d(pilotCarriers, H_est,'linear')
    if method == 1:
        H_interpolated = scipy.interpolate.interp1d(pilotCarriers, H_est,'cubic')
    index = np.arange(K)
    H_interpolated_new = H_interpolated(index)
    return H_interpolated_new

def Normalized_FFT_Matrix(K):
    F = np.zeros((K,K),dtype=complex)
    for i in range(K):
        for j in range(K):
             F[i][j]=1/np.sqrt(K)*np.exp(-1j*2*pi*i*j/K)
    return F

F = Normalized_FFT_Matrix(K)
FH = np.conj(F).T


def OAMP(K,yd,H,sigma2,channel_type=0,mu=2,Mr=4,Nt=4,T=4):
    v_sqr_last = 0.
    x_hat = np.zeros((2*K,1))
    for t in range(T):
        v_sqr = (np.square(np.linalg.norm(yd-H.dot(x_hat),2,axis=0)) - K*sigma2) / np.trace(H.T.dot(H))
        v_sqr = 0.5*v_sqr + 0.5 *v_sqr_last
        v_sqr = np.maximum(v_sqr,1e-5)
        v_sqr_last = v_sqr
        w_hat = v_sqr * H.T.dot( np.linalg.inv(v_sqr*H.dot(H.T)+sigma2/2*np.eye(2*K)) )
        w = 2*K/np.trace(w_hat.dot(H))*w_hat
        r = x_hat + w.dot(yd-H.dot(x_hat))
        B = np.eye(2*K) - w.dot(H)
        tau_sqr = 1/(2*K)*np.trace(B.dot(B.T))*v_sqr + 1/(4*K)*np.trace(w.dot(w.T))*sigma2
        tau_sqr = np.maximum(tau_sqr,1e-5)
        if mu == 2:
            P0 = np.exp(-(-1-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P1 = np.exp(-(1-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            x_hat = (P1-P0) / (P1+P0)
        elif mu == 4:
            clipped_idx = abs(r) > 4.
            r[clipped_idx] = np.divide((r[clipped_idx]*4.),abs(r[clipped_idx]))
            P_3 = np.exp(-(-3-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P_1 = np.exp(-(-1-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P1 = np.exp(-(1-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P3 = np.exp(-(3-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            x_hat = (-3*P_3-P_1+P1+3*P3) / (P_3+P_1+P1+P3)
        else:
            clipped_idx = abs(r) > 8.
            r[clipped_idx] = np.divide((r[clipped_idx]*8.),abs(r[clipped_idx]))
            P_7 = np.exp(-(-7-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P_5 = np.exp(-(-5-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P_3 = np.exp(-(-3-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P_1 = np.exp(-(-1-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P1 = np.exp(-(1-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P3 = np.exp(-(3-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P5 = np.exp(-(5-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            P7 = np.exp(-(7-r)**2/(2*tau_sqr)) / (2*np.pi*tau_sqr)**0.5
            x_hat = (-7*P_7-5*P_5-3*P_3-P_1+P1+3*P3+5*P5+7*P7) / (P_7+P_5+P_3+P_1+P1+P3+P5+P7)
    x_hat = x_hat.reshape((2,K))
    x_hat = x_hat[0,:]+1j*x_hat[1,:]
    if mu == 2:
        x_hat_demod = Demodulation(x_hat)
    elif mu == 4:
        x_hat_demod = Demodulation_16(x_hat)
    else:
        x_hat_demod = Demodulation_64(x_hat)
    return x_hat_demod,x_hat

channel_train = np.load('tools/channel_train.npy')
train_size = channel_train.shape[0]
channel_test = np.load('tools/channel_test.npy')
test_size = channel_test.shape[0]

def get_cyclic_and_cutoff_matrix(h):
    H = np.zeros((K,K),dtype=complex)
    A = np.zeros((K,K),dtype=complex)
    h_ = np.flip(np.append(h,np.zeros((K-CP,1))))
    for i in range(K):
        H[i] = np.roll(h_,i+1)
        if i < (CP-1):
            A[i] = np.hstack([np.zeros(K-CP+i+1),h_[K-CP:K-i-1]])
    return H,A

def get_WMMSE(SNR, CP_flag=True):
    index = np.random.choice(np.arange(test_size), size=1)
    h = channel_test[index].reshape((-1,))
    H,A = get_cyclic_and_cutoff_matrix(h)
    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
    if NoCP:
        signal_output = ofdm_simulate_cp_free(bits, H, A, FH, SNR, mu, K, P, pilotValue, pilotCarriers, dataCarriers, CE_flag=True)
    else:
        signal_output,_ = ofdm_simulate(bits, h, SNR, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers, Clipping_Flag)
    yp_complex = signal_output[0:K] + 1j * signal_output[K:2*K]
    Yp_complex = F @ yp_complex
    _,W_MMSE = MMSE_CE(Yp_complex,pilotValue,pilotCarriers,K,P,h,SNR)
    W_MMSE = np.concatenate(( np.concatenate((np.real(W_MMSE),-np.imag(W_MMSE)),axis=1),np.concatenate((np.imag(W_MMSE),np.real(W_MMSE)),axis=1) ))
    return W_MMSE

def sample_gen(bs, SNR = 20, training_flag=True, NoCP=False, CP_flag=True):
    if training_flag:
        index = np.random.choice(np.arange(train_size), size=bs)
        h_total = channel_train[index]
    else:
        index = np.random.choice(np.arange(test_size), size=bs)
        h_total = channel_test[index]
    H_samples = []
    H_labels = []
    Yp, Xp = [], []
    for h in h_total:
        H_true = np.fft.fft(h,n=K)
        H,A = get_cyclic_and_cutoff_matrix(h)
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
        if NoCP:
            signal_output = ofdm_simulate_cp_free(bits, H, A, FH, SNR, mu, K, P, pilotValue, pilotCarriers, dataCarriers, CE_flag=True)
        else:
            signal_output = ofdm_simulate(bits, h, SNR, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers,
                                          Clipping_Flag, ce_flag=True)
        yp_complex = signal_output[0:K] + 1j * signal_output[K:2*K]
        Yp_complex = F @ yp_complex
        H_LS = LS_CE(Yp_complex,pilotValue,pilotCarriers,K,P,interpolate_method)
        H_true = np.concatenate((np.real(H_true),np.imag(H_true)))
        H_LS = np.concatenate((np.real(H_LS),np.imag(H_LS)))

        H_labels.append(H_true)
        H_samples.append(H_LS)
        Yp.append(np.concatenate((np.real(Yp_complex), np.imag(Yp_complex))))
    Xp = np.tile(np.concatenate((np.real(pilotValue), np.imag(pilotValue))), (bs, 1))
    return np.asarray(H_samples), np.asarray(H_labels), np.asarray(Yp), np.asarray(Xp)

def test_ce(sess, input_holder, output, SNR, est_type, NoCP=False, CP_flag=True):
    num_trail = 1000
    L = 16
    downsampler = allCarriers[::K // L]
    MSE_T, MSE_F = 0., 0.
    for i in range(num_trail):
        index = np.random.choice(np.arange(test_size), size=1)
        h = channel_test[index].reshape((-1,))
        Htrue = np.fft.fft(h, n=K)
        H, A = get_cyclic_and_cutoff_matrix(h)
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
        if NoCP:
            signal_output, sigma2, _ = ofdm_simulate_cp_free(bits, H, A, FH, SNR, mu, K, P, pilotValue, pilotCarriers,
                                                             dataCarriers)
        else:
            signal_output, sigma2 = ofdm_simulate(bits, h, SNR, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers,
                                                  dataCarriers, Clipping_Flag)
        yp_complex = signal_output[0:K] + 1j * signal_output[K:2 * K]
        Yp_complex = F @ yp_complex

        if est_type == 'ls':
            estimated_H = LS_CE(Yp_complex, pilotValue, pilotCarriers, K, P, interpolate_method)
        elif est_type == 'mmse':
            estimated_H, _ = MMSE_CE(Yp_complex, pilotValue, pilotCarriers, K, P, h, SNR)
        else:
            input1 = np.concatenate((np.real(Yp_complex), np.imag(Yp_complex))).reshape(1, 2 * K)
            input2 = np.concatenate((np.real(pilotValue), np.imag(pilotValue))).reshape(1, 2 * K)
            input = np.concatenate((input1, input2), axis=1)
            estimated_H = sess.run(output, feed_dict={input_holder: input}).reshape(-1, )
            estimated_H = estimated_H[:K] + 1j * estimated_H[K:2 * K]

        estimated_h = np.fft.ifft(estimated_H[downsampler])
        MSE_F += np.sum(abs(estimated_H - Htrue) ** 2) / np.sum(abs(Htrue) ** 2)
        MSE_T += np.sum(abs(estimated_h - h) ** 2) / np.sum(abs(h) ** 2)

        sys.stdout.write('\rMSE_T={mse_t:.6f} MSE_F={mse_f:.6f}'.format(mse_t=10 * np.log10(MSE_T / (i + 1)),
                                                                        mse_f=10 * np.log10(MSE_F / (i + 1))))
        sys.stdout.flush()

    return MSE_T / num_trail, MSE_F / num_trail


def sample_gen_for_OAMP(bs, SNR, sess, input_holder, output, training_flag=True):
    if training_flag:
        index = np.random.choice(np.arange(train_size), size=bs)
        h_total = channel_train[index]
    else:
        index = np.random.choice(np.arange(test_size), size=bs)
        h_total = channel_test[index]

    H_ = np.zeros((2*bs*K,2*K),dtype=np.float32)
    x_ = np.zeros((2*bs*K,1),dtype=np.float32)
    y_ = np.zeros((2*bs*K,1),dtype=np.float32)
    sigma2_ = np.zeros((bs,1),dtype=np.float32)

    count = 0
    for h in h_total:
        H,A = get_cyclic_and_cutoff_matrix(h)
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
        if NoCP:
            signal_output,sigma2,bits_mod = ofdm_simulate_cp_free(bits, H, A, FH, SNR, mu, K, P, pilotValue, pilotCarriers, dataCarriers)
        else:
            signal_output,_ = ofdm_simulate(bits, h, SNR, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers, Clipping_Flag)
            bits_mod = Modulation(bits) if mu == 2 else (Modulation_16(bits) if mu == 4 else Modulation_64(bits))
            sigma2 = np.array([[0]], dtype=np.float32)
        yp_complex = signal_output[0:K] + 1j * signal_output[K:2*K]
        Yp_complex = F @ yp_complex

        H_LS = LS_CE(Yp_complex,pilotValue,pilotCarriers,K,P,interpolate_method)
        H_LS = np.concatenate((np.real(H_LS),np.imag(H_LS))).reshape(1,2*K)

        H_out = sess.run(output,feed_dict={input_holder:H_LS.astype(np.float32)}).reshape(-1,)
        H_est = H_out[0:K] + 1j * H_out[K:2*K]
        downsampler = allCarriers[::K//CP]
        H_est = H_est[downsampler]
        h_est = IDFT(H_est)
        H_hat,A_hat = get_cyclic_and_cutoff_matrix(h_est)

        if NoCP:
            yd_complex = (signal_output[2*K:3*K] + 1j * signal_output[3*K:4*K]) - A_hat @ FH @ pilotValue
            yd = np.concatenate((np.real(yd_complex.reshape((K,1))),np.imag(yd_complex.reshape((K,1)))))
            H_bar = (H_hat-A_hat) @ FH
        else:
            yd = signal_output[2*K:4*K].reshape(2*K,1)
            H_bar = H_hat @ FH

        H_bar = np.concatenate(( np.concatenate((np.real(H_bar),-np.imag(H_bar)),axis=1),
            np.concatenate((np.imag(H_bar),np.real(H_bar)),axis=1) ))
        x = np.concatenate((np.real(bits_mod.reshape((K,1))),np.imag(bits_mod.reshape((K,1)))))
        H_[2*K*count:2*K*(count+1)] = H_bar.astype(np.float32)
        x_[2*K*count:2*K*(count+1)] = x.astype(np.float32)
        y_[2*K*count:2*K*(count+1)] = yd.astype(np.float32)
        sigma2_[count] = np.array(sigma2).astype(np.float32)
        count += 1
    H_ = H_.reshape(bs,2*K,2*K)
    x_ = x_.reshape(bs,2*K,1)
    y_ = y_.reshape(bs,2*K,1)
    sigma2_ = sigma2_.reshape(bs,1,1)

    return y_,x_,H_,sigma2_

def test_DL_OAMP(sess,prob,x_hat_T,input_holder,output,SNR,OAMPnet=False):
    err_bits_target = 1000
    total_err_bits = 0
    total_bits = 0
    start = time.time()
    while True:
        index = np.random.choice(np.arange(test_size), size=1)
        h = channel_test[index].reshape((-1,))
        H, A = get_cyclic_and_cutoff_matrix(h)
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
        if NoCP:
            signal_output, sigma2,_ = ofdm_simulate_cp_free(bits, H, A, FH, SNR, mu, K, P, pilotValue, pilotCarriers, dataCarriers)
        else:
            signal_output,sigma2 = ofdm_simulate(bits, h, SNR, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers, Clipping_Flag)
        yp_complex = signal_output[0:K] + 1j * signal_output[K:2*K]
        Yp_complex = F @ yp_complex

        H_LS = LS_CE(Yp_complex,pilotValue,pilotCarriers,K,P,interpolate_method)
        H_LS = np.concatenate((np.real(H_LS),np.imag(H_LS))).reshape(1,2*K)
        H_out = sess.run(output,feed_dict={input_holder:H_LS.astype(np.float32)}).reshape(-1,)

        H_est = H_out[0:K] + 1j * H_out[K:2*K]
        downsampler = allCarriers[::K//CP]
        H_est = H_est[downsampler]
        h_est = IDFT(H_est)
        H_hat,A_hat = get_cyclic_and_cutoff_matrix(h_est)

        if NoCP:
            yd_complex = (signal_output[2*K:3*K] + 1j * signal_output[3*K:4*K]) - A_hat @ FH @ pilotValue
            yd = np.concatenate((np.real(yd_complex.reshape((K,1))),np.imag(yd_complex.reshape((K,1)))))
            H_bar = (H_hat-A_hat) @ FH
        else:
            yd = signal_output[2*K:4*K].reshape(2*K,1)
            H_bar = H_hat @ FH
        H_bar = np.concatenate(( np.concatenate((np.real(H_bar),-np.imag(H_bar)),axis=1),np.concatenate((np.imag(H_bar),np.real(H_bar)),axis=1) ))
        if OAMPnet == False:
            x_hat_demod,x_hat = OAMP(K,yd,H_bar,sigma2,mu=mu)
        else:
            yd = yd.reshape(1,2*K,1).astype(np.float32)
            H_bar = H_bar.reshape(1,2*K,2*K).astype(np.float32)
            sigma2 = np.array(sigma2).reshape(1,1,1).astype(np.float32)
            x_hat = sess.run(x_hat_T,feed_dict={prob.y_:yd,prob.x_:np.zeros((1,2*K,1),dtype=np.float32),
                prob.H_:H_bar,prob.sigma2_:sigma2,prob.sample_size_:1})
            x_hat = x_hat.reshape(2,K)
            x_hat = x_hat[0,:]+1j*x_hat[1,:]
            if mu == 2:
                x_hat_demod = Demodulation(x_hat)
            elif mu == 4:
                x_hat_demod = Demodulation_16(x_hat)
            else:
                x_hat_demod = Demodulation_64(x_hat)

        err_bits = np.sum(np.not_equal(x_hat_demod,bits))
        total_err_bits  += err_bits
        total_bits += mu*K
        if err_bits > 0:
            sys.stdout.write('\rtotal_err_bits={teb} total_bits={tb} BER={BER:.9f}'.format(teb=total_err_bits,tb=total_bits,BER=total_err_bits/total_bits))
            sys.stdout.flush()
        if total_err_bits > err_bits_target:
            end = time.time()
            print("\nSNR=",SNR,"iter_time:",end-start)
            ber = total_err_bits/total_bits
            print("BER:", ber)
            break
    return ber

Pilot_file_name = 'Pilot_' + str(P)+'_mu'+str(mu)+'.txt'
if os.path.isfile(Pilot_file_name):
    print('Load Training Pilots txt')
    bits = np.loadtxt(Pilot_file_name, delimiter=',')
else:
    bits = np.random.binomial(n=1, p=0.5, size=(P * mu, ))
    np.savetxt(Pilot_file_name, bits, delimiter=',')

if mu == 2:
    pilotValue = Modulation(bits)
elif mu == 4:
    pilotValue = Modulation_16(bits)
else:
    pilotValue = Modulation_64(bits)
