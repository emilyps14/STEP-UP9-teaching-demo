
import nitime.algorithms as tsa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian, hamming, hann
from scipy.signal import chirp
from es_utils import spec_utils

plt.style.use('seaborn-deep')

blnsave = False

#%% Windowing functions

plt.close('all')

Fs = 1000

N = int(4 * Fs)
ta = np.linspace(-N/2/Fs,N/2/Fs,N)

rect_win = np.zeros(N)
rect_win[int(N/4):int(3*N/4)] = 1

h1 = np.zeros(N)
h1[int(N/4):int(3*N/4)] = hamming(int(N/2),sym=False)

h2 = np.zeros(N)
h2[int(N/4):int(3*N/4)] = hann(int(N/2),sym=False)

g  = np.zeros(N)
g[int(N/4):int(3*N/4)] = gaussian(int(N/2), N/12)

pad = 5
falims = [-20,20]
sigs = [rect_win,h2,g]
names = ['Rectangular','Hanning','Gaussian']

fig,axs = plt.subplots(nrows=1,ncols=2,figsize=[7,2.5])
for sig in sigs:
    sig_f = np.fft.fft(sig, n=N*pad)
    fa = np.fft.fftfreq(N*pad,1/Fs)
    sig_S = sig_f*sig_f.conj()

    order = np.argsort(fa)
    fa = fa[order]
    sig_S = sig_S[order]
    axs[0].plot(ta,sig.T)
    # axs[0].set_xlim(talims)
    axs[0].set_title('Time Domain')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_xlabel('Time (s)')

    axs[1].plot(fa,10*np.log10(np.real_if_close(sig_S)))
    axs[1].set_title('Frequency Domain')
    axs[1].set_ylabel('Power (dB)')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_xlim(falims)
    axs[1].set_ylim([-50,70])

fig.subplots_adjust(wspace=0.3,bottom=0.2,left=0.1,right=0.95)
axs[0].legend(names)

if blnsave:
    fig.savefig('windowing_functions.png')


#%% Slepian tapers

plt.close('all')
Fs = 1000
params = (4000/Fs,Fs/500,15)
talims = np.array([-2500,2500])/Fs
falims = np.array([-2/500,2/500])*Fs


N = int(params[0] * Fs)
NW = params[0]*params[1]
K = params[2]
dpss, eigvals = tsa.dpss_windows(N, NW, K)
ta = np.linspace(-N/2/Fs,N/2/Fs,N)

dpss_f = np.fft.fft(dpss, n=N*pad)
fa = np.fft.fftfreq(N*pad,1/Fs)
dpss_S = dpss_f*dpss_f.conj()

order = np.argsort(fa)
fa = fa[order]
dpss_S = dpss_S[:,order]


for nplot in [1,5]:
    fig,axs = plt.subplots(nrows=1,ncols=2,figsize=[7,2.5])
    axs[0].plot(ta,dpss[:nplot,:].T)
    axs[0].set_xlim(talims)
    # axs[0].legend(list(np.arange(K)+1))
    axs[0].set_title('Time Domain')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylim([-0.03,0.04])

    axs[1].plot(fa,10*np.log10(np.real_if_close(dpss_S[:nplot,:].T)))
    axs[1].set_title('Frequency Domain')
    axs[1].set_ylabel('Power (dB)')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_xlim(falims)
    # axs[1].set_ylim([-60,40])

    fig.subplots_adjust(wspace=0.35,bottom=0.2,left=0.12,right=0.95)

    if blnsave:
        fig.savefig(f'slepian_tapers_{nplot}.png')

#%% Activity 1: Match TWK with time domain and frequency domain slepian tapers
plt.close('all')

Fs = 1000

param_sets = [(.2,10,3),(4,0.5,3),(1,5,8)]
talims = [-2.5,2.5]
falims = [-20,20]
pad = 5

for i,params in enumerate(param_sets):
    N = int(params[0] * Fs)
    NW = params[0]*params[1]
    K = params[2]
    dpss, eigvals = tsa.dpss_windows(N, NW, K)
    ta = np.linspace(-N/2/Fs,N/2/Fs,N)

    dpss_f = np.fft.fft(dpss, n=N*pad)
    fa = np.fft.fftfreq(N*pad,1/Fs)
    dpss_S = dpss_f*dpss_f.conj()

    order = np.argsort(fa)
    fa = fa[order]
    dpss_S = dpss_S[:,order]


    fig,axs = plt.subplots(nrows=2,ncols=1,figsize=[4.5,6])
    axs[0].plot(ta,dpss.T)
    axs[0].set_xlim(talims)
    axs[0].legend(list(np.arange(K)+1))
    axs[0].set_title('Time Domain Tapers')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_xlabel('Time (s)')

    axs[1].plot(fa,10*np.log10(np.real_if_close(dpss_S.mean(0).T)),color='k')
    axs[1].set_title('Mean Power Spectrum Across Tapers')
    axs[1].set_ylabel('Power (dB)')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_xlim(falims)
    axs[1].set_ylim([-40,40])

    fig.subplots_adjust(hspace=0.5,left=0.2)

    if blnsave:
        fig.savefig(f'quiz_{i}.png')

#%% Activity 2: Match spectrogram to timeseries and TWK
plt.close('all')

T = 15
ta = np.linspace(0,T,T*Fs)

theta = 10
pulselen = Fs*2

X = np.zeros([T*Fs,5])
X[int(T*Fs/2),0] = 50
X[:,1] = np.sin(ta*theta*2*np.pi)
X[:,2] = gaussian(len(ta),Fs)*np.sin(ta*theta*2*np.pi)
X[:,3] = chirp(ta,f0=0,t1=T,f1=20,method='linear')
X[:int(T*Fs/2),4] = np.sin(ta[:int(T*Fs/2)]*theta*2*np.pi)
X[int(T*Fs/2):,4] = 10*np.sin(ta[int(T*Fs/2):]*theta*2*np.pi)

X += 0.25*np.random.randn(*X.shape)

for i in range(5):
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=[7,2])
    ax.plot(ta,X[:,i])
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time (s)')
    fig.subplots_adjust(bottom=0.25)

    if blnsave:
        fig.savefig(f'activity_timeseries_{i}.png')


NFFT = 4*Fs
figs = []
for i in range(5):
    x = X[:,[i]].T

    fig,axs = plt.subplots(nrows=len(param_sets),ncols=1,figsize=[4.2,4.75],
                           sharex=True,sharey=True)
    for j,params in enumerate(param_sets):
        winlen = params[0]
        NW = params[0]*params[1]
        K = params[2]
        S, freqs, times, ntapers = spec_utils.quick_mtspecgram(x, Fs, ta,
                                                               movingwin=(winlen,1/4),
                                                               NFFT=NFFT, NW=NW)

        axs[j].set_title(f'T={params[0]}, W={params[1]}, K={params[2]}')
        axs[j].imshow(10*np.log10(S[0,:,:]),aspect='auto',interpolation=None,
                      extent=[winlen/2,T-winlen/2,freqs[0],freqs[-1]],origin='lower')
        axs[j].set_ylabel('Freq (Hz)')


    axs[0].set_ylim([0,30])
    axs[0].set_xlim([0,T])
    axs[-1].set_xlabel('Time (s)')
    fig.subplots_adjust(hspace=0.4)
    figs.append(fig)

    if blnsave:
        fig.savefig(f'activity_spectrogram_{i}.png')
