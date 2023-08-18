import numpy as np
import numpy.random as nr
import numpy.linalg as nl
import matplotlib.pyplot as plt
import MIMO

t = 32; r = 32; 
numRF = 8;      
N_Beam = 24;    
G = 32;         
ITER = 10;      
L = 5;          

omp_thrld = 10;
SNRdB = np.arange(10,55,10);
SNR = 10**(SNRdB/10);
mseOMP = np.zeros(len(SNRdB));
mseGenie = np.zeros(len(SNRdB));

A_T = MIMO.ArrayDictionary(G,t);
A_R = MIMO.ArrayDictionary(G,r);

FRF,FBB = MIMO.RF_BB_matrices(t,numRF,N_Beam);
WRF,WBB = MIMO.RF_BB_matrices(r,numRF,N_Beam);
Qtil = np.kron(np.matmul(np.transpose(FBB),np.transpose(FRF)),np.matmul(MIMO.H(WBB),MIMO.H(WRF)));

for ix in range(ITER):
    print(ix)
    
    alpha,Psi,A_R_genie,A_T_genie = MIMO.mmWaveMIMOChannelGenerator(A_R,A_T,G,L);
    H = np.sqrt(t*r/L)*nl.multi_dot([A_R_genie,np.diag(alpha),MIMO.H(A_T_genie)]);
    
    ChNoise = 1/np.sqrt(2)*(nr.normal(0,1,(N_Beam,N_Beam))+1j*nr.normal(0,1,(N_Beam,N_Beam)));
    for cx in range(len(SNRdB)):
        Yrec = np.sqrt(SNR[cx])*nl.multi_dot([MIMO.H(WBB),MIMO.H(WRF),H,FRF,FBB]) + ChNoise;
        y = np.reshape(Yrec,(N_Beam*N_Beam,1),order='F');
        
        Q = np.sqrt(SNR[cx])*np.matmul(Qtil,np.kron(np.conj(A_T),A_R));
        hb_OMP = MIMO.OMP(y,Q,omp_thrld);
        H_OMP = nl.multi_dot([A_R, np.reshape(hb_OMP,(r,t),order='F'),MIMO.H(A_T)]);
        mseOMP[cx] = mseOMP[cx] + nl.norm(H-H_OMP)**2/(t*r);
        
        Qbar = np.sqrt(SNR[cx])*np.matmul(Qtil,Psi);
        alphaEst = np.matmul(nl.pinv(Qbar),y);
        H_Genie = nl.multi_dot([A_R_genie,np.diag(alphaEst.flatten()),MIMO.H(A_T_genie)]);
        mseGenie[cx] = mseGenie[cx] + nl.norm(H-H_Genie)**2/(t*r);
        
#write your code here

mseOMP = mseOMP/ITER; 
mseGenie = mseGenie/ITER;

plt.yscale('log')
plt.plot(SNRdB, mseOMP,'r-s');
plt.plot(SNRdB, mseGenie,'b^-.');
plt.grid(1,which='both')
plt.legend(["OMP", "Genie"], loc ="lower left");
plt.suptitle('NMSE Comparision mmWave MIMO Channel Estimation')
plt.ylabel('NMSE')
plt.xlabel('SNRdB') 


