import numpy as np 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm,colors
import os,sys,argparse,time
from matplotlib.colors import LinearSegmentedColormap
import scipy.io as io
from scipy.special import expi as ei
from scipy.special import gamma


nt =100000


rt = 1.8
N_preplasma = 1e15
frequency0 = 0.2;    
lt = 5


taue = 0.6e-12


# E0 = np.array([0.9,2.1,3])
E0 = np.array([0.9,2.1,3])
nuc0 = 10e12/0.9*E0
# nuc0 = 10e12*np.ones(len(E0))
# nuc0 = 1e12*np.array([9.2,26,38.7])
# nuc0 = 1e12*np.array([9.2,26,138.7])

# nuc0

E_O = 12.6; 
n_O2 = 3e22; 




I_p_O = E_O *0.03675



phi0 = np.pi*0


ht = lt/nt



t1 = np.linspace(-lt/2,lt/2,nt)*rt

f_width = nt/(rt*lt)# 频率域长度
frequency = np.linspace(f_width/2,-f_width/2,nt) # 频率坐标


echarge = 1.6e-19
epsilon0 = 8.854e-12
mu0 = 1.257e-6
emass = 9.1e-31
hbar = 1.0546e-34
light = 3.0e8
pi = 3.1415926575897932
kb = 1.380649e-23
K_to_ev = 1.29e-4


        

n2 = 1.3e-19;                      


rt *= 1e-12;
E0 *= 2e8;
frequency0 *= 1e12;
omega0 = frequency0*2*pi;
n_O2 *= 1e6;
E_O *= echarge;

n2 *= 1e-4;

eps = 1e-12;

co_impact_O = 1e-15*1e-4*np.sqrt(8*E_O/pi/emass);

dt = rt*lt/nt




def adk(E,I_p):
    if E>1e5:
        E_au = 1/514.2*(E*1e-9)
        Z_star = 1
        n_star = Z_star/np.sqrt(2*I_p)
        l_star = n_star-1

        A = 2**(2*n_star)/n_star/gamma(n_star+l_star+1)/gamma(n_star-l_star)
        B = 1

        E_au2 = E_au

        rate = np.sqrt(6/np.pi)*A*B*I_p*np.power(2*(2*I_p)**1.5/np.abs(E_au2),2*n_star-1.5)*np.exp(-2*(2*I_p)**1.5/3/np.abs(E_au2))
        rate /= 2.42e-17


        return rate
        # return 0


    else:
        return 0



rate1 = np.array([[12.6534,0],[13,0.016],[14,0.067],[15,0.119],[16,0.197],[17,0.281],[18,0.363],[19,0.446],[20,0.561],[21,0.674],[22,0.784],[23,0.890],[24,0.992],[25,1.089],[26,1.181],[27,1.269],[28,1.353],[29,1.432],[30,1.508],[31,1.579],[32,1.647],[33,1.711],[34,1.771],[35,1.829],[36,1.883],[37,1.935],[38,1.983],[39,2.03,],[40,2.073],[41,2.115],[42,2.154],[43,2.191],[44,2.226],[45,2.260],[46,2.291],[47,2.321],[48,2.349],[49,2.376],[50,2.402],[52,2.448],[54,2.490],[56,2.528],[58,2.561],[60,2.591],[62,2.618],[64,2.641],[66,2.662],[68,2.680],[70,2.696],[72,2.710],[74,2.722],[76,2.732],[78,2.740],[80,2.747],[82,2.753],[84,2.758],[86,2.761],[88,2.763],[90,2.765],[92,2.765],[94,2.765],[96,2.763],[98,2.762],[100,2.75],[110,2.73],[120,2.70],[130,2.67],[140,2.62],[150,2.58],[160,2.54],[170,2.49],[180,2.44],[190,2.40],[200,2.35],[210,2.31],[220,2.26],[230,2.22],[240,2.18],[250,2.14],[260,2.10],[270,2.07],[280,2.03],[290,1.99],[300,1.96],[310,1.93],[320,1.90],[330,1.87],[340,1.84],[350,1.81],[360,1.78],[370,1.75],[380,1.72],[390,1.70],[400,1.67],[410,1.65],[420,1.63],[430,1.60],[440,1.58],[450,1.56],[460,1.54],[470,1.52],[480,1.50],[490,1.48],[500,1.46],[510,1.44],[520,1.42],[530,1.41],[540,1.39],[550,1.37],[560,1.36],[570,1.34],[580,1.33],[590,1.31],[600,1.30],[610,1.28],[620,1.27],[630,1.25],[640,1.24],[650,1.23],[660,1.22],[670,1.20],[680,1.19],[690,1.18],[700,1.17],[710,1.16],[720,1.14],[730,1.13],[740,1.12],[750,1.11],[760,1.10],[770,1.09],[780,1.08],[790,1.07],[800,1.06],[810,1.05],[820,1.04],[830,1.03],[840,1.03],[850,1.02],[860,1.01],[870,1.00],[880,0.99],[890,0.98],[900,0.98],[910,0.97],[920,0.96],[930,0.95],[940,0.94],[950,0.94],[960,0.93],[970,0.92],[980,0.92],[990,0.91]])



def impact_rate(Te,species):

    rate = 0
    if species == 1:
        eta = E_O/kb/Te
        if Te>300 and Te<1e10:
            rate = -co_impact_O*np.sqrt(eta)*ei(-eta)
    ve = np.sqrt(Te*kb*3/emass)
    # norm = Te*K_to_ev/(100/np.exp(1))
    # sigma_i = 2.8e-20/0.4*np.log(norm)/norm

    # ind1 = np.argmin(np.abs(rate1[:,0]-Te*K_to_ev))
    # sigma_i = rate1[ind1,1]*1e-20
    
    # rate = sigma_i*ve
    return rate


N_ion_O = np.zeros(nt)

nu_c = np.zeros(nt)
T_e = np.zeros(nt)
T_pl = np.zeros(nt)
N_e = np.zeros(nt)
E = np.zeros(nt,dtype=np.complex64)

N_ion_O[0] = N_preplasma*1e6

T_e[0] = 300
T_pl[0] = 300
N_e[0] = N_ion_O[0]








for jt in range(nt):
    t = jt*ht-0.5*lt
    E[jt] = np.exp(-t**2/0.72)*np.exp(-1j*(omega0*t*rt+phi0))



ve_new = 0

ne_data = np.zeros((nt,3))
fig, ax = plt.subplots(2,1,figsize=(6,8))

for i in range(3):


    for jt in np.arange(1,nt):



        omega_probe = omega0
        omegap2 = N_e[jt-1]* echarge*echarge/emass/epsilon0
        ep = 1-omegap2/(omega_probe**2+1j*omega_probe*nu_c[jt-1])
        # ep = 1-omegap2/(omega_probe**2+1j)
        n_com = np.sqrt(ep)
        R = np.abs((1-n_com)/(1+n_com))**2
        T1 = 1-R
        T1 = np.max([0,T1])

        nI = np.imag(n_com)
        alpha = omega_probe/light*nI

        T2 = np.exp(-2*alpha*100e-6)

        E1 = E[jt]*np.sqrt(T1*T2)
        # E1 = E[jt]

        amplitude = np.abs(E1);
        # I_inst = epsilon0*light*np.real(E1)*np.real(E1)*E0[i]*E0[i]
        I_inst = 1/2*epsilon0*light*np.abs(E1)*np.abs(E1)*E0[i]*E0[i]
        
        Te = T_e[jt-1]

        vi_O = (n_O2-N_ion_O[jt-1])*impact_rate(Te,1);

        ve = ve_new
        sigma = 5e-22

        # sigma = 0
        # if Te>2*K_to_ev:
        #     sigma = pow(10,2-2/3*np.log10(Te*K_to_ev))*1e-20
        # nu_c[jt] = sigma*ve*n_O2*1e-4
        # if nu_c[jt]<1e12:
        #     nu_c[jt] = 1e12

        # nu_c[jt] = np.abs(sigma*np.sqrt(Te*kb*3/emass)*n_O2)

        nu_c[jt] = nuc0[i]

        # ve_new = ve+(echarge*np.abs(np.real(E1))*E0[i]/emass-nu_c[jt-1]*ve)*dt;
        # nu_c[jt] = sigma*ve_new*n_O2
        # print(ve_new)
        # ve_new = ve+(echarge*np.abs(np.real(E1))*E0[i]/emass)*dt;

        # print("")
        sigma_c = echarge*echarge/emass/epsilon0/light*(nu_c[jt]/(omega0*omega0+nu_c[jt]*nu_c[jt]))
        # sigma_c = emass*nu_c[jt]*ve**2
        # print("%.1e,%.1e,%.1e,%.1e"%(ve,ve_new,nu_c[jt],Te))


        

        heating = 2.0/3/kb*sigma_c*I_inst;
        # heating = 1/2*emass*(ve_new**2-ve**2)/echarge/K_to_ev;
        # cooling1 = 2.0/3/kb*vi_O*np.min([1.5*kb*Te,E_O]);
        # cooling1 = vi_O*np.min([Te*K_to_ev,E_O/echarge])/K_to_ev;
        cooling1 = vi_O*E_O/echarge/K_to_ev;
        # cooling1 = 0;
        # cooling2 = f_pl*(T_e[jt-1]-T_pl[jt-1]);
        cooling2 = 2/33048*nu_c[jt]*(T_e[jt-1]-T_pl[jt-1]);
        # cooling2 = 0;



        # T_e[jt] = Te+dt*(-cooling1-cooling2)+heating;
        # T_e[jt] = (emass*ve_new**2-vi_O*E_O*dt)/kb;
        T_e[jt] = Te+dt*(heating-cooling1-cooling2);
        # T_e[jt] = Te+heating;

        T_pl[jt] = T_pl[jt-1]+dt*cooling2;


        if T_e[jt]*K_to_ev<E_O/echarge and N_ion_O[jt-1]>N_preplasma*1e6:
            N_ion_O[jt] = N_ion_O[jt-1]+dt*adk(np.abs(np.real(E1))*E0[i],I_p_O)*(n_O2-N_ion_O[jt-1])+(vi_O*N_e[jt-1])*dt-N_ion_O[jt-1]/taue*dt
        else:
            N_ion_O[jt] = N_ion_O[jt-1]+dt*adk(np.abs(np.real(E1))*E0[i],I_p_O)*(n_O2-N_ion_O[jt-1])+(vi_O*N_e[jt-1])*dt
        # N_ion_O[jt] = N_ion_O[jt-1]+dt*adk(np.abs(np.real(E1))*E0[i],I_p_O)*(n_O2-N_ion_O[jt-1])+(vi_O*N_e[jt-1])*dt-N_ion_O[jt-1]/taue*dt


        N_ion_O[jt] = np.min([N_ion_O[jt],n_O2]);

        N_e[jt] = N_ion_O[jt]

    ne_data[:,i] = N_e/3e6

    ax[0].plot(t1,T_e*K_to_ev,label=r'$%.1f$ MV/cm'%(E0[i]/2e8))
    # ax[0].plot(t1,nu_c/1e12,label=r'$%.1f$ MV/cm'%(E0[i]/2e8))
    ax[1].plot(t1,N_e/3e6,label=r'$%.1f$ MV/cm'%(E0[i]/2e8))
    # ax[1].semilogy(t1,N_e/1e6,label=r'$%.1f$ MV/cm'%(E0[i]/2e8))




two_columns = np.column_stack((t1, ne_data))


# np.savetxt('data.txt',two_columns[::100], fmt='%.2f\t%.2e\t%.2e\t%.2e\t')


ax[1].set_ylabel(r'$\rho_e$ (cm$^{-3}$)',fontsize=13)
ax[0].legend(fontsize=13)



ax[0].set_ylabel('nuc (THz)',fontsize=13)
ax[1].legend(fontsize=13)

plt.subplots_adjust(left=0.16,right=0.9)

ax[1].set_xlabel('t (ps)',fontsize=13)

plt.show()

