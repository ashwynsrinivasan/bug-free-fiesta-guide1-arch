function DPIDT = capti(Nt,ET,ETOW,T,PHIT,REVPIT,LASEI1)

format long g;
Ws = 2.5*(10^9);
elfa = 3;
GN = 1.1*(10^-12);
Ts = 2*(10^-9);
Tp = 2*(10^-12);
Tow = 1*(10^-9);
No = 1.1*(10^24);
NTH = No+1/(GN*Tp);
K = 0.02;
R = 1.71*(10^12);
X = 1.45;
Wo = 5*(10^9);                        %-0.193460952792062
V = 4*(10^-16);
Delt = 10*(10^-12);
pico = deg2rad(-82.7) +mod((3*X),(2*pi));

DPIDT = (0.5*elfa*GN*(Nt-NTH) ) - [ (X/Tow)*(ETOW/ET)*sin((pico) + ((PHIT) - REVPIT) ) ];% + LASEI1/Delt ;
    
     