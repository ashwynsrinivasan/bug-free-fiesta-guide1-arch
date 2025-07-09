function DEDT = caleti(Nt,ET,ETOW,ERin,T,PHIT,PHITOW);

Ws = 2.5*(10^9);
elfa = 3;
format long g;
GN = 1.1*(10^-12);
Ts = 2*(10^-9);
Tp = 2*(10^-12);
Tow = 1*(10^-9);
No = 1.1*(10^24);
NTH = No+1/(GN*Tp);
K = 0.02;
R = 1.71*(10^12);   %2.30661270860069 -0.193460952792062
X = 1.45;
Wo = 5*(10^9);
V = 4*(10^-16);
Delt = 10*(10^-12);
pico = deg2rad(-82.7) + mod((3*X),(2*pi));

DEDT = (0.5*GN*(Nt-NTH)*ET) + (X/Tow)*ETOW*cos( (pico) + (PHIT - PHITOW) ) + (R/(2*V*ET));% + (ERin/(Delt)) ;



