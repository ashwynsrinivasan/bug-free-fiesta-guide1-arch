function [DNDT] = calinty(ET,NT)

Ts = 2*(10^-9);
No = 1.1*(10^24);
R = 1.71*(10^12);
GN = 1.1*(10^-12);
Tp = 2*(10^-12);
NTH = No+1/(GN*Tp);
Jth = (NTH/Ts); 
J  = 1.3*(Jth);

DNDT = J - (NT/Ts) - [GN*(NT-No)*(abs(ET)^2)] ;
