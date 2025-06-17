function K = rungek1(Nt,ET,ETOW,ERin,T,PHIT,PHITOW);

h = 10*(10^-12);
DELT = 1*(10^-9);
T = T;
KK1  =  caleti(Nt,ET,ETOW,ERin,T,PHIT,PHITOW);
K1 = KK1*h;

T1 = T + (DELT/2);
ET1 = ET + (K1/2) ;
KK2  =  caleti(Nt,ET1,ETOW,ERin,T1,PHIT,PHITOW);
K2 = KK2*h;

T2 = T + (DELT/2);
ET2 = ET + (K2/2) ;
KK3  =  caleti(Nt,ET2,ETOW,ERin,T2,PHIT,PHITOW);
K3 = KK3*h;

T3 = T + (DELT);
ET3 = ET + (K3) ;
KK4  =  caleti(Nt,ET3,ETOW,ERin,T3,PHIT,PHITOW);
K4 = KK4*h;

K = (1/6)*[(K1)+(2*K2)+(2*K3)+(K4)];

