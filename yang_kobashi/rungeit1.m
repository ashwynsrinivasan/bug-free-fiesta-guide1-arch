function dedit = rungeit1(nnT,eeT,T);

h = 10*(10^-12);
DELT = 1*(10^-9);
T = T;
KK1  =  firsti(nnT,eeT);
K1 = KK1*h;

T1 = T + (DELT/2);
eeT1 = eeT + (K1/2) ;
KK2  =  firsti(nnT,eeT1);
K2 = KK2*h;

T2 = T + (DELT/2);
eeT2 = eeT + (K2/2) ;
KK3  =  firsti(nnT,eeT2);
K3 = KK3*h;

T3 = T + (DELT);
eeT3 = eeT + (K3) ;
KK4  =  firsti(nnT,eeT3);
K4 = KK4*h;

K = (1/6)*[(K1)+(2*K2)+(2*K3)+(K4)];
dedit = K;