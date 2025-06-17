function P = rungek2(Nt,ET,ETOW,Delt,T,PHIT,PHITOW,LASEI1);

T = T ;
PK1  =  capti(Nt,ET,ETOW,T,PHIT,PHITOW,LASEI1);
P1 = PK1*Delt;

T1 = T + (Delt/2);
PHIT1 = PHIT + (P1/2) ;
PK2  =  capti(Nt,ET,ETOW,T1,PHIT1,PHITOW,LASEI1);
P2 = PK2*Delt;

T2 = T + (Delt/2);
PHIT2 = PHIT + (P2/2) ;
PK3  =  capti(Nt,ET,ETOW,T2,PHIT2,PHITOW,LASEI1);
P3 = PK3*Delt;

T3 = T + (Delt);
PHIT3 = PHIT + (P3) ;
PK4  =  capti(Nt,ET,ETOW,T3,PHIT3,PHITOW,LASEI1);
P4 = PK4*Delt;

P = (1/6)*[(P1)+(2*P2)+(2*P3)+(P4)];


