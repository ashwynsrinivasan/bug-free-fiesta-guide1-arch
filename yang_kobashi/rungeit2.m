function dpditl = rungeit2(pit,nnT,eeT,T);

Delt = 10*(10^-12);
T = T ;
PK1  =  secondi(pit,nnT,eeT,T);
P1 = PK1*Delt;

T1 = T + (Delt/2);
pit1 = pit + (P1/2);
PK2  =  secondi(pit1,nnT,eeT,T);
P2 = PK2*Delt;

T2 = T + (Delt/2);
pit2 = pit + (P2/2);
PK3  =  secondi(pit2,nnT,eeT,T);
P3 = PK3*Delt;

T3 = T + (Delt);
pit3 = pit + (P3);
PK4  =  secondi(pit3,nnT,eeT);
P4 = PK4*Delt;

P = (1/6)*[(P1)+(2*P2)+(2*P3)+(P4)];
dpditl = P;