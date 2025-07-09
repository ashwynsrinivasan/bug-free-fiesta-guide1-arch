function dnditl = rungeit3(nnT,eeT,T);

Delt = 10*(10^-12);
ET = eeT;
NT = nnT;
LK1 = thirdi(eeT,nnT);
L1 = LK1*Delt;

eeT1 = eeT ;
nnT1 = nnT + (L1/2) ;
LK2  =  thirdi(eeT1,nnT1);
L2 = LK2*Delt;

eeT2 = eeT ;
nnT2 = nnT + (L2/2) ;
LK2  =  thirdi(eeT2,nnT2);
L3 = LK2*Delt;

eeT3 = eeT ;
nnT3 = nnT + (L3/2) ;
LK3  =  thirdi(eeT3,nnT3);
L4 = LK3*Delt;

dnditl = (1/6)*[(L1)+(2*L2)+(2*L3)+(L4)];
