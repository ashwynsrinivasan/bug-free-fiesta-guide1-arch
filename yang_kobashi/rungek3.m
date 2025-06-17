function L = rungek3(ET,NT,Delt);

ET = ET;
NT = NT;
LK1 = calinty(ET,NT);
L1 = LK1*Delt;

ET1 = ET ;
NT1 = NT + (L1/2) ;
LK2  =  calinty(ET1,NT1);
L2 = LK2*Delt;

ET2 = ET ;
NT2 = NT + (L2/2) ;
LK2  =  calinty(ET2,NT2);
L3 = LK2*Delt;

ET3 = ET ;
NT3 = NT + (L3/2) ;
LK3  =  calinty(ET3,NT3);
L4 = LK3*Delt;


L = (1/6)*[(L1)+(2*L2)+(2*L3)+(L4)];
