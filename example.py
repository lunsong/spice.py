from spice import V,R,MOSFET,solve,clear,Vertex
from numpy import linspace

#FIXME if a loop is consist only of the mosfets, then it provide
#a constraint on Vds 

#Circuit extracted from
#https://sites.bu.edu/engcourses/files/2016/08/mosfet-differential-amplifier.pdf

NMOS = lambda G,D,S,ratio,V_DS=2: MOSFET(G=G,D=D,S=S,mode="depletion",
        V_TN=1,channel="N",K=ratio*1e-5,l=1e-2,V_DS=V_DS)
PMOS = lambda G,D,S,ratio,V_DS=2: MOSFET(G=G,D=D,S=S,mode="depletion",
        V_TN=1,channel="P",K=ratio*5e-6,l=1e-2,V_DS=V_DS)

clear()

V("VCC","GND",10)
V1=V("V1","GND",5)
V2=V("V2","V1",4e-3)

# Lower current mirror
R("VCC",1,1e4)
NMOS(D=1,     G=1,S="GND",ratio= 1)
NMOS(D=4,     G=1,S="GND",ratio=.2)
NMOS(D="Vout",G=1,S="GND",ratio= 1)

# FIXME without following resistors the equations are singular
R("VCC","X1",1e-3)
R("VCC","X2",1e-3)
R("VCC","X3",1e-3)

# Upper current mirror
PMOS(D=2,G=2,S="X1",ratio=5/11)
PMOS(D=3,G=2,S="X2",ratio=5/11)

# Symmetric input
NMOS(D=2,G="V1",S=4,ratio=4)
NMOS(D=3,G="V2",S=4,ratio=4)

PMOS(S="X3",G=3,D="Vout",ratio=9/2)

print(solve(eps=1e-8,full_output=True,disp=1,N=5000))
vdc = linspace(-2,2,200)
vout = []
for v in vdc:
    V2.U = v
    res = solve(eps=1e-8,N=1000,disp=0,full_output=True)
    print(v,"\t",res.step,end="\r")
    if not res.done:
        print(res.msg)
        break
    vout.append(Vertex("Vout").V @ res.sol)
