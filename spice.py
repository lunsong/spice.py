from numpy import zeros, array, concatenate as concat, linspace, eye
from numpy.linalg import solve as lsolve, eig
from numpy.random import normal
from collections import namedtuple

blank_eq = lambda : zeros(Edge.total+len(MOSFET.all)+1)

class Vertex:
    all = {}
    def __new__(cls,name=None):
        if isinstance(name, Vertex):
            return name
        if name==None: name = "_"+str(len(Vertex.all))
        instance = Vertex.all.get(name,None)
        if instance!=None:
            return instance
        instance = super().__new__(cls)
        instance.number = len(Vertex.all)
        instance.edges = {}
        Vertex.all[name] = instance
        return instance

    def get_loops(self):
        if not hasattr(self,"parent"):
            self.parent = None
        for n in filter(lambda x: x!=self.parent, self.edges.keys()):
            if hasattr(n,"parent"):
                if n.number > self.number:
                    yield (self,n)
            else:
                n.parent = self
                yield from n.get_loops()
    @property
    def loops(self):
        if not hasattr(self,"_loops"):
            self._loops = list(self.get_loops())
        return self._loops
    def divergence(self):
        return sum(e.I(self) for e in self.edges.values())
    @property
    def V(self):
        if self.parent == None:
            return blank_eq()
        return self.edges[self.parent].V(self) + self.parent.V
    def __repr__(self):
        return "n"+str(self.number)

def IV_Wrapper(f):
    def g(obj,x):
        eq = blank_eq()
        f(obj, eq)
        if x==obj.vertices[1]:
            eq *= -1
        return eq
    return g


class Edge:
    total = 0
    def __init__(self,x,y):
        x = Vertex(x)
        y = Vertex(y)
        if x in y.edges.keys():
            z = Vertex()
            R(x,z)
            x = z
        self.vertices = [x,y]
        x.edges[y] = self
        y.edges[x] = self
        self.number = Edge.total
        Edge.total += 1

    @IV_Wrapper
    def I(self,eq):
        eq[self.number] = 1

class Resistor(Edge):
    def __init__(self,x,y,R=0):
        super().__init__(x,y)
        self.R = R

    def V(self,x):
        return self.I(x) * self.R


class VoltageSource(Edge):
    def __init__(self,x,y,V):
        super().__init__(x,y)
        self.U = V

    @IV_Wrapper
    def V(self,eq):
        eq[-1] = self.U


class Capacitor(VoltageSource):
    def __init__(self,x,y,C,U=0):
        super().__init__(x,y,U)
        self.C = C


class Diode(Edge):
    all = []
    def __init__(self,x,y,VD=0.7):
        super().__init__(x,y)
        self.VD = VD
        self.on = False
        Diode.all.append(self)

    @IV_Wrapper
    def I(self, eq):
        if self.on:
            eq[self.number] = 1

    @IV_Wrapper
    def V(self,eq):
        if self.on:
            eq[-1] = self.VD
        else:
            eq[self.number] = 1

class MOSFET(Edge):
    all = []
    def __init__(self,D,S,G,V_TN=1.8,K=90,l=1e-2,V_DS=2,
            channel="N",mode="enhancement"):
        """
        D,S,G are drain,source,gate respectively
        V_TN is the threshold voltage, and is always postive
        regarless of the channel type. default 1.8
        K and l(ambda) are model parameters
        """
        super().__init__(D,S)
        self.mos_number = len(MOSFET.all)
        MOSFET.all.append(self)
        self.D = Vertex(D)
        self.S = Vertex(S)
        self.G = Vertex(G)
        self.V_TN = V_TN
        self.V_DS = V_DS
        self.K = K
        self.l = l
        assert channel in ("N","P")
        self.channel = channel
        assert mode in ("enhancement","depletion")
        self.mode = mode

    @IV_Wrapper
    def V(self,eq):
        eq[Edge.total + self.mos_number] = 1

    def Id_and_derivs(self, cur):
        V_GS = (self.G.V-self.S.V) @ cur
        V_DS = (self.D.V-self.S.V) @ cur
        if self.channel == "P":
            V_DS = -V_DS
            V_GS = -V_GS
        if self.mode == "depletion":
            V_GS += self.V_TN
        if V_GS<self.V_TN or V_GS<0:
            Id = dV_GS = dV_DS = 0
        elif V_GS-self.V_TN > V_DS:
            Id = self.K * V_DS * (2*(V_GS-self.V_TN)-V_DS) * (1+V_DS*self.l)
            dV_GS = self.K * 2 * V_DS * (1+V_DS*self.l)
            dV_DS = self.K * 2 * (V_GS - self.V_TN - V_DS) * (1+V_DS*self.l)
            dV_DS += self.K * V_DS * (2*(V_GS-self.V_TN)-V_DS) * self.l
        else:
            Id = self.K * (V_GS-self.V_TN)**2 * (1+V_DS*self.l)
            dV_GS = self.K * 2 * (V_GS-self.V_TN) * (1+V_DS*self.l)
            dV_DS = self.K * (V_GS-self.V_TN)**2 * self.l
        if self.channel == "P":
            Id = -Id
        return Id, dV_GS, dV_DS

def solve(gnd=None, alp=1, eps=1e-14,N=1000, disp=0,full_output=False):
    """
    solve the current circuit
    return (sol, done, step) where sol is the solution, done is True if
    converged, step is the total iteration steps.
    try smaller alp if not converged.
    """
    if full_output:
        output = namedtuple("res",["sol","done","step","msg"])
    else:
        output = lambda *args: args[0]
    if gnd==None: gnd = GND
    loops = gnd.loops
    vertices = list(Vertex.all.values())[1:]
    mos_sol = array([mos.V_DS for mos in MOSFET.all],dtype=float)
    step = 0
    for _ in range(100):
        eqs = [n.divergence() for n in vertices]
        eqs += [a.V-b.V-a.edges[b].V(a) for a,b in loops]
        eqs = array(eqs)
        if disp>1: print(eqs[:,:Edge.total])
        if disp>1: print(eig(eqs[:,:Edge.total])[1][:,-1])
        sol = -lsolve( eqs[:,:Edge.total], eqs[:,Edge.total:])
        sol = concat((sol, eye(len(MOSFET.all)+1)))
        done = True
        cur = sol @ concat((mos_sol,(1,)))
        for e in Diode.all:
            if e.on and cur[e.number] < 0:
                done = False
                e.on = False
            if not e.on and cur[e.number] > e.VD:
                done = False
                e.on = True 
        for sub_step in range(N):
            cur = sol @ concat((mos_sol,(1,)))
            Jacobian = []
            F = []
            sub_done = False
            for e in MOSFET.all:
                Id,df_dVGS,df_dVDS = e.Id_and_derivs(cur)
                F.append(Id-cur[e.number])
                J_Id = sol[e.number,:-1]
                J_VDS = zeros(len(MOSFET.all))
                J_VDS[e.mos_number] = 1
                J_VGS = (e.G.V-e.S.V) @ sol[:,:-1]
                Jacobian.append(df_dVGS * J_VGS + df_dVDS * J_VDS-J_Id)
            Jacobian = array(Jacobian)
            if disp>1: print(Jacobian)
            if disp>1: print(eig(Jacobian))
            delta = lsolve(Jacobian, F)
            max_delta = max(abs(delta))
            if disp>0: print(max_delta)
            if max_delta > alp:
                delta *= alp / max_delta
            elif max_delta < eps:
                sub_done = True
                break
            mos_sol -= delta
        step += 1 + sub_step
        if not sub_done: return output(cur,False,step,
                "Maximum MOSFET step reached")
        if step > N: return output(cur,False,step,
                "Maximum total step reached")
        if done:
            for idx,mos in enumerate(MOSFET.all):
                mos.V_DS = mos_sol[idx]
            return output(cur,True,step,"")
    return output(cur, False, step,
            "Maximum outer step reached")

V = VoltageSource
C = Capacitor
R = Resistor
D = Diode
GND = Vertex("GND")

def clear():
    Vertex.all = {}
    Edge.total = 0
    Diode.all = []
    MOSFET.all = []
    global GND
    GND = Vertex("GND")
