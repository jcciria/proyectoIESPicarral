import numpy as np
import matplotlib.pyplot as plt
from spatialmath.base import plotvol2, transl2, trot2, trplot2,plot_point, e2h, h2e,vexa,skewa, trnorm2,trlog2,trexp2
from matplotlib.lines import Line2D
from scipy.linalg import expm, logm
from scipy.stats import uniform

def transformacion2D (alpha=0, t=(0,0)):
    v = np.array(t)
    if v.size != 2:
        raise ValueError ('t tiene %d componentes, en lugar de 2'%v.size)
    T = np.zeros ((3,3))
    T[:2, :2] = np.array ([[np.cos(alpha), -np.sin(alpha)],[ np.sin(alpha),np.cos(alpha)]])
    T[2,2] = 1
    T[:2, 2] = np.array(v).reshape(2,)
    return T
    
def inversa2D (T):
    if not esSE2(T):
        raise ValueError ('T no es una transformación válida en SE(2)')
    TInv = np.zeros ((3,3))
    TInv [:2, :2]= np.transpose (T[:2, :2])
    TInv[:2, 2] = - (TInv [:2, :2] @  np.array(T[:2,2]).reshape(-1,1)).reshape(2,)
    TInv [2,2]= 1
    return TInv


def angulo (w):
    v = np.array(w)
    if v.size != 2:
        raise ValueError("angulo: v debe tener 2 componentes, y no %d "%v.size)

    x,y= [x for x in v.reshape(2)]
    if x==0 and y==0:
            raise ValueError("El ángulo del vector (0,0) no está definido")
    else:
        return np.arctan2 (y,x)

def esSE2 (T):
    atol = 1e-12
    
    if not isinstance(T, np.ndarray):
        raise ValueError ('T debe ser un np.array, no un %s'%type(T))
        
    if T.shape != (3,3): # Matriz 3x3
        return False
    elif not np.allclose(T[2, :], (0, 0, 1), atol=atol, rtol=0):  #Última fila: (0,0,1)
        return False;
    elif  np.allclose ( (T[0,0]-T[1,1], T[0,1]+T[1,0], np.linalg.det(T[:2,:2])), (0,0,1), atol = atol, rtol=0):
        return True;
    else: 
        return False;
        
def muestraDesplazamiento (T0, T, nIntervalos=10, color='gray', originsize=0, origincolor='red'):
    if not all (esSE2(t) for t in (T0,T)):
        raise ValueError ('Matrices no válidas en SE(2)')

    nIntervalos = int (nIntervalos)
    if nIntervalos <1:
        nIntervalos = 10
        
    logT =logm(T)
    Ti = T0
    deltaT = 1/nIntervalos
    for i in range(nIntervalos):
        Ti =trnorm2(Ti @ expm(logT*deltaT))
        trplot2(Ti, color=color,originsize=originsize, originColor=origincolor)

    
def transfCoord (T, coord):
    if not esSE2(T):
        raise ValueError ('La matriz no es una transformación válida en SE(2)')
    return h2e(inversa2D(T)@ e2h(coord))


def perpendicular (w):
    v = np.array(w).reshape(2,)
    if v.size != 2:
        raise ValueError('v debe tener dos componentes')
    if (np.linalg.norm(v) == 0):
        raise ValueError('No está definido el vector perpendicular a (0,0) ')
    v2 = np.array ((-v[1], v[0]))
    return v2/np.linalg.norm(v2)
    
def inicializa (L=5):
    if L<=0:
        L = 5
    angulo = uniform.rvs(loc=0, scale=2*np.pi)  
    P1 = np.array ((np.cos(angulo), np.sin(angulo))) * 0.5
    P2 = -P1

    angulo = uniform.rvs(loc=0, scale=2*np.pi)  
    anguloOrigen = uniform.rvs(loc=0, scale=2*np.pi)  
    origen = uniform.rvs(loc=0, scale=L, size=1)
    T0 = transformacion2D(alpha =angulo, t= (origen*np.cos(anguloOrigen), origen*np.sin(anguloOrigen)) )

    return P1, P2, T0

def trayectoria (listTransf, nIntervalos = 100):

    nIntervalos = int (nIntervalos)
    if nIntervalos <1:
        nIntervalos = 100
        
    if type (listTransf) not in (tuple, list):
        listTransf = [listTransf]
        
    for T,i in zip(listTransf, range(len(listTransf))):
        if not esSE2(T):
            raise ValueError ('trayectoria: La matriz %d no es válida'%i)
    
    if len(listTransf) !=0:      
        T = listTransf[0]
        if nIntervalos <1:
            nIntervalos = 100
        deltaT = 1/nIntervalos
        centro = transfCoord (inversa2D(T),(0,0))
        centros= [tuple (centro.reshape(2,)) ]
        for t in listTransf[1:]:
            lt =logm(t)
            if not np.allclose(vexa(lt)[2],0, atol=1e-12) and not np.allclose(vexa(lt)[:2],0, atol=1e-12):
                for i in range(nIntervalos):
                    T = trnorm2(T @ expm(lt*deltaT))
                    centro = transfCoord (inversa2D(T),(0,0))
                    centros.append(tuple (centro.reshape(2,)))
            else:
                T = T @ t
                centro = transfCoord (inversa2D(T),(0,0))
                centros.append(tuple (centro.reshape(2,)))
        centros = np.array(centros)
    else:
        centros = np.empty((0,2))
    return centros
    
def muestraMovimiento (listTransf, colores = ('black', 'blue', 'orange','red', 'green')):
    if type (listTransf) not in (tuple, list):
        listTransf = [listTransf]

    if colores is None or len(colores)==0:
        colores = ('black', 'blue', 'orange','red', 'green')
        
    for T,i in zip(listTransf, range(len(listTransf))):
        if not esSE2(T):
            raise ValueError ('muestraMovimiento: La matriz %d no es válida'%i)
            
    if len (listTransf) == 0:  
        ejes=plotvol2([-5,5])
    else:
        #L = np.linalg.norm( transfCoord(listTransf[0], (0,0)))
        tray = trayectoria (listTransf)
        extremos = (np.min(tray [:,0])-1.5, np.max(tray [:,0])+1.5, np.min (tray [:,1])-1.5, np.max(tray [:,1])+1.5)
        ejes=plotvol2(extremos)
        
    
        T= listTransf[0]
        for t in listTransf[1:]:
            lt =logm(t)
            if not np.allclose(vexa(lt)[2],0, atol=1e-12) and not np.allclose(vexa(lt)[:2],0, atol=1e-12):
                    muestraDesplazamiento(T,t, color='gray')
            T = T @ t
                
        T= transformacion2D()
        for t,i in zip(listTransf, range(len(listTransf))):
            T = trnorm2(T @ t)
            trplot2(T, colores[i%len(colores)], originsize=0)
    
        handles, labels = ejes.get_legend_handles_labels()
    
        marco='T0'
        marcos =[(marco, colores[0])]
        for t, i in zip (listTransf[1:], range(1, len(listTransf))):
            marco=marco+'@T%d'%i
            marcos.append((marco, colores[i%len(colores)]))
    
        for nombre, color in marcos:
            ejes.add_line(Line2D([], [], color=color, label=nombre))
    
  
    ejes.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    a,b = ejes.get_legend_handles_labels()