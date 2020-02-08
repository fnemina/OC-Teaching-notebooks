import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import interact, interactive, fixed, interact_manual, AppLayout, GridspecLayout, Layout
import ipywidgets as widgets
from IPython.display import clear_output
from tqdm.notebook import tqdm
import pyOSOAA

list_satellite = np.load('list_satellite.npy',allow_pickle='TRUE').item()
list_aerosols = {"Troposférico":1, "Urbano":2, "Maritimo":3, "Costero":4}

satellite=widgets.Dropdown(options=list_satellite.keys(), description="Satélite");

chl = widgets.BoundedFloatText(value=5, min=0.001, max=100, description="Chla [mg/m3]");
wind = widgets.BoundedFloatText(value=0, min=0, max=30, description="W [m/s]");

aero = widgets.Dropdown(options=list_aerosols, value=3, description="Aerosol");
rh = widgets.BoundedIntText(value=80, min=0, max=100, description="RH [%]");
aot = widgets.BoundedFloatText(value=0.1, min=0, max=1, description="AOT");

view = widgets.BoundedIntText(value=0, min=0, max=45, description=r"$\theta [^\circ]$");
sun = widgets.BoundedIntText(value=30, min=30, max=65, description=r"$\theta_0[^\circ]$");
phi = widgets.BoundedIntText(value=90, min=0, max=270, description=r"$\phi[^\circ]$");

toa = widgets.Checkbox(description="Mostrar TOA", value=True)
too = widgets.Checkbox(description="Mostrar TOO")
rayleigh = widgets.Checkbox(description="Mostrar Rayleigh")
aerosol = widgets.Checkbox(description="Mostrar aerosol")
glint = widgets.Checkbox(description="Mostrar glint")
tau = widgets.Checkbox(description="Optical thickness")

button=widgets.Button(description='Run')

grid = GridspecLayout(5, 3)

grid[0, 0] = satellite; grid[1, 0] = chl; grid[1, 1] = wind;
grid[2, 0] = aero; grid[3, 0] = rh; grid[4, 0] = aot;
grid[2, 1] = view; grid[3, 1] = sun; grid[4, 1] = phi;
grid[0,1] = toa; grid[0,2] = too; grid[1,2] = rayleigh; grid[2,2] = aerosol; grid[3,2] = glint
grid[4,2] = tau;

def simulate(satellite="seawifs", sun=30, view=0, phi=90, level=1, chl=1, aero=3, 
             rh=80, aot=0.1, wind=0, ap=False, gettau=False):
    s = pyOSOAA.OSOAA(cleanup=True)

    # Oceano
    s.phyto.chl = chl
    s.sea.depth = 100
    s.sea.botalb = 0
    if chl == 0:
        s = pyOSOAA.osoaahelpers.ConfigureOcean(s, ocean_type="black")

    # Superficie
    s.sea.wind = wind
    s.sea.surfalb = 0

    # Modelo de aerosoles de Shetle and Fenn
    s.aer.SetModel(model=2, sfmodel=aero, rh=rh)
    s.aer.aotref = aot

    # Geometria de observacion
    s.view.phi = phi
    s.ang.thetas = sun
    view = view

    s.view.level = level
    
    if ap is True:
        s.ap.SetMot(0)
    # Corremos el código

    rho = []
    tauaer = []
    tauray = []
    for wa in tqdm(list_satellite[satellite], leave=False):
        if ap is True:
            s.ap.SetMot(0)
        s.wa = wa/1e3
        s.run()
        rho = np.append(rho, np.interp(view, s.outputs.vsvza.vza, s.outputs.vsvza.I))
        tauaer = np.append(tauaer, (s.outputs.profileatm.tau*np.mean(s.outputs.profileatm.mixaer))[-1])
        tauray = np.append(tauray, (s.outputs.profileatm.tau*np.mean(s.outputs.profileatm.mixray))[-1])
        
    if gettau:
        return tauray, tauaer
    else:
        return list_satellite[satellite], rho

def click(b):
    data =  {}
    i = 1
    total = toa.value+too.value+aerosol.value+rayleigh.value+2*glint.value+tau.value
    wl = list_satellite[satellite.value]
    if total > 0:
        if toa.value:
            print(f"Simulación {i} de {total}...", end="\r")
            wl, rtoa = simulate(satellite=satellite.value, level=1, 
                           chl=chl.value, aero=aero.value, rh=rh.value,
                           aot=aot.value, wind=wind.value,
                           sun=sun.value, view=view.value, phi=phi.value)
            i = i+1
            data["rtoa"] = rtoa
        
        if rayleigh.value:
            print(f"Simulación {i} de {total}...", end="\r")
            wl, rray = simulate(satellite=satellite.value, level=1, 
                               chl=0, aero=3, rh=80,
                               aot=0, wind=0,
                               sun=sun.value, view=view.value, phi=phi.value)
            i = i+1
            data["rray"] = rray
        
        if aerosol.value:
            print(f"Simulación {i} de {total}...", end="\r")
            wl, raer = simulate(satellite=satellite.value, level=1, 
                               chl=0, aero=aero.value, rh=rh.value,
                               aot=aot.value, wind=0, ap=True,
                               sun=sun.value, view=view.value, phi=phi.value)
            i = i+1
            data["raer"] = raer
        
        if too.value:
            print(f"Simulación {i} de {total}...", end="\r")
            wl, rtoo = simulate(satellite=satellite.value, level=3, 
                           chl=chl.value, aero=aero.value, rh=rh.value,
                           aot=aot.value, wind=0,
                           sun=sun.value, view=view.value, phi=phi.value)
            i = i+1
            data["rtoo"] = rtoo
            
        if glint.value:
            print(f"Simulación {i} de {total}...", end="\r")
            wl, tmpNO = simulate(satellite=satellite.value, level=1, 
                           chl=0, aero=aero.value, rh=rh.value,
                           aot=aot.value, wind=0,
                           sun=sun.value, view=view.value, phi=phi.value)
            i = i+1
            print(f"Simulación {i} de {total}...", end="\r")
            wl, tmpSI = simulate(satellite=satellite.value, level=1, 
                           chl=0, aero=aero.value, rh=rh.value,
                           aot=aot.value, wind=wind.value,
                           sun=sun.value, view=view.value, phi=phi.value)
            rgli = tmpSI-tmpNO
            i = i+1
            data["rglit"] = rgli
        if tau.value:
            print(f"Simulación {i} de {total}...", end="\r")
            taur, taua = simulate(satellite=satellite.value, level=1, 
                           chl=chl.value, aero=aero.value, rh=rh.value,
                           aot=aot.value, wind=wind.value,
                           sun=sun.value, view=view.value, phi=phi.value, gettau=True)
            i = i+1
            data["taur"] = taur
            data["taua"] = taua
            
    df = pd.DataFrame().from_dict(data)
    df.index = list_satellite[satellite.value]
    b.results = df
    clear_output(wait=True)
    display(grid, button)
    fig, ax1 = plt.subplots()
    if total > 0:
        plt.xlabel(r"$\lambda$ [nm]")
        plt.ylabel(r"$\rho$")    
        if toa.value:
            ax1.plot(wl, rtoa,'o-C0', label=r"$\rho_{t}$")
        
        if rayleigh.value:
            ax1.plot(wl, rray,'o-C1', label=r"$\rho_{r}$")
        
        if aerosol.value:
            ax1.plot(wl, raer,'o-C3', label=r"$\rho_{A}$")
        
        if too.value:
            ax1.plot(wl, rtoo,'o-C2', label=r"$\rho_{s}$")
            
        if glint.value:
            ax1.plot(wl, rgli,'o-C4', label=r"$T\rho_{g}$")
        if tau.value:
            ax1.plot([], [],':C1', label=r"$\tau_r$")
            ax1.plot([], [],':C3', label=r"$\tau_a$")
        
        plt.xlim(400,900)
        plt.ylim(bottom=0)
        plt.legend()
        
        if tau.value:    
            ax2 = ax1.twinx()
            ax2.plot(wl, taur,':C1')
            ax2.plot(wl, taua,':C3')
            ax2.set_ylabel(r"$\tau$")
            
        plt.show()