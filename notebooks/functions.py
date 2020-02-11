import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import interact, interactive, fixed, interact_manual, AppLayout, GridspecLayout, Layout
import ipywidgets as widgets
from IPython.display import clear_output
from tqdm.notebook import tqdm
import pyOSOAA
from scipy.interpolate import interp1d, interpn


chl_range = np.logspace(-3,2,11)
sed_range = np.array([0,1,10,100,1000])

wl = np.linspace(405,700,20)
wl = np.append(wl, [750, 865, 900])

c = np.round(np.random.uniform(0,5),1); s = 0

rho_arr = np.load("rho_arr.npy")

def f(chl, sed):
    return interpn((chl_range, sed_range), rho_arr[:,:,0,0,:], (chl,sed))[0]

list_satellite = np.load('list_satellite.npy',allow_pickle='TRUE').item()
list_satellite[None] = None

list_data = np.load('list_data.npy',allow_pickle='TRUE').item()

def plt_f(chl, sed=0, satellite="czcs", data=None, compute=False, text=False):
    if data is not None:
        plt.plot(list_satellite["seawifs"],list_data[data], 'ko')
    if compute is True:
        if satellite is not None:
            rho = interp1d(wl, f(chl, sed), fill_value="extrapolate")(list_satellite[satellite])
            plt.plot(list_satellite[satellite], rho, 'ko')

    elif compute == "data":
        if satellite is not None:
            plt.plot(list_satellite[satellite], interp1d(wl, f(c, 0), fill_value="extrapolate")(list_satellite[satellite]), 'ko')

    plt.plot(wl, f(chl, sed))
    plt.xlabel(rf"$\lambda$ [nm]"), plt.ylim(bottom=-0.003, top=0.063), plt.ylabel(rf"$\rho$")
    plt.show()
    if text is True:
        rho = interp1d(wl, f(chl, sed), fill_value="extrapolate")(list_satellite[satellite])
        if satellite is not None:
            print("wl[nm]", end='\t')
            for i in range(list_satellite[satellite].size):
                print(f"   {list_satellite[satellite][i]}", end='\t')
            print("")
            print("rho ", end='\t')
            for i in range(list_satellite[satellite].size):
                print(f"{np.round(rho[i],4)}", end='\t')
    return None

# We define the list of satellites and kinds of atmospheres according to the Shettle and Fenn models
list_satellite = np.load('list_satellite.npy',allow_pickle='TRUE').item()
list_aerosols = {"Troposférico":1, "Urbano":2, "Maritimo":3, "Costero":4}
list_kind = ["TOA", "TOO", "Rayleigh", "Aerosol", "Glint", "Optical thickness"]

# Widgets to configure the coupled run
# Satellite
satellite=widgets.Dropdown(options=list_satellite.keys(), description="Satélite");
# Surface and ocean
chl = widgets.BoundedFloatText(value=5, min=0.001, max=100, description="Chla [mg/m3]");
wind = widgets.BoundedFloatText(value=0, min=0, max=30, description="W [m/s]");
# Atmosphere
aero = widgets.Dropdown(options=list_aerosols, value=3, description="Aerosol");
rh = widgets.BoundedIntText(value=80, min=0, max=100, description="RH [%]");
aot = widgets.BoundedFloatText(value=0.1, min=0, max=1, step=0.05, description="AOT");
# Geometry
view = widgets.BoundedIntText(value=0, min=0, max=45, description=r"$\theta [^\circ]$");
sun = widgets.BoundedIntText(value=30, min=30, max=65, description=r"$\theta_0[^\circ]$");
phi = widgets.BoundedIntText(value=90, min=0, max=270, description=r"$\phi[^\circ]$");
# Choose what to plot
toa = widgets.Checkbox(description="Mostrar TOA", value=True)
too = widgets.Checkbox(description="Mostrar TOO")
rayleigh = widgets.Checkbox(description="Mostrar Rayleigh")
aerosol = widgets.Checkbox(description="Mostrar aerosol")
glint = widgets.Checkbox(description="Mostrar glint")
tau = widgets.Checkbox(description="Optical thickness")
# Button to run
button=widgets.Button(description='Run')
# Database to save the runs. If we run the same configuration twice
# we don't do the simulation again. 
button.DataBase = {}
# Configuration in a grid
grid = GridspecLayout(5, 3)
grid[0, 0] = satellite; grid[1, 0] = chl; grid[1, 1] = wind;
grid[2, 0] = aero; grid[3, 0] = rh; grid[4, 0] = aot;
grid[2, 1] = view; grid[3, 1] = sun; grid[4, 1] = phi;
grid[0,1] = toa; grid[0,2] = too; grid[1,2] = rayleigh; grid[2,2] = aerosol; grid[3,2] = glint
grid[4,2] = tau;

# Radiative transfer code with the OSOAA to simulate the atmosphere-ocean system
def simulate(satellite="seawifs", sun=30, view=0, phi=90, level=1, chl=1, aero=3, 
             rh=80, aot=0.1, wind=0, ap=False, gettau=False):
    s = pyOSOAA.OSOAA(cleanup=True)
    # Ocean
    s.phyto.chl = chl
    s.sea.depth = 100
    s.sea.botalb = 0
    if chl == 0:
        s = pyOSOAA.osoaahelpers.ConfigureOcean(s, ocean_type="black")
    # Surface
    s.sea.wind = wind
    s.sea.surfalb = 0
    # Aerosol
    s.aer.SetModel(model=2, sfmodel=aero, rh=rh)
    s.aer.aotref = aot
    # Geometry
    s.view.phi = phi
    s.ang.thetas = sun
    view = view
    s.view.level = level
    if ap is True:
        s.ap.SetMot(0)
    # Run code for all bands
    rho = []
    tauaer = []
    tauray = []
    for wa in tqdm(list_satellite[satellite], leave=False):
        # If we want no molecular scattering
        if ap is True:
            s.ap.SetMot(0)
        s.wa = wa/1e3
        s.run()
        rho = np.append(rho, np.interp(view, s.outputs.vsvza.vza, s.outputs.vsvza.I))
        # This is an approximation to the optical thickness
        tauaer = np.append(tauaer, (s.outputs.profileatm.tau*np.mean(s.outputs.profileatm.mixaer))[-1])
        tauray = np.append(tauray, (s.outputs.profileatm.tau*np.mean(s.outputs.profileatm.mixray))[-1])
    # If we asked for the optical thicknes
    if gettau:
        return tauray, tauaer
    else:
        return list_satellite[satellite], rho

def click(b):
    # Create the dictionary to save the data
    data =  {}
    # Create the run identifier
    s = ""
    total = 1
    for i in grid.children:
        # Ignore bool values
        if type(i.value) is bool:
            total = total+1
            continue
        if type(i.value) is float:
            # We round all the floats to 3 decimal
            s = s+str(np.round(i.value, 3))+"-"
        else:
            s = s+str(i.value)+"-"
    s = s[:-1]
    
    # We get the wavelengths for the satellite
    wl = list_satellite[satellite.value]
    
    # If the identifier is in the DataBase, we recover it
    if s in b.DataBase:
        df = b.DataBase[s]
    # If not we run the simulation
    else:
        i = 1
        # TOA simulation
        print(f"Simulación {i} de {total}...", end="\r")
        wl, rtoa = simulate(satellite=satellite.value, level=1, 
                       chl=chl.value, aero=aero.value, rh=rh.value,
                       aot=aot.value, wind=wind.value,
                       sun=sun.value, view=view.value, phi=phi.value)
        i = i+1
        data["rtoa"] = rtoa
        # Rayleigh simulation
        print(f"Simulación {i} de {total}...", end="\r")
        wl, rray = simulate(satellite=satellite.value, level=1, 
                           chl=0, aero=3, rh=80,
                           aot=0, wind=0,
                           sun=sun.value, view=view.value, phi=phi.value)
        i = i+1
        data["rray"] = rray
        # Aerosol simulation
        print(f"Simulación {i} de {total}...", end="\r")
        wl, raer = simulate(satellite=satellite.value, level=1, 
                           chl=0, aero=aero.value, rh=rh.value,
                           aot=aot.value, wind=0, ap=True,
                           sun=sun.value, view=view.value, phi=phi.value)
        i = i+1
        data["raer"] = raer
        # TOO simulation
        print(f"Simulación {i} de {total}...", end="\r")
        wl, rtoo = simulate(satellite=satellite.value, level=3, 
                       chl=chl.value, aero=aero.value, rh=rh.value,
                       aot=aot.value, wind=0,
                       sun=sun.value, view=view.value, phi=phi.value)
        i = i+1
        data["rtoo"] = rtoo
        # Glint simulation
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
        data["rgli"] = rgli
        # Optical thicknes simulation
        print(f"Simulación {i} de {total}...", end="\r")
        taur, taua = simulate(satellite=satellite.value, level=1, 
                       chl=chl.value, aero=aero.value, rh=rh.value,
                       aot=aot.value, wind=wind.value,
                       sun=sun.value, view=view.value, phi=phi.value, gettau=True)
        i = i+1
        data["taur"] = taur
        data["taua"] = taua
        # We convert the dictionary to a dataframe and save it
        df = pd.DataFrame().from_dict(data)
        df.index = list_satellite[satellite.value]
        b.DataBase[s] = df
    
    # We clear the output to avoid multiple plots
    clear_output(wait=True)
    # We display the buttons again
    display(grid, button)
    fig, ax1 = plt.subplots()
    
    # Recover simulated data
    rtoa = df["rtoa"]
    rray = df["rray"]
    raer = df["raer"]
    rtoo = df["rtoo"]
    rgli = df["rgli"]
    taur = df["taur"]
    taua = df["taua"]
    
    # If we asked for some plots, we plot the ones we requested
    # Reflectance and optical thickness are in different axis.
    if total > 1:
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
    return None

def plot_multiple(plots, kind):
    fig, ax1 = plt.subplots()
    # We fix a minimun value for the maximun of the plot
    maximo = 0.
    if kind == "Optical thickness":
        plt.plot([],[],"o-k", label=r"$\tau_r$")
        plt.plot([],[],"o:k", label=r"$\tau_a$")
    for s in plots:
        satellite = s.split("-")[0]
        wl = list_satellite[satellite]
        df = button.DataBase[s]
        # Recover simulated data
        rtoa = df["rtoa"]
        rray = df["rray"]
        raer = df["raer"]
        rtoo = df["rtoo"]
        rgli = df["rgli"]
        taur = df["taur"]
        taua = df["taua"]

        # We plot the requested variable
        plt.xlabel(r"$\lambda$ [nm]")
        plt.ylabel(r"$\rho$")    
        if kind == "TOA":
            plt.plot(wl, rtoa,'o-', label=s)
            maximo = np.max([maximo, rtoa.max()])
        if kind == "Rayleigh":
            plt.plot(wl, rray,'o-', label=s)
            maximo = np.max([maximo, rray.max()])
        if kind == "Aerosol":
            plt.plot(wl, raer,'o-', label=s)
            maximo = np.max([maximo, raer.max()])
        if kind == "TOO":
            plt.plot(wl, rtoo,'o-', label=s)
            maximo = np.max([maximo, rtoo.max()])
        if kind == "Glint":
            plt.plot(wl, rgli,'o-', label=s)
            maximo = np.max([maximo, rgli.max()])
        if kind == "Optical thickness":
            p =plt.plot(wl, taur,'o-', label=s)
            color = p[-1].get_color()
            plt.plot(wl, taua,'o:', color=color)
            maximo = np.max([maximo, taur.max(), taua.max()])
            plt.ylabel(r"$\tau$")
        plt.legend()
        
    plt.title(kind)
    plt.xlim(400,900)
    plt.ylim(bottom=0, top = maximo+0.005)
    plt.show()
    return None
