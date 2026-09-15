"""Core commands: pred, calc, traj, zmat, fdf, sample, add, addall, gp."""
import os
import subprocess
import pickle
import numpy as np
from os import getcwd, chdir, mkdir
from os.path import exists

from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF,DotProduct, WhiteKernel,
                                              ConstantKernel as C,RationalQuadratic,
                                              Matern,
                                              ExpSineSquared)
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
# from ase import build
from ase.io import read,write
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from irff.md.gulp import opt,get_reax_energy,write_gulp_in
from mlpkit.utils import (read_individuals, search_structure,generate_hbond_lib,
                            write_input,run_gulp, # add_structure,
                            lammps_opt_mtp,
                            write_output,write_geometry)
# from irff.md.lammps import writeLammpsData,writeLammpsIn,get_lammps_thermal,lammpstraj_to_ase
from irff.md.gulp import write_gulp_in,get_reax_energy ,opt
# from irff.dft.dftb import dftb_opt
from irff.dft.siesta import siesta_opt
from irff.molecule import Molecules,enlarge, SuperCell # moltoatoms


''' A work flow in combination with USPEX 
    High-Throughput Evolutionary Crystal Structure Prediction Method
'''

def supercell(gen=None,traj=None,x=1,y=1,z=1):
    if traj is None:
        A = read(gen)
        # build.make_supercell(A,[2,2,2])
        _,atoms = SuperCell(A,fac=1.0,supercell=[x,y,z])
        write(f'POSCAR.supercell_{x}_{y}_{z}',atoms)
    else:
        images = Trajectory(traj)
        # A = images[-1]
        # build.make_supercell(A,[2,2,2])
        his    = TrajectoryWriter(f'{traj.split(".")[0]}_{x}{y}{z}.traj',mode='w')
        for i,A in enumerate(images):
            # _,atoms = SuperCell(A,fac=1.0,supercell=[x,y,z])
            atoms   = A*(x,y,z)
            atoms.calc = SinglePointCalculator(atoms,energy=A.get_potential_energy()*x*y*z)
            his.write(atoms=atoms)
        his.close()


def addall(traj='structures.traj',step=1000,tolerance=0.005,ncpu=1):
    images = Trajectory(traj)
    for atoms_dft in images:
        add(atoms_dft,step=step,tolerance=tolerance,ncpu=ncpu)


def add(atoms_dft=None,traj='structures.traj',step=1000,tolerance=0.005,i=-1,ncpu=1):
    if atoms_dft is None:
       atoms_dft = read(traj,i)
    masses  = np.sum(atoms_dft.get_masses())
    volume  = atoms_dft.get_volume()
    density = masses/volume/0.602214129
    energy  = atoms_dft.get_potential_energy()

    atoms = opt(atoms=atoms_dft,step=step,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')              ## compute feature
    e     = get_feature(atoms,n=ncpu,lib='reaxff_nn')
    e_cho = get_hbond_feature(atoms,n=ncpu,elements='H core C core O core')
    e_chn = get_hbond_feature(atoms,n=ncpu,elements='H core C core N core')
    e_chc = get_hbond_feature(atoms,n=ncpu,elements='H core C core C core')

    volume   = atoms.get_volume()
    density_ = masses/volume/0.602214129
    
    data = np.loadtxt('feature_mlp.csv',delimiter=',',skiprows=1)      ## get crystal feature data
    data_= np.loadtxt('feature.csv',delimiter=',',skiprows=1)          ## get crystal feature data
    d    = data[:,1:]         # 去掉索引
    i    = int(data[-1][0])+1   # 获取索引
    # print(cry)
    feature = np.array([e[0],e[1],e[5],e[8],e[10],e_cho[11],e_chn[11],e_chc[11],e[12],density_])
    res  = np.sum(np.square(d - feature),axis=1)
    ind  = np.where(res<tolerance)
    
    if len(ind[0])>0:
       print(f'Structure already in database with index {ind[0]}!') 
       print(f'energy: {d[ind[0],0]}')
    else:
       with open('feature_mlp.csv','a') as fd:
            print(i,',',feature[0],',',feature[1],',',feature[2],',',feature[3],',',feature[4],',',
                     feature[5],',',feature[6],',',feature[7],',',
                     feature[8],',',feature[9],
                     file=fd) 
       with open('feature.csv','a') as fd:
            print(i,',',energy,',',feature[1],',',feature[2],',',feature[3],',',feature[4],',',
                  feature[5],',',feature[6],',',feature[7],',',
                  feature[8],',',density,file=fd)  
    
       atoms.calc = SinglePointCalculator(atoms,energy=e[0])
       with TrajectoryWriter('structures_mlp.traj',mode='a') as traj:
            traj.write(atoms=atoms)
       with TrajectoryWriter('structures.traj',mode='a') as traj:
            traj.write(atoms=atoms_dft)


def get_feature(atoms,n=1,lib='reaxff_nn'):
    write_gulp_in(atoms,runword='gradient nosymmetry conv qite verb',lib=lib)
    if n==1:
       subprocess.call('gulp<inp-gulp>out',shell=True)
    else:
       subprocess.call('mpirun -n {:d} gulp<inp-gulp>out'.format(n),shell=True)
    e = get_reax_energy(fo='out')
    return e


def get_hbond_feature(atoms,n=1,elements='H core C core O core'):
    lib = generate_hbond_lib(elements)
    e = get_feature(atoms,n=n,lib=lib)
    return e


def gp(tolerance=0.005,step=1000,n=1,b=1.5,u=0.04,f=1,dat='data',dft=0,den=1.82,pop=100,ref='results1'):
    ''' Gaussian Process '''
    write_input(inp='inp-grad',keyword='grad conv qiterative verb')
    run_gulp(n=n,inp='inp-grad')
    e = get_reax_energy(fo='output')
    write_output(e=e[0])

    atoms  = read('gulp.cif')
    # atoms  = opt(atoms=atoms,step=step,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')
    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density = masses/volume/0.602214129
    atoms.calc = SinglePointCalculator(atoms,energy=e[0])

    e_cho = get_hbond_feature(atoms,n=n,elements='H core C core O core')
    e_chn = get_hbond_feature(atoms,n=n,elements='H core C core N core')
    e_chc = get_hbond_feature(atoms,n=n,elements='H core C core C core')

    if f==1:
       # feature = np.array([e[0],e[1],e[5],e[8],e[10],e[11],e[12],density])
       feature = np.array([e[0],e[1],e[5],e[8],e[10],e_cho[11],e_chn[11],e_chc[11],e[12],density])
    else:
       feature = np.array([e[0],e[5],e[8],e[10],e[11],e[12],density])
 
    data   = np.loadtxt('../{:s}/feature_mlp.csv'.format(dat),delimiter=',',skiprows=1)  ## get crystal feature data
    data_  = np.loadtxt('../{:s}/feature.csv'.format(dat),delimiter=',',skiprows=1)      ## get crystal feature data
    images = Trajectory('../{:s}/structures.traj'.format(dat))
    d      = data[:,1:]    # 去掉索引

    # Train a Gaussian Process 
    res    = np.sum(np.square(d - feature),axis=1)
    ind    = np.where(res<tolerance)
    imin   = np.argmin(res)

    ### prepare data 
    X_raw  = data[:,1:]
    y      = data_[:,-1]
    y_eng  = data_[:,1]

    d_scaler= np.mean(y)/np.mean(data[:,-1])
    e_mean = np.mean(data[:,1])
    e_scaler= e_mean - np.mean(y_eng)

    scaler = preprocessing.StandardScaler().fit(X_raw)
    X      = scaler.transform(X_raw)
 
    gpr_energy, gpr_density = load_gaussian_process(X, y, y_eng)
    rfr_density = load_rfr(X, y)
    if not exists('gp.csv'):
        with open('gp.csv','w') as fd:
             print(',   index,          residual,        density_min,         density_rf,   density_gp,'
                '          uncertainty,           energy_min,       eng_pred,        uncertainty_eng',file=fd)

    # X_ = np.concatenate((X,np.expand_dims(feature,axis=0)))  #X_train.extend(feature)
    X_ = scaler.transform(np.expand_dims(feature,axis=0))
    mean_prediction, std_prediction = gpr_density.predict(X_, return_std=True)
    mean_eng_pred, std_eng_pred = gpr_energy.predict(X_, return_std=True)
    density_rf = rfr_density.predict(X_)
    # print('95% confidence interval: \n', 1.96 * std_prediction)
    
    density_= mean_prediction[0] # data_[ind[0][im],-1]
    # if ((density_>np.max(y)*1.1 and (density_/density>1.5 or  density/density_>1.5)) or 
    #     (density_>np.max(y) and res[imin]>10) ):
    if res[imin]>10:
       if density_rf[0]/density>1.5 or  density/density_rf[0]>1.5:
          density_ = density*d_scaler
       else:
          density_ = density_rf[0]

    # 用 gp.csv 行数获取下一个晶体 ID（替代解析整个 Individuals 文件）
    try:
        with open('gp.csv') as f:
            id_ = sum(1 for _ in f)
    except FileNotFoundError:
        id_ = 1
           
    if dft:
       data_pred = np.loadtxt('gp.csv',delimiter=',',skiprows=1)      ## get crystal feature data
       if data_pred.size > 0:
          if  data_pred.ndim==2:
              if data_pred.shape[0]>pop:
                 U         = data_pred[:,6]
                 R         = data_pred[:,2]
                 Density   = np.where(np.logical_and(U<0.2,R<10.0),
                                      data_pred[:,5],data_pred[:,4])
                
                 imax      = np.argmax(Density)
                 # with open('gp.log','a') as fg:
                 #     print(data_pred.ndim,file=fg)
                 #     print(data_pred.shape[0],file=fg)
                 #     print(density_,0.9*Density[imax],file=fg)
                 if density_ >= 0.98*Density[imax] and density_ >= den:
                    if std_prediction[0]>u and res[imin]< 5.0:
                       subprocess.call("cp ../Specific/*.psf ./", shell=True)
                       img = siesta_opt(atoms, ncpu=n, us="F", VariableCell="true", tstep=step,
                                             xcf="GGA", xca="PBE", basistype="split")
                       subprocess.call(f"mv siesta.traj id_{id_}.traj", shell=True)
                       subprocess.call("rm siesta.* ", shell=True)
                       subprocess.call("rm *.xml ", shell=True)
                       subprocess.call("rm INPUT_TMP.* ", shell=True)
                       subprocess.call("rm fdf-* ", shell=True)
                       # img[0].write(f"POSCAR.{s}")
                       atoms_opt = img[-1]
                       # atoms_opt.write(f"POSCAR.{s}_opt")
                       masses = np.sum(atoms_opt.get_masses())
                       volume = atoms_opt.get_volume()
                       density_ = masses / volume / 0.602214129
                       energy = atoms_opt.get_potential_energy()
                       # with open('refit','w') as fr:
                       #      print(1,file=fr)
                       subprocess.call("rm gpr_density.pkl gpr_energy.pkl", shell=True) 
                       subprocess.call("rm rfr.pkl", shell=True) 
                              
                       # add structure to database
                       # chdir(data_dir)
                       with open(f"../{dat}/feature_mlp.csv", "a") as fd:
                            print(f"{id_},{feature[0]},{feature[1]},{feature[2]},{feature[3]},"
                                  f"{feature[4]},{feature[5]},{feature[6]},{feature[7]},{feature[8]},{feature[9]}", file=fd)
                       with open(f"../{dat}/feature.csv", "a") as fd:
                            print(f"{id_},{energy},{feature[1]},{feature[2]},"
                                  f"{feature[3]},{feature[4]},{feature[5]},{feature[6]},{feature[7]},{feature[8]},{density_}",
                                  file=fd)

                       # atoms_opt.calc = SinglePointCalculator(atoms_opt, energy=energy)
                       with TrajectoryWriter(f"../{dat}/structures_mlp.traj", mode="a") as traj_w:
                            traj_w.write(atoms=atoms)
                       with TrajectoryWriter(f"../{dat}/structures.traj", mode="a") as traj_w:
                            traj_w.write(atoms=atoms_opt)
                          
    with open('gp.csv','a') as fd:
        # id_ = fd.tell()
        print(id_,',',imin,',',res[imin],',',data_[imin][-1],',',
            density_rf[0],',',mean_prediction[0],',',
            1.96*std_prediction[0],',',data_[imin][1],',',mean_eng_pred[0],',',1.96*std_eng_pred[0],
            file=fd)
        
    energy  = -density_ # mean_eng_pred[0]
    write_output(e=energy)
    write_geometry(atoms=atoms)
    

def calcdata(traj='structures.traj',n=8,c='nn',step=1000):
    ''' c: calculator, which mathine learning potential to be used '''
    images      = Trajectory(traj)
    traj_       = TrajectoryWriter('structures_mlp.traj',mode='w')

    with open('feature_mlp.csv','w') as fd:
        print(', etot, ebond, eang, etor, evdw, ehb_cho,ehb_chn,ehb_chc, ecoul, density',file=fd)
    with open('feature.csv','w') as fd_:
        print(', etot, ebond, eang, etor, evdw, ehb_cho, ehb_chn,ehb_chc, ecoul, density',file=fd_)

    for i,atoms in enumerate(images):
        masses = np.sum(atoms.get_masses())
        volume = atoms.get_volume()
        density_ = masses/volume/0.602214129
        energy = atoms.get_potential_energy()
        if c=='nn':
           atoms = opt(atoms=atoms,step=step,l=1,t=0.000001,n=n, lib='reaxff_nn')
        elif c=='mtp':
           atoms = lammps_opt_mtp(atoms=atoms,step=step,n=n,lib='pot.almtp')
        else:
           raise RuntimeError("Caluclator not supported!") 
        e     = get_feature(atoms,n=n,lib='reaxff_nn')
        e_cho = get_hbond_feature(atoms,n=n,elements='H core C core O core')
        e_chn = get_hbond_feature(atoms,n=n,elements='H core C core N core')
        e_chc = get_hbond_feature(atoms,n=n,elements='H core C core C core')

        # atoms = read('gulp.cif')
        atoms.calc = SinglePointCalculator(atoms,energy=e[0])
        traj_.write(atoms=atoms)
        # e,ebond,elp,eover,eunder,eang,epen,tconj,etor,fconj,evdw,ehb,ecl,esl
        
        volume = atoms.get_volume()
        density = masses/volume/0.602214129
        #  print(e)
        print('ID {:4d}: etol {:8.4f} ebond: {:8.4f} eang: {:8.4f} etor: {:8.4f} evdw: {:8.4f} '
            'ehb: {:8.4f}  {:8.4f} {:8.4f} {:8.4f} ' 
            'ecoul: {:8.4f} density: {:9.6}'.format(i,e[0],e[1],e[5],e[8],e[10],
                                        e[11],e_cho[11],e_chn[11],e_chc[11],
                                        e[12],density))
        with open('feature_mlp.csv','a') as fd:
            print(i,',',e[0],',',e[1],',',e[5],',',e[8],',',e[10],',',e_cho[11],',',e_chn[11],',',e_chc[11],',',
                e[12],',',density,file=fd) 
        with open('feature.csv','a') as fd_:
            print(i,',',energy,',',e[1],',',e[5],',',e[8],',',e[10],',',e_cho[11],',',e_chn[11],',',e_chc[11],',',
                e[12],',',density_,file=fd_) 
            
    traj_.close()

# ──────────────────────────────────────────────
#  fix broken molecule
# ──────────────────────────────────────────────
def fixbroken(broken=1.5,dat='data',scale=1.2,ncpu=1):
    write_input(inp='inp-grad',keyword='grad conv qiterative verb')
    run_gulp(n=ncpu,inp='inp-grad')
    e = get_reax_energy(fo='output')
    write_output(e=e[0])

    atoms  = read('gulp.cif')
    # atoms  = opt(atoms=atoms,step=step,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')
    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density = masses/volume/0.602214129
    atoms.calc = SinglePointCalculator(atoms,energy=e[0])
    # feature = np.array([e[0],e[1],e[5],e[8],e[10],e[11],e[12],density])

    data   = np.loadtxt('../{:s}/feature_mlp.csv'.format(dat),delimiter=',',skiprows=1)  ## get crystal feature data
    data_  = np.loadtxt('../{:s}/feature.csv'.format(dat),delimiter=',',skiprows=1)      ## get crystal feature data
    images = Trajectory('../{:s}/structures.traj'.format(dat))
    d      = data[:,1:]    # 去掉索引

    ### prepare data 
    X_raw  = data[:,1:]
    y      = data_[:,-1]
    y_eng  = data_[:,1]
    d_scale= np.mean(y)/np.mean(data[:,-1])
    e_mean = np.mean(data[:,1])
    e_scale= e_mean - np.mean(y_eng)

    if e_mean-e[0]>broken:
       if exists("molecule.pkl"):
          with open("molecule.pkl", "rb") as f:
               m_ = pickle.load(f)
          for m in m_:
              for i,na in enumerate(m.mol_index):
                  m.mol_x[i] = atoms.positions[na]

          for m in m_:
              m.center       = np.sum(m.mol_x,axis=0)/m.natom
        
          nmol    = len(m_)
          cell    = atoms.get_cell()
          irun = 0
          fac  = 1.0
          while e_mean-e[0]>broken and irun < 4:
                fac = fac*scale
                _,atoms = enlarge(m_,cell=cell,fac=fac,supercell=[1,1,1])
                atoms,e,density = get_gulp_energy(atoms, ncpu=ncpu,o=False)
                irun += 1
    else:
       if not exists("molecule.pkl"):
          m_  = Molecules(atoms,rcut={"H-H":1.0,"H-O":1.02,"O-O":1.4,"H-N":1.22,"H-C":1.35,
                                "others": 1.75},check=True)
          with open("molecule.pkl", "wb") as f:
               pickle.dump(m_, f)
    
    write_output(e=e[0])
    write_geometry(atoms=atoms)

# ──────────────────────────────────────────────
#  GULP energy helper
# ──────────────────────────────────────────────

def get_gulp_energy(atoms, ncpu=8,o=True):
    if o:
       atoms_opt = opt(atoms=atoms, step=1000, l=1, t=0.000001, n=ncpu, lib="reaxff_nn")
    else:
       atoms_opt = atoms
    write_gulp_in(atoms_opt, runword="gradient nosymmetry conv qite verb", lib="reaxff_nn")
    if ncpu == 1:
        subprocess.call("gulp<inp-gulp>out", shell=True)
    else:
        subprocess.call(f"mpirun -n {ncpu:d} gulp<inp-gulp>out", shell=True)
    e = get_reax_energy(fo="out")
    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density = masses / volume / 0.602214129
    return atoms_opt, e, density

# ──────────────────────────────────────────────
#  ML model helpers
# ──────────────────────────────────────────────

def load_gaussian_process(X, y, y_eng):
    length_scale = [0.1 for i in range(X.shape[1])]
    # if exists("refit"):
    #    if exists("gpr_energy.pkl"): 
    #       subprocess.call("rm gpr_energy.pkl", shell=True) 
    #    if exists("gpr_density.pkl"): 
    #       subprocess.call("rm gpr_density.pkl", shell=True) 
    #    if exists("rfr.pkl"): 
    #       subprocess.call("rm rfr.pkl", shell=True)
    #    subprocess.call("rm refit", shell=True)
    if not exists("gpr_density.pkl"):
        kernel = (
            0.00581**2 * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50))
            + 0.35**2 * Matern(length_scale=length_scale, nu=2.5)
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-8, 1e-1))
        )
        gpr_density = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        gpr_density.fit(X, y)
        with open("gpr_density.pkl", "wb") as f:
            pickle.dump(gpr_density, f)
    else:
        with open("gpr_density.pkl", "rb") as f:
            gpr_density = pickle.load(f)

    if not exists("gpr_energy.pkl"):
        kernel = (
            0.00581**2 * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50))
            + 0.35**2 * Matern(length_scale=length_scale, nu=2.5)
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-8, 1e-1))
        )
        gpr_energy = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
        gpr_energy.fit(X, y_eng)
        with open("gpr_energy.pkl", "wb") as f:
            pickle.dump(gpr_energy, f)
    else:
        with open("gpr_energy.pkl", "rb") as f:
            gpr_energy = pickle.load(f)

    with open("gpcsp.log", "w") as fl:
        print(gpr_density.kernel_, file=fl)
        print(gpr_density.log_marginal_likelihood(), file=fl)
        print(gpr_energy.kernel_, file=fl)
        print(gpr_energy.log_marginal_likelihood(), file=fl)

    return gpr_energy, gpr_density

# def load_mlp(X, y):
#     mlp = MLPRegressor((16, 8), max_iter=20000)
#     mlp.fit(X, y)
#     return mlp

def load_rfr(X, y):
    if not exists("rfr.pkl"):
       rfr = RandomForestRegressor(random_state=37, n_estimators=300,
                                min_weight_fraction_leaf=0.0, oob_score=True)
       rfr.fit(X, y)
       with open("rfr.pkl", "wb") as f:
            pickle.dump(rfr, f)
    else:
       with open("rfr.pkl", "rb") as f:
            rfr = pickle.load(f)
    return rfr

# ──────────────────────────────────────────────
#  pred — 高斯过程预测
# ──────────────────────────────────────────────

def pred(t="Individuals.traj", g=None, f=1, den=1.88, ids=None,x=-1,
         c='nn',step=300, ncpu=8, dat="data", tolerance=0.001):
    """
    Predict density and energy using Gaussian Process + MLP + RandomForest.

    Args:
        t: trajectory file name
        g: generation number (None = latest)
        f: feature flag (1 = 8D feature, else 7D)
        den: density threshold
        ids: comma/space separated crystal indices
        step: optimization steps
        ncpu: number of CPUs
        dat: data directory name
        tolerance: structure matching tolerance
    """
    if g is not None:
       atoms  = read(g,index=x)
       images = [atoms]
    else:
       images = Trajectory(t)
    
    if g is not None:
       ids_list = [1]
    elif not ids:
        ids_list = []
        res = read_individuals()  # g
        for i, e, d, _f in res:
            if d > den and _f < 0.0:
                ids_list.append(i)
    else:
        ids_list = [int(i) for i in ids.split()]

    root_dir = getcwd()
    if not exists("density_predict.log"):
        with open("density_predict.log", "w") as fd:
            print("# Crystal_id Residual Density_mlp Density_rf Density_gp std_den Energy std_eng", file=fd)

    masses = np.sum(images[0].get_masses())
    for s in ids_list:
        dir_list = root_dir.split("/")
        rootdir = "/".join(dir_list[:-1])
        data_dir = f"{rootdir}/{dat}"
        # print(images,s)
        atoms = images[s - 1]

        chdir(data_dir)
        if c=='nn':
           # atoms_mlp, e, density = get_gulp_energy(atoms, ncpu=ncpu)
           atoms_mlp = opt(atoms=atoms,step=step,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')
        elif c=='mtp':
           # atoms_mlp, e, density = get_gulp_energy(atoms, ncpu=ncpu)
           atoms_mlp = lammps_opt_mtp(atoms=atoms,step=step,n=ncpu,lib='pot.almtp')
        else:
           raise RuntimeError("Caluclator not supported!") 

        volume  = atoms_mlp.get_volume()
        density = masses / volume / 0.602214129
        e       = get_feature(atoms_mlp,n=ncpu,lib='reaxff_nn')
        e_cho   = get_hbond_feature(atoms_mlp,n=ncpu,elements='H core C core O core')
        e_chn   = get_hbond_feature(atoms_mlp,n=ncpu,elements='H core C core N core')
        e_chc   = get_hbond_feature(atoms_mlp,n=ncpu,elements='H core C core C core')
        
        # if f == 1:
        feature = np.array([e[0],e[1],e[5],e[8],e[10],e_cho[11],e_chn[11],e_chc[11],e[12],density])
        # else:
        #    feature = np.array([e[0],e[1],e[5], e[8], e[10], e[11], e[12], density])

        assert exists("structures.traj"), "Error, datafile not found in data directory!"
        data  = np.loadtxt("feature_mlp.csv", delimiter=",", skiprows=1)
        data_ = np.loadtxt("feature.csv", delimiter=",", skiprows=1)
        struc = Trajectory('structures.traj')

        D = data[:, 1:]
        D_ = data_[:, 1:]
        ind, imin, res_ = search_structure(feature, D, tolerance=tolerance)

        X_raw = data[:, 1:]
        y = data_[:, -1]
        y_eng = data_[:, 1]
        scaler = preprocessing.StandardScaler().fit(X_raw)
        X = scaler.transform(X_raw)

        gpr_energy, gpr_density = load_gaussian_process(X, y, y_eng)
        rfr = load_rfr(X, y)
        # mlp = load_mlp(X, y)

        X_ = scaler.transform(np.expand_dims(feature, axis=0))
        density_pred, std_den_pred = gpr_density.predict(X_, return_std=True)
        energy_pred, std_eng_pred = gpr_energy.predict(X_, return_std=True)
        energy_pred  = energy_pred[0]
        density_pred = density_pred[0]
        std_den_pred = std_den_pred[0]
        std_eng_pred = std_eng_pred[0]
        density_rf   = rfr.predict(X_)[0]
        # density_mlp  = mlp.predict(X_)[0]

        # if f == 1:
        print(f"{s:5d} res: {res_}  "
              f"rf: {density_rf:7.4f}  "
              f"gp(den): {density_pred:7.4f} uncert: {std_den_pred:7.4f}  "
              f"gp(eng): {energy_pred:7.4f} uncert: {std_eng_pred:7.4f}" )
        # else:
        #     print(f"{s:5d} rf: {density_rf:9.4f} "
        #           f"{feature[3]:9.4f} {feature[4]:9.4f} {feature[5]:9.4f} {feature[6]:9.4f} "
        #           f"gp: {density_pred:7.4f} uncert: {std_den_pred:7.4f}")

        chdir(root_dir)
        with open("density_predict.log", "a") as fd:
             print(f"{s:5d} {res_} "
                   f"{density:7.4f} "
                   f"{density_rf:7.4f} "
                   f"{density_pred:7.4f} {std_den_pred:7.4f} "
                   f"{energy_pred:7.4f} {std_eng_pred:7.4f} " ,file=fd)

# ──────────────────────────────────────────────
#  calc — DFT 高通量计算
# ──────────────────────────────────────────────

def calc(t="Individuals.traj", den=1.88, ids=None, step=500,
         ncpu=8, dat="data", tolerance=0.01):
    """
    High-throughput DFT calculation with structure matching.

    Args:
        t: trajectory file name
        den: density threshold
        ids: comma/space separated crystal indices
        step: MD steps
        ncpu: number of CPUs
        dat: data directory name
        tolerance: structure matching tolerance
    """
    images = Trajectory(t)
    if not ids:
        ids_list = []
        res = read_individuals()
        for i, e, d, _f in res:
            if d > den and _f < 0.0:
                ids_list.append(i)
    else:
        ids_list = [int(i) for i in ids.split()]

    root_dir = getcwd()
    if not exists("density.log"):
        with open("density.log", "w") as fd:
            print("# Crystal_id Density Energy", file=fd)

    for s in ids_list:
        dir_list = root_dir.split("/")
        rootdir = "/".join(dir_list[:-1])
        data_dir = f"{rootdir}/{dat}"
        work_dir = os.path.join(root_dir, str(s))
        atoms = images[s - 1]

        if exists(str(s)):
            continue
        else:
            mkdir(str(s))

        chdir(data_dir)
        atoms_mlp, e, density = get_gulp_energy(atoms, ncpu=ncpu)
        e_cho = get_hbond_feature(atoms_mlp,n=ncpu,elements='H core C core O core')
        e_chn = get_hbond_feature(atoms_mlp,n=ncpu,elements='H core C core N core')
        e_chc = get_hbond_feature(atoms_mlp,n=ncpu,elements='H core C core C core')

        feature = np.array([e[0], e[1], e[5], e[8], e[10], e_cho[11], e_chn[11], e_chc[11],e[12], density])

        if exists("structures.traj"):
            data = np.loadtxt("feature_mlp.csv", delimiter=",", skiprows=1)
            data_ = np.loadtxt("feature.csv", delimiter=",", skiprows=1)
            struc = Trajectory("structures.traj")
            try:
                D = data[:, 1:]
                D_ = data_[:, 1:]
            except IndexError:
                D = data[1:]
                D_ = data_[1:]
            ind, imin, res_ = search_structure(feature, D, tolerance=tolerance)
        else:
            ind = [[]]
            with open("feature_mlp.csv", "w") as fd:
                print(", etot, ebond, eang, etor, evdw, ehb_cho, ehb_chn, ehb_chc, ecoul, density", file=fd)
            with open("feature.csv", "w") as fd_:
                print(", etot, ebond, eang, etor, evdw, ehb_cho, ehb_chn, ehb_chc, ecoul, density", file=fd_)
            masses = np.sum(atoms.get_masses())
            volume = atoms.get_volume()
            density = masses / volume / 0.602214129
            res_ = 0.0

        chdir(work_dir)
        if len(ind[0]) > 0:
            atoms.write(f"POSCAR.{s}")
            struc[imin].write(f"POSCAR.{s}_opt")
            if D.ndim == 2:
                energy = D_[imin, 0]
                density = D_[imin, -1]
            else:
                energy = D_[0]
                density = D_[-1]
            print(f"{s:5d} mt {energy:9.4f} {feature[1]:9.4f} {feature[2]:9.4f} "
                  f"{feature[3]:9.4f} {feature[4]:9.4f} {feature[5]:9.4f} "
                  f"{feature[6]:9.4f} {density:7.4f} {res_:7.4f}")
            traj_w = TrajectoryWriter(f"id_{s}.traj", mode="w")
            traj_w.write(atoms=struc[imin])
            traj_w.close()
        else:
            subprocess.call(f"cp {rootdir}/Specific/*.psf ./", shell=True)
            img = siesta_opt(atoms, ncpu=ncpu, us="F", VariableCell="true", tstep=step,
                             xcf="GGA", xca="PBE", basistype="split")
            subprocess.call(f"mv siesta.out siesta-{s}.out", shell=True)
            subprocess.call(f"mv siesta.MDE siesta-{s}.MDE", shell=True)
            subprocess.call(f"mv siesta.MD_CAR siesta-{s}.MD_CAR", shell=True)
            subprocess.call(f"mv siesta.traj id_{s}.traj", shell=True)
            subprocess.call("rm siesta.* ", shell=True)
            subprocess.call("rm *.xml ", shell=True)
            subprocess.call("rm INPUT_TMP.* ", shell=True)
            subprocess.call("rm fdf-* ", shell=True)
            img[0].write(f"POSCAR.{s}")
            atoms_opt = img[-1]
            atoms_opt.write(f"POSCAR.{s}_opt")
            masses = np.sum(atoms_opt.get_masses())
            volume = atoms_opt.get_volume()
            density = masses / volume / 0.602214129
            energy = atoms_opt.get_potential_energy()

            print(f"{s:5d} cl {energy:9.4f} {feature[1]:9.4f} {feature[2]:9.4f} "
                  f"{feature[3]:9.4f} {feature[4]:9.4f} {feature[5]:9.4f} "
                  f"{feature[6]:9.4f} {density:7.4f} {res_:7.4f}")

            chdir(data_dir)
            with open("feature_mlp.csv", "a") as fd:
                print(f"0,{feature[0]},{feature[1]},{feature[2]},{feature[3]},"
                      f"{feature[4]},{feature[5]},{feature[6]},{feature[7]},{feature[8]},{feature[9]}", file=fd)
            with open("feature.csv", "a") as fd:
                print(f"0,{energy},{feature[1]},{feature[2]},"
                      f"{feature[3]},{feature[4]},{feature[5]},{feature[6]},{feature[7]},{feature[8]},{density}", file=fd)

            atoms_opt.calc = SinglePointCalculator(atoms_opt, energy=energy)
            with TrajectoryWriter("structures_mlp.traj", mode="a") as traj_w:
                traj_w.write(atoms=atoms_mlp)
            with TrajectoryWriter("structures.traj", mode="a") as traj_w:
                traj_w.write(atoms=atoms_opt)

        chdir(root_dir)
        with open("density.log", "a") as fd:
            print(f"{s:5d} {density:10.6f} {energy:10.8f}", file=fd)

# ──────────────────────────────────────────────
#  update structure
# ──────────────────────────────────────────────

def update(traj,inde=None,step=1000,tolerance=0.005,ncpu=1):
    atoms_dft = read(traj,-1)
    subprocess.call('cp structures.traj structures.backup.traj',shell=True)
    subprocess.call('cp feature.csv feature.backup.csv',shell=True)
    subprocess.call('cp feature_mlp.csv feature_mlp.backup.csv',shell=True)
    masses  = np.sum(atoms_dft.get_masses())
    volume  = atoms_dft.get_volume()
    density = masses/volume/0.602214129
    energy  = atoms_dft.get_potential_energy()

    atoms = opt(atoms=atoms_dft,step=step,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')              ## compute feature
    e     = get_feature(atoms,n=ncpu,lib='reaxff_nn')
    e_cho = get_hbond_feature(atoms,n=ncpu,elements='H core C core O core')
    e_chn = get_hbond_feature(atoms,n=ncpu,elements='H core C core N core')
    e_chc = get_hbond_feature(atoms,n=ncpu,elements='H core C core C core')
    
    volume   = atoms_dft.get_volume()
    density_ = masses/volume/0.602214129
    
    data = np.loadtxt('feature_mlp.csv',delimiter=',',skiprows=1)      ## get crystal feature data
    data_= np.loadtxt('feature.csv',delimiter=',',skiprows=1)          ## get crystal feature data
    d    = data[:,1:]         # 去掉索引
    d_   = data_[:,1:]   
     
    images     = Trajectory('structures.backup.traj')
    # images_  = Trajectory('structures_mlp.traj')
    feature = np.array([e[0], e[1], e[5], e[8], e[10], e_cho[11], e_chn[11], e_chc[11],e[12], density])

    res  = np.sum(np.square(d - feature),axis=1)
    ind  = np.where(res<tolerance)
    if inde is None:
       ind_ = ind[0]
    else:
       ind_ = [inde]

    if len(ind_)>0:
       his  = TrajectoryWriter('structures.traj',mode='w')
       fd    = open('feature.csv','w') 
       print(', etot, eang, etor, evdw, ehb, ecoul, density',file=fd)
       for i,d in enumerate(d_):
           if i in ind_:
              print(i,data_[i][1],energy,data_[i][7],density)
              print(i,',',energy,',',d[1],',',d[2],',',d[3],',',d[4],',',d[5],',',d[4],',',d[7],',',d[8],',',density,file=fd)  
              his.write(atoms=atoms_dft)
           else:  
              print(i,',',d[0],',',d[1],',',d[2],',',d[3],',',d[4],',',d[5],',',d[6],',',d[7],',',d[8],',',d[9],file=fd)  
              his.write(atoms=images[i])
       fd.close()
       his.close()
    else:
       print(f'Specified structure not found in database!') 
       print(f'energy: {energy}')
    

# ──────────────────────────────────────────────
#  traj — POSCAR to trajectory
# ──────────────────────────────────────────────

def traj(fposcar="gatheredPOSCARS"):
    """Convert gatheredPOSCARS to ASE trajectory file."""
    from mlpkit.utils import Stack

    with open(fposcar) as fbp:
        lines = fbp.readlines()

    traj_w = TrajectoryWriter("Individuals.traj", mode="w")
    k = 0
    s = 0
    energies = []

    with open("Individuals") as f:
        for line in f.readlines():
            st = Stack([])
            for x in line:
                if x != "]":
                    st.push(x)
                else:
                    x_ = " "
                    while x_ != "[":
                        x_ = st.pop()
            line = "".join(st.entry)
            l = line.split()
            if len(l) >= 10 and l[0] != "Gen":
                energies.append(float(l[3]))
        st.close()

    for line in lines:
        if "EA" in line:
            if k > 0:
                fpos.close()
                atoms = read("POSCAR")
                atoms.calc = SinglePointCalculator(atoms, energy=energies[s])
                traj_w.write(atoms=atoms)
                s += 1
            fpos = open("POSCAR", "w")
            print(line[:-1], file=fpos)
            k += 1
        else:
            print(line[:-1], file=fpos)

    fpos.close()
    atoms = read("POSCAR")
    atoms.calc = SinglePointCalculator(atoms, energy=energies[s])
    traj_w.write(atoms=atoms)
    traj_w.close()

# ──────────────────────────────────────────────
#  zmat — 内坐标
# ──────────────────────────────────────────────

def zmat(geo="POSCAR", i=-1):
    """Convert structure to USPEX Z-matrix format."""
    from irff.AtomDance import AtomDance
    atoms = read(geo, index=i)
    ad = AtomDance(atoms=atoms, rcut={"H-O": 2.7, "O-H": 2.7})
    zmat_data = ad.InitZmat
    ad.write_zmat(zmat_data, uspex=True)
    ad.close()

# ──────────────────────────────────────────────
#  fdf — 写 SIESTA 输入
# ──────────────────────────────────────────────

def fdf(gen="poscar.gen", xcf="gga", i=-1):
    """Generate SIESTA input files."""
    from irff.dft.siesta import write_siesta_in
    A = read(gen, index=i)
    print("\n-  writing siesta input ...")
    if xcf == "gga":
        write_siesta_in(A, coord="cart", md=False, opt="CG",
                        VariableCell="true", xcf="GGA", xca="PBE", basistype="split")
    elif xcf == "vdw":
        write_siesta_in(A, coord="cart", md=False, opt="CG",
                        VariableCell="true", xcf="VDW", xca="DRSLL", basistype="split")
    else:
        print("Not supported yet!")

# ──────────────────────────────────────────────
#  info - 打印能量与晶格常数信息
# ──────────────────────────────────────────────

def info(gen=None, traj=None, i=-1, symmetry=True, symprec=0.1):
    """Print energy, lattice, and symmetry information of a structure.

    Args:
        gen: geometry file name (e.g. POSCAR, gulp.cif)
        traj: trajectory file name
        i: frame index (default: -1, last frame)
        symmetry: if True, perform pymatgen symmetry analysis (default: True)
        symprec: symmetry tolerance for pymatgen (default: 0.1)
    """
    if gen is not None:
        atoms = read(gen)
    elif traj is not None:
        images = Trajectory(traj)
        atoms = images[i]
    else:
        print("Error: please specify a structure file (--gen) or trajectory (--traj)")
        return

    cell = atoms.get_cell()
    lengths = cell.lengths()
    angles = cell.angles()
    volume = atoms.get_volume()
    masses = np.sum(atoms.get_masses())
    density = masses / volume / 0.602214129
    natoms = len(atoms)
    formula = atoms.get_chemical_formula()

    print(f"\n─ Structure Info ─────────────────────────────")
    print(f"  Formula:          {formula}")
    print(f"  Number of atoms:  {natoms}")
    print(f"\n─ Lattice ────────────────────────────────────")
    print(f"  a = {lengths[0]:12.6f} Å")
    print(f"  b = {lengths[1]:12.6f} Å")
    print(f"  c = {lengths[2]:12.6f} Å")
    print(f"  α = {angles[0]:12.6f}°")
    print(f"  β = {angles[1]:12.6f}°")
    print(f"  γ = {angles[2]:12.6f}°")
    print(f"  Volume  = {volume:12.4f} ų")
    print(f"  Density = {density:12.6f} g/cm³")

    try:
        energy = atoms.get_potential_energy()
        print(f"\n─ Energy ─────────────────────────────────────")
        print(f"  Total energy = {energy:16.8f} eV")
        if natoms > 0:
            print(f"  Energy/atom  = {energy / natoms:16.8f} eV/atom")
    except Exception:
        print(f"\n─ Energy ─────────────────────────────────────")
        print("  (no energy data available)")

    # ── Symmetry Analysis (pymatgen) ──
    if symmetry:
        _print_symmetry_info(atoms, symprec)

    print(f"──────────────────────────────────────────────\n")


def _print_symmetry_info(atoms, symprec):
    """Print symmetry analysis using pymatgen SpacegroupAnalyzer.

    Args:
        atoms: ASE Atoms object
        symprec: symmetry tolerance for pymatgen
    """
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError:
        print(f"\n─ Symmetry ───────────────────────────────────")
        print("  (pymatgen not available — install pymatgen)")
        return

    try:
        # Convert ASE Atoms → pymatgen Structure directly (no temp file)
        struct = Structure.from_ase_atoms(atoms)

        analyzer = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=5.0)

        spg_symbol = analyzer.get_space_group_symbol()
        spg_number = analyzer.get_space_group_number()
        crystal_system = analyzer.get_crystal_system()
        point_group = analyzer.get_point_group_symbol()
        hall = analyzer.get_hall()
        lattice_type = analyzer.get_lattice_type()
        has_inversion = analyzer.is_laue()
        num_sym_ops = len(analyzer.get_symmetry_operations())
        sym_struct = analyzer.get_symmetrized_structure()

        print(f"\n─ Symmetry (pymatgen, symprec={symprec}) ──────")
        print(f"  Space Group (HM):    {spg_symbol}")
        print(f"  Space Group Number:  {spg_number}")
        print(f"  Hall Symbol:         {hall}")
        print(f"  Crystal System:      {crystal_system}")
        print(f"  Point Group:         {point_group}")
        print(f"  Lattice Type:        {lattice_type}")
        print(f"  Laue (inversion):    {'yes' if has_inversion else 'no'}")
        print(f"  Symmetry Operations: {num_sym_ops}")
        print(f"  Wyckoff Sites:       {len(sym_struct.wyckoff_symbols)}")

    except Exception as e:
        print(f"\n─ Symmetry ───────────────────────────────────")
        print(f"  (analysis failed: {e})")


# ──────────────────────────────────────────────
#  fingerprint - 分子指纹计算 (Cython 加速)
# ──────────────────────────────────────────────

def _atoms_to_fingerprint_input(atoms):
    """从 ASE Atoms 提取指纹计算所需的 numpy 数组。

    将原子按元素种类分组，返回 lattice, coords, numIons, atomType。
    """
    from collections import OrderedDict

    cell = atoms.get_cell()
    lattice = np.ascontiguousarray(np.array(cell, dtype=np.float64))
    frac = atoms.get_scaled_positions(wrap=False)
    coords = np.ascontiguousarray(np.array(frac, dtype=np.float64))

    symbols = atoms.get_chemical_symbols()
    # 按 POSCAR 顺序分组：同元素连续排列
    seen = OrderedDict()
    for s in symbols:
        if s not in seen:
            seen[s] = 0
        seen[s] += 1
    numIons = np.ascontiguousarray(np.array(list(seen.values()), dtype=np.int32))

    # 元素序号 (原子序数)
    _symbol_to_z = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
        'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    }
    atomType = np.ascontiguousarray(
        np.array([_symbol_to_z.get(s, 0) for s in seen.keys()], dtype=np.int32))
    return lattice, coords, numIons, atomType


def fingerprint(gen=None, traj=None, i=-1,
                rmax=12.0, sigma=0.05, delta=0.08, dimension=3,
                output=None, intra_map=None, soap=False,
                soap_r_cut=6.0, soap_n_max=8, soap_l_max=6):
    """Compute the structural fingerprint of a molecular crystal.

    By default computes the USPEX RDF fingerprint via the Cython-accelerated
    ``uspex_fast_core`` module (makeMatrices + fingerprint_calc).

    When *soap* is True, additionally computes a SOAP (Smooth Overlap of
    Atomic Positions) fingerprint via ``dscribe``.  SOAP captures local
    angular environment and is far more discriminating for molecular
    crystal polymorphs than the pure radial RDF.

    Args:
        gen: geometry structure file (e.g. POSCAR).  Read with ASE.
        traj: trajectory file name (alternative to gen).
        i: frame index for trajectory (default: -1).
        rmax: cutoff radius for neighbour search (Å).
        sigma: Gaussian broadening for distance bins.
        delta: bin width (Å).
        dimension: 3 for 3D crystals, 0 for cluster, 2 for 2D.
        output: output file name (.npz).  If None, print summary only.
        intra_map: optional (N, N) int8 array.  Entries of 0 mark
            intra-molecular atom pairs within the basic cell whose
            distances should be zeroed (matching Octave
            ``Intra_MOL_dist``).  Only zero-shift (basic-cell) pairs
            are filtered; periodic-image pairs are always kept.
        soap: if True, also compute SOAP fingerprint.
        soap_r_cut: SOAP local-environment cutoff (Å, default 6.0).
        soap_n_max: SOAP radial basis count (default 8).
        soap_l_max: SOAP angular momentum maximum (default 6).
    """
    from mlpkit.uspex_fast_core import build_distance_matrix, fingerprint_calc

    if gen is not None:
        atoms = read(gen)
    elif traj is not None:
        images = Trajectory(traj)
        atoms = images[i]
    else:
        print("Error: please specify a structure file (--gen) or trajectory (--traj)")
        return

    lattice, coords, numIons, atomType = _atoms_to_fingerprint_input(atoms)

    n_species = len(numIons)
    n_atoms = int(np.sum(numIons))
    volume = abs(np.linalg.det(lattice))
    masses = np.sum(atoms.get_masses())
    density = masses / volume / 0.602214129

    print(f"\n─ Fingerprint ────────────────────────────────")
    print(f"  Structure:    {gen or traj}")
    print(f"  Atoms:        {n_atoms}  ({n_species} species)")
    symbols = atoms.get_chemical_symbols()
    from collections import OrderedDict
    seen = OrderedDict()
    for s in symbols:
        if s not in seen:
            seen[s] = 0
        seen[s] += 1
    species_str = "  ".join(f"{s}:{c}" for s, c in seen.items())
    print(f"  Composition:  {species_str}")
    print(f"  Volume:       {volume:.4f} ų")
    print(f"  Density:      {density:.6f} g/cm³")
    print(f"  Parameters:   Rmax={rmax}  σ={sigma}  δ={delta}  dim={dimension}")
    if intra_map is not None:
        print(f"  Intra_map:    {intra_map.shape}  (filtering zero-shift pairs)")

    import time as _time
    t0 = _time.time()

    dist_arr, cc_idx, bc_idx, ti_arr, tj_arr, shift_arr, N_out, V, N = build_distance_matrix(
        coords, lattice, rmax, numIons, atomType
    )
    t1 = _time.time()

    # Apply Intra_map filtering (matching Octave ReadJobs_310.m behaviour):
    # Only zero-shift (basic-cell) pairs are filtered.  Periodic-image
    # pairs are always kept, even if the atoms belong to the same molecule.
    if intra_map is not None:
        intra_map = np.asarray(intra_map, dtype=np.int8)
        zero_shift_mask = (shift_arr == 0)
        intra_mask = (intra_map[cc_idx, bc_idx] == 1)
        keep_mask = np.where(zero_shift_mask, intra_mask, True)
        dist_arr = dist_arr[keep_mask]
        cc_idx = cc_idx[keep_mask]
        bc_idx = bc_idx[keep_mask]
        ti_arr = ti_arr[keep_mask]
        tj_arr = tj_arr[keep_mask]

    order, fing, atom_fing, _ = fingerprint_calc(
        dist_arr, cc_idx, bc_idx, ti_arr, tj_arr, N_out, V, numIons,
        rmax, sigma, delta, dimension
    )
    t2 = _time.time()

    t_matrix = t1 - t0
    t_fp = t2 - t1
    t_total = t2 - t0
    n_pairs = len(dist_arr)

    print(f"\n─ Results ────────────────────────────────────")
    print(f"  Pairs:        {n_pairs}")
    print(f"  order:        shape={order.shape}  "
          f"min={order.min():.6e}  max={order.max():.6e}  "
          f"mean={order.mean():.6e}")
    print(f"  fing:         shape={fing.shape}  "
          f"min={fing.min():.6e}  max={fing.max():.6e}")
    print(f"  atom_fing:    shape={atom_fing.shape}  "
          f"min={atom_fing.min():.6e}  max={atom_fing.max():.6e}")
    print(f"  Time:         matrix={t_matrix:.4f}s  "
          f"fingerprint={t_fp:.4f}s  total={t_total:.4f}s")

    result = {
        'order': order,
        'fing': fing,
        'atom_fing': atom_fing,
        'V': V,
        'n_pairs': n_pairs,
        'time_matrix': t_matrix,
        'time_fingerprint': t_fp,
        'time_total': t_total,
    }

    # ── optional SOAP fingerprint ──
    if soap:
        from mlpkit.soap import soap_fingerprint as _soap_fp
        t_s0 = _time.time()
        soap_fp = _soap_fp(atoms, r_cut=soap_r_cut, n_max=soap_n_max,
                           l_max=soap_l_max)
        t_s1 = _time.time()
        print(f"\n─ SOAP ───────────────────────────────────────")
        print(f"  Parameters:   r_cut={soap_r_cut}  n_max={soap_n_max}  "
              f"l_max={soap_l_max}")
        print(f"  soap_fp:      shape={soap_fp.shape}  "
              f"min={soap_fp.min():.6e}  max={soap_fp.max():.6e}")
        print(f"  nonzero:      {np.count_nonzero(soap_fp)}/{soap_fp.size} "
              f"({np.count_nonzero(soap_fp)/soap_fp.size:.0%})")
        print(f"  Time:         {t_s1-t_s0:.4f}s")
        result['soap_fp'] = soap_fp
        result['time_soap'] = t_s1 - t_s0

    if output:
        save_dict = dict(order=order, fing=fing, atom_fing=atom_fing,
                         V=V, n_pairs=n_pairs,
                         numIons=numIons, atomType=atomType,
                         rmax=rmax, sigma=sigma, delta=delta,
                         dimension=dimension)
        if soap and 'soap_fp' in result:
            save_dict['soap_fp'] = result['soap_fp']
        np.savez(output, **save_dict)
        print(f"\n  Saved to: {output}")

    print(f"──────────────────────────────────────────────\n")
    return result


# ──────────────────────────────────────────────
#  sample - 采样结构
# ──────────────────────────────────────────────

def sample(ind="", t=None, s=None, e=None, i=1, o=None, f=""):
    """Sample structures by index or range to a trajectory file.

    Three modes (in priority order):
      1. --f: explicit frame indices (space-separated)
      2. --ind: legacy index mode (space-separated)
      3. --s/--e: range mode (start to end, with interval)
    """
    out = o or "samples.traj"
    traj_w = TrajectoryWriter(out, mode="w")

    # ── Mode 1: explicit frames ──
    if f:
        images = Trajectory(t or "md.traj")
        frames = [int(x) for x in f.split()]
        for fi in frames:
            traj_w.write(atoms=images[fi])
        traj_w.close()
        return

    # ── Mode 2: legacy index mode ──
    if ind:
        ids = [int(x) for x in ind.split()]
        if t is not None:
            images = Trajectory(t)
            for fi in ids:
                traj_w.write(atoms=images[fi])
        else:
            for fi in ids:
                atoms = read(f"{fi}/POSCAR.{fi}_opt")
                atoms.calc = SinglePointCalculator(atoms, energy=0.0)
                traj_w.write(atoms=atoms)
        traj_w.close()
        return

    # ── Mode 3: range mode ──
    if s is not None:
        images = Trajectory(t or "md.traj")
        end = e + 1 if e is not None else len(images)
        for fi in range(s, end):
            if fi % i == 0:
                traj_w.write(atoms=images[fi])
        traj_w.close()
        return

    traj_w.close()


# ──────────────────────────────────────────────
#  lib — 将 ffield.json 转换为 reaxff_nn.lib
# ──────────────────────────────────────────────

def lib(jsonfile="ffield.json", libfile="reaxff_nn.lib"):
    """
    Convert ffield.json to reaxff_nn.lib format.

    Args:
        jsonfile: path to ffield.json (default: ffield.json)
        libfile:  output lib file name (default: reaxff_nn.lib)
    """
    import json as js
    from irff.reaxfflib import write_lib

    with open(jsonfile, "r") as lf:
        j = js.load(lf)
    p_ = j["p"]
    m_ = j["m"]
    vdw_layer = j.get("vdw_layer", None)
    mf_layer = j.get("mf_layer", None)
    be_layer = j.get("be_layer", None)

    spec, bonds, offd, angs, torp, hbs = _init_bonds(p_)
    write_lib(p_, spec, bonds, offd, angs, torp, hbs,
              m=m_, mf_layer=mf_layer, be_layer=be_layer,
              vdw_layer=vdw_layer, libfile=libfile)
    print(f"✓  Converted {jsonfile} → {libfile}")


def _init_bonds(p_):
    """Categorize force-field parameters from JSON keys."""
    spec, bonds, offd, angs, torp, hbs = [], [], [], [], [], []
    for key in p_:
        k = key.split("_")
        if k[0] == "bo1":
            bonds.append(k[1])
        elif k[0] == "rosi":
            kk = k[1].split("-")
            if len(kk) == 2:
                if kk[0] != kk[1]:
                    offd.append(k[1])
            elif len(kk) == 1:
                spec.append(k[1])
        elif k[0] == "theta0":
            angs.append(k[1])
        elif k[0] == "tor1":
            torp.append(k[1])
        elif k[0] == "rohb":
            hbs.append(k[1])
    return spec, bonds, offd, angs, torp, hbs


# ──────────────────────────────────────────────
#  ffield — 将 ffield.json 转换为 ffield
# ──────────────────────────────────────────────

def ffield(jsonfile="ffield.json", ffieldfile="ffield"):
    """
    Convert ffield.json to ReaxFF ffield format.

    Args:
        jsonfile:   path to ffield.json (default: ffield.json)
        ffieldfile: output ffield file name (default: ffield)
    """
    import json as js
    from irff.reaxfflib import write_ffield

    with open(jsonfile, "r") as lf:
        j = js.load(lf)
    p_ = j["p"]
    m_ = j["m"]
    mf_layer = j.get("mf_layer", None)
    be_layer = j.get("be_layer", None)

    spec, bonds, offd, angs, torp, hbs = _init_bonds(p_)
    write_ffield(p_, spec, bonds, offd, angs, torp, hbs,
                 m=m_, mf_layer=mf_layer, be_layer=be_layer,
                 libfile=ffieldfile)
    print(f"✓  Converted {jsonfile} → {ffieldfile}")


# ──────────────────────────────────────────────
#  molinfo — 分子识别与原子分组信息
# ──────────────────────────────────────────────

def molinfo(gen="data.traj", i=-1, jsonfile=None, verbose=True):
    """
    Print molecule atom indices for each molecule in the structure.

    Uses the IRFF Molecules module to identify molecules based on bond
    cutoffs from ffield.json (or defaults for CHON systems).

    Args:
        gen:      geometry file (ASE-readable: traj, POSCAR, xyz, lammps-data, etc.)
        i:        frame index in trajectory (default: -1 = last frame)
        jsonfile: path to ffield.json for rcutBond parameters (optional)
        verbose:  print molecule boundary info (default: True)
    """
    import json as js
    from ase.io import read
    from irff.molecule import Molecules, press_mol

    # ── Load structure ──
    if gen.endswith('.lammps') or gen.endswith('.lammps-data'):
        atoms = read(gen, format='lammps-data', atom_style='charge')
    else:
        atoms = read(gen, index=i)

    atoms = press_mol(atoms)

    # ── Build rcut dict from ffield.json or use defaults ──
    rcut = {}
    if jsonfile is not None:
        try:
            with open(jsonfile, 'r') as lf:
                j = js.load(lf)
            rcut = j.get('rcutBond', {})
            if verbose:
                print(f"# Loaded bond cutoffs from {jsonfile}")
        except FileNotFoundError:
            if verbose:
                print(f"# Warning: {jsonfile} not found, using defaults")
        except Exception as e:
            if verbose:
                print(f"# Warning: failed to read {jsonfile}: {e}, using defaults")

    if not rcut:
        # Default rcut for CHON systems (ReaxFF-style)
        rcut = {
            "H-O": 1.2, "H-C": 1.2, "H-H": 0.8, "H-N": 1.2,
            "O-N": 1.45, "O-C": 1.45, "O-O": 1.4,
            "N-C": 1.68, "N-N": 1.68,
            "C-C": 1.68,
            "others": 1.8,
        }
        if verbose:
            print("# Using default bond cutoffs for CHON systems")

    if verbose:
        print(f"# rcut: {rcut}")
        print(f"# Total atoms: {len(atoms)}")
        print(f"# Species: {sorted(set(atoms.get_chemical_symbols()))}")

    # ── Identify molecules ──
    mols = Molecules(atoms, rcut=rcut, check=True)
    nmol = len(mols)

    if verbose:
        print(f"# Number of molecules: {nmol}")

    # ── Print molecule atom indices (1-based for LAMMPS compatibility) ──
    for i_mol, mol in enumerate(mols):
        idx = np.array(mol.mol_index) + 1  # 1-based
        nat = len(idx)
        if verbose:
            print(f"# Mol {i_mol+1}: {nat} atoms")
        print(' '.join(str(x) for x in idx))

    return mols

# ──────────────────────────────────────────────
#  critical — 多信号综合评判提取 MD 失稳帧
# ──────────────────────────────────────────────

# 共价键参考长度 (C-H-N-O 含能材料体系)
_COVALENT_BONDS = {
    ('C','C'): (1.54, 2.00), ('C','H'): (1.09, 1.50),
    ('C','N'): (1.47, 1.90), ('C','O'): (1.43, 1.85),
    ('N','N'): (1.45, 1.90), ('N','O'): (1.40, 1.85),
    ('O','O'): (1.48, 1.90), ('H','N'): (1.01, 1.45),
    ('H','O'): (0.97, 1.40), ('H','H'): (0.74, 1.10),
}

# LAMMPS real → ASE 单位转换因子
_REAL_FORCE_TO_ASE = 0.043364   # kcal/mol/Å → eV/Å
_REAL_ENERGY_TO_ASE = 0.043364  # kcal/mol → eV


def _force_stats(atoms):
    """力的统计量: max, mean, std, highF%原子占比."""
    if 'forces' not in atoms.arrays:
        return None
    fn = np.linalg.norm(atoms.arrays['forces'], axis=1)
    return {
        'maxF': float(np.max(fn)),
        'meanF': float(np.mean(fn)),
        'stdF': float(np.std(fn)),
        'pct_highF': float(np.mean(fn > 1.5 * np.mean(fn))) * 100,
    }


def _bond_stats(atoms, stretch_factor=1.15):
    """键异常统计: (stretched, broken, collapsed, close_pairs).

    截断用 ref*1.3 只检查真正的共价键候选, max_ok 判断裂, min_ok (=ref*0.65) 判塌缩/新键.
    """
    from ase.geometry import get_distances
    natoms = len(atoms)
    D, D_len = get_distances(atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
    stretched, broken, collapsed, close = 0, 0, 0, 0
    for i in range(natoms):
        row = D_len[i]
        si = atoms.symbols[i]
        for j in range(i + 1, natoms):
            sj = atoms.symbols[j]
            key = (si, sj) if si <= sj else (sj, si)
            if key not in _COVALENT_BONDS:
                continue
            ref, max_ok = _COVALENT_BONDS[key]
            min_ok = ref * 0.65  # 范德华接触下限, 低于此为新键/碰撞
            d = row[j]
            if d > ref * 1.3:
                continue
            close += 1
            if d > max_ok:
                broken += 1
            elif d < min_ok:
                collapsed += 1
            elif d > ref * stretch_factor:
                stretched += 1
    return stretched, broken, collapsed, close


def _disp_stats(atoms, prev):
    """帧间位移统计 (近似速度): max_disp, rmsd."""
    if prev is None:
        return None
    disp = atoms.positions - prev.positions
    norms = np.linalg.norm(disp, axis=1)
    return {
        'max_disp': float(np.max(norms)),
        'rmsd': float(np.sqrt(np.mean(np.sum(disp**2, axis=1)))),
    }


def _stability_score(fstats, bstats, dstats, baselines):
    """综合稳定性评分 (0=正常, 越高越不稳定).

    信号与权重:
      断键增量 (broken)  — 键断裂
      塌缩增量 (collapsed) — 新键形成/原子碰撞
      maxF Z-score — 力偏离基线
      highF% 增量  — 力分布变宽
      max_disp Z-score — 原子位移
      RMSD Z-score — 全局漂移确认
    """
    score, flags = 0.0, []

    fz = (fstats['maxF'] - baselines['maxF_mean']) / max(baselines['maxF_std'], 1e-6)
    if fz > 2.0:
        score += min(fz - 2.0, 8.0)
        flags.append(f'F{fz:.1f}')

    hp_delta = fstats['pct_highF'] - baselines['highF_mean']
    if hp_delta > baselines['highF_std'] * 2:
        score += min(hp_delta / 5, 6.0)
        flags.append(f'hF+{hp_delta:.0f}%')

    # 键异常 — 断裂 + 塌缩/新键, 必须有力异常配合
    bdelta = bstats[1] - baselines['broken_mean']
    cdelta = bstats[2] - baselines['collapsed_mean']
    if fz > 2.0:
        if bdelta >= 1.0:
            score += bdelta * 1.5
            flags.append(f'Br+{bdelta:.0f}')
        if cdelta >= 1.0:
            score += cdelta * 1.5
            flags.append(f'Col+{cdelta:.0f}')

    if dstats is not None and baselines.get('max_disp_mean', 0) > 0:
        dz = (dstats['max_disp'] - baselines['max_disp_mean']) / max(baselines['max_disp_std'], 1e-8)
        if dz > 3.0:
            score += min(dz - 3.0, 5.0)
            flags.append(f'D{dz:.1f}')

    if dstats is not None and baselines.get('rmsd_mean', 0) > 0:
        rz = (dstats['rmsd'] - baselines['rmsd_mean']) / max(baselines['rmsd_std'], 1e-8)
        if rz > 5.0:
            score += min(rz - 5.0, 3.0)
            flags.append(f'R{rz:.1f}')

    return score, flags


def _parse_lammps_dump(path, max_frames=None):
    """解析 LAMMPS dump 文件, 生成 Atoms (forces 转 eV/Å, triclinic cell)."""
    from ase import Atoms as AseAtoms
    elem_map = {1: 'C', 2: 'H', 3: 'N', 4: 'O'}

    with open(path) as f:
        lines = f.readlines()
    i, N, n = 0, len(lines), 0

    while i < N:
        if not lines[i].startswith('ITEM: TIMESTEP'):
            i += 1; continue
        step = int(lines[i + 1].strip()); i += 2
        natoms = int(lines[i + 1].strip()); i += 2
        # BOX BOUNDS
        tilt_labels = lines[i].split()[3:]
        bounds = []
        for _ in range(3):
            bounds.append([float(x) for x in lines[i + 1].split()]); i += 1
        i += 1  # blank
        cols = lines[i].split()[2:]; i += 1

        diagdisp = np.array([bounds[0][0], bounds[0][1],
                             bounds[1][0], bounds[1][1],
                             bounds[2][0], bounds[2][1]])
        if len(bounds[0]) > 2 and len(tilt_labels) >= 3:
            offdiag = np.array([b[2] for b in bounds])
            order = [tilt_labels.index(t) for t in ('xy', 'xz', 'yz')]
            offdiag = offdiag[order]
        else:
            offdiag = np.zeros(3)
        xlo, xhi, ylo, yhi, zlo, zhi = diagdisp
        xy, xz, yz = offdiag
        cell = np.array([[xhi - xlo - abs(xy) - abs(xz), 0, 0],
                         [xy, yhi - ylo - abs(yz), 0],
                         [xz, yz, zhi - zlo]])

        ix = cols.index('xu') if 'xu' in cols else cols.index('x')
        iy = cols.index('yu') if 'yu' in cols else cols.index('y')
        iz = cols.index('zu') if 'zu' in cols else cols.index('z')
        itype = cols.index('type') if 'type' in cols else None
        has_f = 'fx' in cols

        symbols, pos, forces = [], [], None
        for _ in range(natoms):
            p = lines[i].split(); i += 1
            symbols.append(elem_map.get(int(p[itype]), 'X') if itype else 'X')
            pos.append([float(p[ix]), float(p[iy]), float(p[iz])])
            if has_f:
                if forces is None:
                    forces = []
                forces.append([float(p[cols.index('fx')]),
                               float(p[cols.index('fy')]),
                               float(p[cols.index('fz')])])

        atoms = AseAtoms(symbols=symbols, positions=np.array(pos), cell=cell, pbc=[True]*3)
        if forces is not None:
            atoms.set_array('forces', np.array(forces) * _REAL_FORCE_TO_ASE)
        atoms.info['timestep'] = step
        yield atoms
        n += 1
        if max_frames and n >= max_frames:
            break


def critical(dump='meta_nvt.lammpstrj', log=None, output='critical.traj',
             score_threshold=3.0, crash_score=50.0, baseline_frames=10,
             one_per_run=True, min_persist=3):
    """多信号综合评判提取 MD 失稳帧 (力 + 键长 + 位移).

    用法:
      mlpkit critical --dump meta_nvt.lammpstrj --log lmp.log \\
              --threshold 3.0 --crash 30.0 -o critical.traj

    参数:
      dump:             LAMMPS dump 文件路径
      log:              LAMMPS log 文件 (读取 E_pair, 可选)
      output:           输出 trajectory 文件
      score_threshold:  评分超过此值标记为失稳帧
      crash_score:      评分超过此值视为崩溃, 丢弃该帧及后续
      baseline_frames:  用于计算基线的前 N 帧
      one_per_run:      只取第一个失稳帧
      min_persist:      需连续 N 帧超阈值才确认, 过滤瞬时波动 (默认3)
    """
    from ase.io import write as ase_write

    if not exists(dump):
        print(f'❌ dump 不存在: {dump}')
        return

    # ── 基线 ──
    base_fstats, base_bstats, base_dstats = [], [], []
    prev = None
    for atoms in _parse_lammps_dump(dump, baseline_frames):
        fs = _force_stats(atoms)
        if fs is None:
            continue
        base_fstats.append(fs)
        base_bstats.append(_bond_stats(atoms))
        ds = _disp_stats(atoms, prev)
        if ds is not None:
            base_dstats.append(ds)
        prev = atoms

    if not base_fstats:
        print('❌ 无法读取帧 (无 force 信息)')
        return

    baselines = {
        'maxF_mean': float(np.mean([f['maxF'] for f in base_fstats])),
        'maxF_std':  float(np.std([f['maxF'] for f in base_fstats])),
        'highF_mean': float(np.mean([f['pct_highF'] for f in base_fstats])),
        'highF_std':  float(np.std([f['pct_highF'] for f in base_fstats])),
        'broken_mean': float(np.mean([b[1] for b in base_bstats])),
        'collapsed_mean': float(np.mean([b[2] for b in base_bstats])),
    }
    if base_dstats:
        baselines['max_disp_mean'] = float(np.mean([d['max_disp'] for d in base_dstats]))
        baselines['max_disp_std']  = float(np.std([d['max_disp'] for d in base_dstats]))
        baselines['rmsd_mean'] = float(np.mean([d['rmsd'] for d in base_dstats]))
        baselines['rmsd_std']  = float(np.std([d['rmsd'] for d in base_dstats]))

    print(f'📊 基线 ({len(base_fstats)} 帧): '
          f'maxF={baselines["maxF_mean"]:.1f}±{baselines["maxF_std"]:.1f}, '
          f'断键均={baselines["broken_mean"]:.0f}')
    print(f'   异常阈值={score_threshold}, 崩溃阈值={crash_score}')

    # ── 扫描 ──
    frames, crashed, taken = [], False, False
    prev = None
    total, after_crash, n_anom = 0, 0, 0
    trig_count = 0
    candidate_atoms = None

    for atoms in _parse_lammps_dump(dump):
            total += 1
            fs = _force_stats(atoms)
            if fs is None:
                prev = atoms; continue
            bs = _bond_stats(atoms)
            ds = _disp_stats(atoms, prev)
            score, flags = _stability_score(fs, bs, ds, baselines)
            step = atoms.info.get('timestep', total)

            if not crashed and score > crash_score:
                crashed = True
                print(f'💥 崩溃: step {step} score={score:.1f} [{", ".join(flags)}]')
                trig_count = 0; candidate_atoms = None
            if crashed:
                after_crash += 1
                prev = atoms; continue

            is_anom = score > score_threshold
            if is_anom and one_per_run and taken:
                after_crash += 1
                prev = atoms; continue

            if is_anom:
                trig_count += 1
                if candidate_atoms is None:
                    candidate_atoms = atoms
            else:
                trig_count = 0
                candidate_atoms = None

            if trig_count >= min_persist and not taken:
                taken = True
                a = candidate_atoms.copy()
                fs0 = _force_stats(candidate_atoms)
                bs0 = _bond_stats(candidate_atoms)
                a.info['maxF'] = fs0['maxF'] if fs0 else 0
                a.info['meanF'] = fs0['meanF'] if fs0 else 0
                a.info['broken_bonds'] = bs0[1] if bs0 else 0
                a.info['collapsed'] = bs0[2] if bs0 else 0
                a.info['class'] = 'anomalous'
                a.info['score'] = score
                a.info['source'] = dump
                a.calc = SinglePointCalculator(
                    a, energy=a.info.get('energy', 0.0),
                    forces=a.arrays.get('forces', None))
                frames.append(a)
                n_anom += 1
                print(f'⚠️  失稳帧: step {candidate_atoms.info.get("timestep")} '
                      f'maxF={a.info["maxF"]:.1f} broken={a.info["broken_bonds"]} '
                      f'(连续 {trig_count} 帧, 当前 score={score:.1f})')

            prev = atoms

    if not frames:
        print('⚠️  没有提取到失稳帧!')
        return

    ase_write(output, frames)
    print(f'\n✅ {output}: {len(frames)} 帧 '
          f'(扫描 {total} 帧, 丢弃崩溃后 {after_crash} 帧)')

