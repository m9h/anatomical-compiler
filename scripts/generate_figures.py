#!/usr/bin/env python3
"""Generate all 8 benchmark figures from preprocessed data."""
import numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import jax, jax.numpy as jnp, equinox as eqx, optax, hgx, devograph, diffrax
from sklearn.decomposition import PCA
import networkx as nx
import time

D = Path("data/processed")
FIG = Path("figures"); FIG.mkdir(exist_ok=True)

inc = jnp.array(np.load(D/"incidence.npy"))
feat = jnp.array(np.load(D/"node_features_pca.npy"))
temp = np.load(D/"temporal_expression.npy")
pt = np.load(D/"pseudotime_centers.npy")
lf = np.load(D/"lineage_fractions.npy")
fp = np.load(D/"fate_probabilities.npy")
ml = np.load(D/"module_labels.npy")
eigvals = np.load(D/"eigenvalues.npy")
masks = jnp.array(np.load(D/"perturbation_masks.npy"))
effects = jnp.array(np.load(D/"perturbation_effects.npy"))
fates = jnp.array(np.load(D/"perturbation_fates.npy"))
with open(D/"gene_names.json") as f: genes = json.load(f)
with open(D/"tf_names.json") as f: tfs = json.load(f)
with open(D/"key_tf_indices.json") as f: kti = {k:int(v) for k,v in json.load(f).items()}
with open(D/"tf_gene_indices.json") as f: tgi = {k:int(v) for k,v in json.load(f).items()}

n_genes, n_edges = inc.shape
fd = feat.shape[1]
nc = int(ml.max()) + 1
hg = hgx.from_incidence(inc, node_features=feat)
key = jax.random.PRNGKey(42)
C8 = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628","#f781bf","#999999"]
t0_total = time.time()
print(f"Loaded: {n_genes} genes, {n_edges} regulons, {fd}-dim features")
print(f"Device: {jax.devices()}")

# ================================================================
# FIGURE 1: GRN Architecture + TF Centrality
# ================================================================
print("\nFig 1: GRN Architecture...")
t0 = time.time()
degrees = np.array(hg.node_degrees)
adj = np.array(hgx.clique_expansion(hg))
_, evecs = np.linalg.eigh(adj)
ec = np.abs(evecs[:,-1]); ec /= ec.max()+1e-8
bw = nx.betweenness_centrality(nx.from_numpy_array((adj>0).astype(float)), k=50)
ba = np.array([bw.get(i,0) for i in range(n_genes)])
ed = np.array(hg.edge_degrees)

fig, axes = plt.subplots(2,2,figsize=(14,12))
axes[0,0].hist(degrees, bins=50, color="#2c3e50", alpha=0.7)
for tf,idx in list(kti.items())[:4]:
    axes[0,0].axvline(x=degrees[idx], color="red", alpha=0.7, ls="--")
    axes[0,0].text(degrees[idx]+1, axes[0,0].get_ylim()[1]*0.85, tf, fontsize=8, rotation=45)
axes[0,0].set(xlabel="Degree", ylabel="Count", title=f"A: Degree Distribution ({n_genes} genes)")

x = np.arange(len(kti)); ti = [kti[t] for t in kti]
axes[0,1].bar(x-0.2,[degrees[i]/degrees.max() for i in ti],0.2,label="Degree",color="#3498db")
axes[0,1].bar(x,[ec[i] for i in ti],0.2,label="Eigenvector",color="#e74c3c")
axes[0,1].bar(x+0.2,[ba[i]/max(ba.max(),1e-8) for i in ti],0.2,label="Betweenness",color="#2ecc71")
axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(list(kti.keys()), rotation=45, ha="right")
axes[0,1].set(ylabel="Centrality", title="B: Key TF Centrality"); axes[0,1].legend(fontsize=8)

axes[1,0].imshow(np.array(inc[:200,:30]).T, aspect="auto", cmap="Blues")
axes[1,0].set(xlabel="Gene (first 200)", ylabel="Regulon", title=f"C: Incidence ({n_genes}x{n_edges})")

axes[1,1].hist(ed, bins=50, color="#8e44ad", alpha=0.7)
axes[1,1].set(xlabel="Regulon Size", ylabel="Count", title=f"D: Regulon Sizes (mean={ed.mean():.0f})")
fig.suptitle("Figure 1: GRN Architecture (Fleck et al. 2023)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(FIG/"figure_01_grn.png", dpi=150, bbox_inches="tight"); plt.close()
print(f"  done ({time.time()-t0:.1f}s)")

# ================================================================
# FIGURE 2: Module Detection (SheafDiffusion)
# ================================================================
print("Fig 2: Module Detection...")
t0 = time.time()
k1,k2 = jax.random.split(key)
nnz = int(jnp.sum(inc>0)); labels = jnp.array(ml)

class SM(eqx.Module):
    s: hgx.SheafDiffusion
    r: eqx.nn.Linear
    def __call__(self,hg): return jax.vmap(self.r)(self.s(hg))

mdl = SM(s=hgx.SheafDiffusion(num_steps=3,in_dim=fd,edge_stalk_dim=fd,num_incidences=nnz,key=k1),
         r=eqx.nn.Linear(fd,nc,key=k2))
opt=optax.adam(1e-3); ost=opt.init(eqx.filter(mdl,eqx.is_array))

@eqx.filter_jit
def stp(m,o,hg,lb):
    def lf(m,hg,lb): return -jnp.mean(jnp.sum(jax.nn.one_hot(lb,nc)*jax.nn.log_softmax(m(hg),-1),-1))
    l,g=eqx.filter_value_and_grad(lf)(m,hg,lb); u,no=opt.update(g,o,m); return eqx.apply_updates(m,u),no,l

losses=[]
for ep in range(200):
    mdl,ost,l=stp(mdl,ost,hg,labels); losses.append(float(l))
    if (ep+1)%50==0:
        p=jnp.argmax(mdl(hg),-1); acc=float(jnp.mean(p==labels))
        print(f"  Ep {ep+1}: loss={l:.4f} acc={acc:.1%}")

preds=np.array(jnp.argmax(mdl(hg),-1))
fig,axes=plt.subplots(1,2,figsize=(14,5))
axes[0].plot(losses,lw=1,color="#2c3e50"); axes[0].set_yscale("log")
axes[0].set(xlabel="Epoch",ylabel="Loss",title=f"A: SheafDiffusion ({fd}-dim, acc={acc:.1%})")
axes[0].grid(True,alpha=0.3)
ms2=sorted([(m,int((ml==m).sum())) for m in range(nc)],key=lambda x:-x[1])[:20]
ma=[float((preds[ml==m]==m).mean()) for m,_ in ms2]
mn=[f"{tfs[m] if m<len(tfs) else m}\n({s})" for m,s in ms2]
axes[1].bar(range(len(ma)),ma,color="steelblue")
axes[1].set_xticks(range(len(ma))); axes[1].set_xticklabels(mn,rotation=90,fontsize=6)
axes[1].set(ylabel="Accuracy",title="B: Per-Module (top 20)"); axes[1].set_ylim(0,1.1)
fig.suptitle("Figure 2: Module Detection",fontsize=14,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(FIG/"figure_02_modules.png",dpi=150,bbox_inches="tight"); plt.close()
print(f"  done ({time.time()-t0:.1f}s)")

# ================================================================
# FIGURE 3: Trajectory + Fate + PCA
# ================================================================
print("Fig 3: Trajectory...")
t0 = time.time()
fig,axes=plt.subplots(2,2,figsize=(14,12))
for i,(tf,idx) in enumerate(kti.items()):
    axes[0,0].plot(pt,temp[:,idx],"o-",color=C8[i],label=tf,lw=2,ms=4)
axes[0,0].set(xlabel="Pseudotime",ylabel="Expression",title="A: Key TF Expression")
axes[0,0].legend(fontsize=7,ncol=2); axes[0,0].grid(True,alpha=0.3)

axes[0,1].stackplot(pt,lf[:,0],lf[:,1],lf[:,2],labels=["Telencephalon","Early","NT"],
                    colors=["#e41a1c","#377eb8","#4daf4a"],alpha=0.7)
axes[0,1].set(xlabel="Pseudotime",ylabel="Fraction",title="B: Lineage"); axes[0,1].legend(fontsize=8); axes[0,1].set_ylim(0,1)

axes[1,0].plot(pt,fp[:,0],"o-",color="#e41a1c",label="DF (cortical)",lw=2)
axes[1,0].plot(pt,fp[:,1],"s-",color="#377eb8",label="VF (GE)",lw=2)
axes[1,0].plot(pt,fp[:,2],"^-",color="#4daf4a",label="MH (NT)",lw=2)
axes[1,0].set(xlabel="Pseudotime",ylabel="Fate Prob",title="C: CellRank Fates"); axes[1,0].legend(); axes[1,0].grid(True,alpha=0.3)

pc=PCA(2).fit_transform(np.array(feat))
axes[1,1].scatter(pc[:,0],pc[:,1],c=ml,cmap="tab20",s=3,alpha=0.5)
axes[1,1].set(xlabel="PC1",ylabel="PC2",title=f"D: Gene PCA ({fd}-dim)")
fig.suptitle("Figure 3: Developmental Trajectory",fontsize=14,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(FIG/"figure_03_trajectory.png",dpi=150,bbox_inches="tight"); plt.close()
print(f"  done ({time.time()-t0:.1f}s)")

# ================================================================
# FIGURE 4: PPCA Eigenspectrum
# ================================================================
print("Fig 4: Eigenspectrum...")
fig,axes=plt.subplots(1,2,figsize=(14,5))
axes[0].plot(range(1,201),eigvals[:200],"o-",ms=2,color="#2c3e50")
axes[0].axvline(x=26,color="red",ls="--",alpha=0.7,label="BIC=26")
axes[0].axvline(x=fd,color="blue",ls="--",alpha=0.7,label=f"Consensus={fd}")
axes[0].axvline(x=168,color="green",ls="--",alpha=0.7,label="AIC=168")
axes[0].set(xlabel="Component",ylabel="Eigenvalue",title="A: Scree"); axes[0].legend(); axes[0].grid(True,alpha=0.3)
cv=np.cumsum(eigvals)/eigvals.sum()
axes[1].plot(range(1,len(cv)+1),cv,color="#8e44ad",lw=2)
axes[1].axhline(y=0.9,color="red",ls=":",alpha=0.5,label="90%")
axes[1].axhline(y=0.95,color="orange",ls=":",alpha=0.5,label="95%")
axes[1].axvline(x=fd,color="blue",ls="--",alpha=0.7,label=f"k={fd} ({cv[fd-1]:.1%})")
axes[1].set(xlabel="Components",ylabel="Cum Var",title="B: Variance"); axes[1].legend(); axes[1].grid(True,alpha=0.3); axes[1].set_xlim(0,300)
fig.suptitle("Figure 4: PPCA Dimensionality (Minka/MELODIC)",fontsize=14,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(FIG/"figure_04_eigenspectrum.png",dpi=150,bbox_inches="tight"); plt.close()
print("  done")

# ================================================================
# FIGURE 5: Spectral Analysis
# ================================================================
print("Fig 5: Spectral...")
t0 = time.time()
lap=np.array(hgx.hypergraph_laplacian(hg,normalized=True))
ev2=np.linalg.eigvalsh(lap); nz=int((np.abs(ev2)<1e-6).sum())
fig,axes=plt.subplots(1,2,figsize=(14,5))
axes[0].hist(ev2[ev2>1e-6],bins=80,color="#2c3e50",alpha=0.7)
axes[0].set(xlabel="Eigenvalue",ylabel="Count",title=f"A: Laplacian (beta0={nz})")
axes[1].plot(sorted(ev2),"o",ms=2,color="#e74c3c"); axes[1].set(xlabel="Index",ylabel="Eigenvalue",title="B: Sorted"); axes[1].grid(True,alpha=0.3)
fig.suptitle("Figure 5: Spectral Analysis",fontsize=14,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(FIG/"figure_05_spectral.png",dpi=150,bbox_inches="tight"); plt.close()
print(f"  done ({time.time()-t0:.1f}s)")

# ================================================================
# FIGURE 6: Neural ODE + SDE
# ================================================================
print("Fig 6: ODE/SDE...")
t0 = time.time()
T,ng=temp.shape; times=jnp.array(pt)
snaps=[hgx.from_incidence(inc,node_features=jnp.array(temp[t]).reshape(-1,1)) for t in range(T)]
thg=devograph.from_snapshots(snaps,times=times)
k1,k2,k3,k4=jax.random.split(key,4)

ode=devograph.fit_neural_ode(thg,hgx.UniGCNConv(1,1,key=k1),key=k2,epochs=200,lr=1e-3)
pr=temp[0].reshape(-1,1); op=[temp[0]]
for t in range(T-1):
    pr=devograph.evolve(ode,hgx.from_incidence(inc,node_features=jnp.array(pr)),t0=float(times[t]),t1=float(times[t+1])).node_features
    op.append(np.array(pr).flatten())
op=np.array(op); mse=float(np.mean((op[1:]-temp[1:])**2))
print(f"  ODE MSE: {mse:.6f}")

sde=devograph.HypergraphNeuralSDE(conv=hgx.UniGCNConv(1,1,key=k3),num_nodes=ng,node_dim=1,sigma_init=0.1,dt=0.05,key=k4)
sopt=optax.adam(1e-3); sst=sopt.init(eqx.filter(sde,eqx.is_array))
@eqx.filter_jit
def ss_step(sde,st,hg,tgt,sk):
    def lf(s): return jnp.mean((s(hg,t0=0.,t1=0.1,key=sk).ys[-1].reshape(ng,1)-tgt)**2)
    l,g=eqx.filter_value_and_grad(lf)(sde); u,ns=sopt.update(g,st,sde); return eqx.apply_updates(sde,u),ns,l
for ep in range(200):
    ti=ep%(T-1); sde,sst,l=ss_step(sde,sst,snaps[ti],jnp.array(temp[ti+1]).reshape(-1,1),jax.random.fold_in(k4,ep))
    if (ep+1)%50==0: print(f"  SDE ep {ep+1}: loss={l:.6f}")

ts3=jnp.linspace(0,0.1*(T-1),T)
straj=[np.array(sde(snaps[0],t0=0.,t1=float(ts3[-1]),key=jax.random.fold_in(k4,1000+i),saveat=diffrax.SaveAt(ts=ts3)).ys) for i in range(20)]
sig=np.exp(np.array(sde.diffusion.log_sigma))
print(f"  Sigma: [{sig.min():.4f},{sig.max():.4f}]")

fig,axes=plt.subplots(2,2,figsize=(14,12))
for i,(tf,idx) in enumerate(list(kti.items())[:4]):
    axes[0,0].plot(pt,temp[:,idx],"o-",color=C8[i],label=f"{tf} obs",lw=2)
    axes[0,0].plot(pt,op[:,idx],"s--",color=C8[i],label=f"{tf} pred",alpha=0.6)
axes[0,0].set(xlabel="Pseudotime",ylabel="Expr",title=f"A: Neural ODE (MSE={mse:.4f})"); axes[0,0].legend(fontsize=6,ncol=2); axes[0,0].grid(True,alpha=0.3)
for tr in straj: axes[0,1].plot(np.array(ts3),[float(np.mean(np.abs(tr[t]))) for t in range(tr.shape[0])],alpha=0.2,color="#3498db",lw=1)
axes[0,1].plot(pt,[float(np.mean(np.abs(temp[t]))) for t in range(T)],"ko-",label="Obs",lw=2); axes[0,1].set(xlabel="Time",ylabel="Mean |expr|",title="B: SDE Ensemble"); axes[0,1].legend()
res=np.abs(op[1:]-temp[1:]).mean(axis=0); axes[1,0].hist(res,bins=50,color="#e74c3c",alpha=0.7); axes[1,0].set(xlabel="Residual",ylabel="Count",title="C: ODE Error")
axes[1,1].hist(sig.flatten(),bins=50,color="#8e44ad",alpha=0.7); axes[1,1].axvline(x=np.mean(sig),color="red",ls="--",label=f"mean={np.mean(sig):.4f}"); axes[1,1].set(xlabel="sigma",ylabel="Count",title="D: Diffusion"); axes[1,1].legend()
fig.suptitle("Figure 6: Neural ODE/SDE",fontsize=14,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(FIG/"figure_06_dynamics.png",dpi=150,bbox_inches="tight"); plt.close()
print(f"  done ({time.time()-t0:.1f}s)")

# ================================================================
# FIGURE 7: Perturbation Screen
# ================================================================
print("Fig 7: Perturbation...")
t0 = time.time()
kp1,kp2=jax.random.split(jax.random.PRNGKey(99))
pp=devograph.PerturbationPredictor(gene_dim=fd,hidden_dim=64,num_fates=3,conv_cls=hgx.UniGCNConv,num_layers=2,key=kp1)
te=jnp.broadcast_to(effects[:6,:,None],(6,n_genes,fd))
pp=devograph.train_perturbation_predictor(pp,hg,perturbations=masks[:6],targets=(te,fates[:6]),key=kp2,epochs=200,lr=1e-3)
for i,tf in enumerate(list(kti.keys())[6:]):
    ec2,fp2=devograph.in_silico_knockout(pp,hg,gene_idx=kti[tf])
    r=float(np.corrcoef(np.array(ec2).mean(-1),np.array(effects[6+i]))[0,1]) if np.std(np.array(effects[6+i]))>0 else 0
    print(f"  {tf}: r={r:.3f}")
tia=jnp.array([tgi[t] for t in tfs if t in tgi])
print(f"  Screening {len(tia)} TFs...")
ac,af=devograph.perturbation_screen(pp,hg,tia)

fig,axes=plt.subplots(1,2,figsize=(14,5))
cn=np.array(ac.mean(axis=-1)); mg=np.abs(cn).mean(axis=1); t20=np.argsort(mg)[-20:]
stfs=[t for t in tfs if t in tgi]
im=axes[0].imshow(cn[t20,:50],aspect="auto",cmap="RdBu_r",vmin=-0.3,vmax=0.3)
axes[0].set_yticks(range(20)); axes[0].set_yticklabels([stfs[i] if i<len(stfs) else "" for i in t20],fontsize=6)
axes[0].set(xlabel="Gene",title="A: Top 20 KOs"); plt.colorbar(im,ax=axes[0])
fn=np.array(af); axes[1].scatter(fn[:,0],fn[:,1],c=fn[:,2],cmap="RdYlGn",s=10,alpha=0.5)
axes[1].set(xlabel="DF",ylabel="VF",title=f"B: Fate Space ({len(tia)} KOs)"); axes[1].grid(True,alpha=0.3)
fig.suptitle("Figure 7: Perturbation Screen (720 TFs)",fontsize=14,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(FIG/"figure_07_perturbation.png",dpi=150,bbox_inches="tight"); plt.close()
print(f"  done ({time.time()-t0:.1f}s)")

# ================================================================
# FIGURE 8: Persistent Homology
# ================================================================
print("Fig 8: Persistence...")
t0 = time.time()
try:
    rng=np.random.default_rng(42); si=rng.choice(n_genes,500,replace=False)
    si2=np.array(inc)[si]; ae=si2.sum(0)>=2
    shg=hgx.from_incidence(jnp.array(si2[:,ae]),node_features=feat[si])
    dgms=devograph.compute_persistence(shg,filtration="weight",max_dim=1)
    print(f"  H0={len(dgms[0])}, H1={len(dgms[1])}")
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for d,dgm in enumerate(dgms):
        if len(dgm)>0: axes[0].scatter(dgm[:,0],dgm[:,1],c="#3498db" if d==0 else "#e74c3c",s=15,alpha=0.5,label=f"H{d} ({len(dgm)})")
    ap=np.concatenate([d for d in dgms if len(d)>0])
    axes[0].plot([ap.min(),ap.max()],[ap.min(),ap.max()],"k--",alpha=0.3)
    axes[0].set(xlabel="Birth",ylabel="Death",title="A: Persistence"); axes[0].legend()
    if len(dgms[0])>0:
        lt=dgms[0][:,1]-dgms[0][:,0]; axes[1].hist(lt,bins=50,color="#3498db",alpha=0.7)
    axes[1].set(xlabel="Lifetime",ylabel="Count",title="B: H0 Lifetimes")
    fig.suptitle("Figure 8: Persistent Homology",fontsize=14,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(FIG/"figure_08_persistence.png",dpi=150,bbox_inches="tight"); plt.close()
    print(f"  done ({time.time()-t0:.1f}s)")
except Exception as e: print(f"  Failed: {e}")

total = time.time() - t0_total
print(f"\n{'='*60}")
print(f"  ALL 8 FIGURES COMPLETE in {total:.0f}s")
print(f"  Saved to: {FIG}")
print(f"  Device: {jax.devices()}")
print(f"{'='*60}")
for f in sorted(FIG.glob("*.png")):
    print(f"  {f.name} ({f.stat().st_size/1024:.0f} KB)")
