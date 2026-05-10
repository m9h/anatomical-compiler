import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

def main():
    print("Loading Fleck Organoid GRN...")
    data_dir = Path("data/processed")
    with open(data_dir / "gene_names.json") as f: genes = json.load(f)
    with open(data_dir / "tf_names.json") as f: tfs = json.load(f)
    incidence = np.load(data_dir / "incidence.npy") # (n_genes, n_tfs)
    
    print("Loading Disease Gene Lists (Sonthalia 2026)...")
    with open("data/disease_gene_lists.json") as f: disease_lists = json.load(f)
    
    categories = ["SFARI_Score1", "ASD_HC65", "DD", "Microcephlaly", "Epilepsy", "SCZ"]
    
    # Target TFs from previous analyses
    target_tfs = ["NR2E1", "ARX", "MEIS2", "SOX2", "ASCL1", "FOXG1", "TBR1", "EMX1"]
    
    results = []
    print(f"Analyzing {len(target_tfs)} master TFs for disease enrichment in their regulons...")
    
    for tf in target_tfs:
        if tf not in tfs: continue
        ti = tfs.index(tf)
        regulon_mask = (incidence[:, ti] > 0)
        reg_genes = [genes[i] for i, val in enumerate(regulon_mask) if val]
        n_reg = len(reg_genes)
        
        if n_reg == 0: continue
        
        for cat in categories:
            if cat not in disease_lists: continue
            dis_genes = set(disease_lists[cat])
            
            both = set(reg_genes) & dis_genes
            n_both = len(both)
            n_dis = len(dis_genes)
            n_total = len(genes)
            
            table = [[n_both, n_reg - n_both], [n_dis - n_both, n_total - n_reg - n_dis + n_both]]
            _, pval = stats.fisher_exact(table, alternative="greater")
            
            results.append({
                "tf": tf,
                "category": cat,
                "n_both": n_both,
                "n_reg": n_reg,
                "n_dis": n_dis,
                "p_val": pval,
                "genes": sorted(list(both))
            })
            if pval < 0.05:
                print(f"  {tf:<6} - {cat:<15}: p={pval:.2e}, shared={n_both}")

    df = pd.DataFrame(results)
    if not df.empty:
        # Multiple testing correction (TF x Category)
        df['p_adj'] = df['p_val'] * len(target_tfs) * len(categories)
        df.loc[df['p_adj'] > 1.0, 'p_adj'] = 1.0
        
        # Plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df[df.p_val < 0.1], x="tf", y=-np.log10(df[df.p_val < 0.1].p_val), hue="category")
        plt.axhline(-np.log10(0.05 / (len(target_tfs) * len(categories))), color="red", linestyle="--", label="Bonferroni")
        plt.title("Disease Enrichment in Master TF Regulons (Organoid GRN)")
        plt.ylabel("-log10(p-value)")
        
        out_path = Path("figures/tf_disease_enrichment.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"\n  Figure saved to {out_path}")

    # Save results
    def _sanitize(o):
        if isinstance(o, (np.int32, np.int64, np.integer)): return int(o)
        if isinstance(o, (np.float32, np.float64, np.floating)): return float(o)
        if isinstance(o, list): return [_sanitize(x) for x in o]
        if isinstance(o, dict): return {k: _sanitize(v) for k, v in o.items()}
        return o

    with open("figures/tf_disease_results.json", "w") as f:
        json.dump(_sanitize(results), f, indent=2)

if __name__ == "__main__":
    main()
