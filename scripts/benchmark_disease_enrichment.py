import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

def main():
    print("Loading Fleck Organoid GRN and Modules...")
    data_dir = Path("data/processed")
    with open(data_dir / "gene_names.json") as f: genes = json.load(f)
    module_labels = np.load(data_dir / "module_labels.npy") # (n_genes,)
    
    print("Loading Disease Gene Lists (Sonthalia 2026)...")
    with open("data/disease_gene_lists.json") as f: disease_lists = json.load(f)
    
    # 1. Filter for major categories
    categories = ["SFARI_Score1", "ASD_HC65", "DD", "Microcephlaly", "Epilepsy", "SCZ"]
    
    results = []
    unique_modules = sorted(set(module_labels))
    if -1 in unique_modules: unique_modules.remove(-1) # Remove unassigned
    
    print(f"Analyzing {len(unique_modules)} modules for disease enrichment...")
    
    for mod in unique_modules:
        mod_mask = (module_labels == mod)
        mod_genes = [genes[i] for i, val in enumerate(mod_mask) if val]
        n_mod = len(mod_genes)
        
        for cat in categories:
            if cat not in disease_lists: continue
            dis_genes = set(disease_lists[cat])
            
            # Overlap
            both = set(mod_genes) & dis_genes
            n_both = len(both)
            n_dis = len(dis_genes)
            n_total = len(genes)
            
            # Fisher test
            # Table: [[both, mod-both], [dis-both, total-mod-dis+both]]
            table = [[n_both, n_mod - n_both], [n_dis - n_both, n_total - n_mod - n_dis + n_both]]
            _, pval = stats.fisher_exact(table, alternative="greater")
            
            if pval < 0.05:
                results.append({
                    "module": mod,
                    "category": cat,
                    "n_both": n_both,
                    "n_mod": n_mod,
                    "n_dis": n_dis,
                    "p_val": pval,
                    "genes": sorted(list(both))
                })
    
    df = pd.DataFrame(results)
    if not df.empty:
        # Multiple testing correction
        df['p_adj'] = df['p_val'] * len(unique_modules) * len(categories)
        df.loc[df['p_adj'] > 1.0, 'p_adj'] = 1.0
        
        print("\nSignificant Enrichments:")
        print(df[df.p_adj < 0.05][['module', 'category', 'n_both', 'p_adj']])
        
        # 2. Plot
        plt.figure(figsize=(12, 6))
        # Top 10 by significance
        top_plot = df.nsmallest(15, "p_val")
        sns.barplot(data=top_plot, x="module", y=-np.log10(top_plot.p_val), hue="category")
        plt.axhline(-np.log10(0.05 / (len(unique_modules) * len(categories))), color="red", linestyle="--", label="Bonferroni")
        plt.title("Disease Enrichment in Organoid GRN Modules (Sonthalia 2026 Lists)")
        plt.ylabel("-log10(p-value)")
        
        out_path = Path("figures/disease_enrichment_sonthalia.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"\n  Figure saved to {out_path}")
    else:
        print("No significant enrichments found.")

    # Save results
    def _sanitize(o):
        if isinstance(o, (np.int32, np.int64, np.integer)):
            return int(o)
        if isinstance(o, (np.float32, np.float64, np.floating)):
            return float(o)
        if isinstance(o, list):
            return [_sanitize(x) for x in o]
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        return o

    with open("figures/disease_enrichment_results.json", "w") as f:
        json.dump(_sanitize(results), f, indent=2)

if __name__ == "__main__":
    main()
