import os
import sys
from typing import IO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Dossier de sortie pour graphes et résultats

OUTPUT_DIR = "tp3_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class Tee:
    def __init__(self, *files: IO[str]) -> None:
        self.files = files

    def write(self, obj: str) -> None:
        for f in self.files:
            f.write(obj)
            try:
                f.flush()
            except ValueError:
                pass

    def flush(self) -> None:
        for f in self.files:
            try:
                f.flush()
            except ValueError:
                pass


log_file: IO[str] = open(os.path.join(OUTPUT_DIR, "output.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)  # type: ignore[assignment]


print("TP3 ")
print("=" * 60)

#  CHARGEMENT DES DONNÉES

df_raw: pd.DataFrame = pd.read_excel("IRIS.xlsx", sheet_name="Feuil1", index_col=0)
df_raw.index = pd.Index(
    [f"{num}_{esp}" for num, esp in zip(df_raw.index, df_raw["Espece"])]
)
df: pd.DataFrame = df_raw.copy()

print("DataFrame charge avec succes.")
print(f"Apercu :\n{df.head()}")

#  DESCRIPTION DES DONNÉES

print(f"\n Nombre d'individus (fleurs) : {df.shape[0]}")
print(f"Nombre de variables          : {df.shape[1]}")
print(f"Noms des variables           : {list(df.columns)}")

print("\n Statistiques elementaires globales ")
print(df.describe())

print("\n Statistiques par espece ")
for espece in df["Espece"].unique():
    print(f"\n== {espece} ==")
    print(df[df["Espece"] == espece].describe())

print(
    """
 Interpretation  :
# PE_L et PE_W ont les plus grands ecarts-types relatifs (CV eleve)
# => distributions plus dispersees, surtout entre especes
# SE_W est la variable la moins dispersee globalement
# La difference de moyenne entre especes est plus marquee pour PE_L et PE_W
# => ces variables sont probablement plus discriminantes
"""
)

#  GRAPHIQUES
print("\n Graphiques")
numeric_vars: list[str] = ["SE_L", "SE_W", "PE_L", "PE_W"]
color_map: dict[str, str] = {
    "SETOSA": "blue",
    "VERSICOLOR": "orange",
    "VIRGINICA": "green",
}

#  Histogramme + densité globale pour SE_L
fig, ax = plt.subplots(figsize=(7, 4))
sel_series: pd.Series = df["SE_L"]  # type: ignore[type-arg]
sel_series.plot.hist(
    ax=ax, bins=15, density=True, alpha=0.5, color="steelblue", label="Histogramme"
)
sel_series.plot.kde(ax=ax, color="red", label="Densite")
ax.set_title("Histogramme et densite de SE_L (global)")
ax.set_xlabel("SE_L (mm)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hist_density_SE_L_global.png"), dpi=150)
plt.close()
print("Graphe enregistre : hist_density_SE_L_global.png")

#  Histogramme et densité par espèce pour SE_L
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
for ax2, (espece, color) in zip(axes2, color_map.items()):
    subset_s: pd.Series = df[df["Espece"] == espece]["SE_L"]  # type: ignore[type-arg]
    subset_s.plot.hist(
        ax=ax2, bins=10, density=True, alpha=0.5, color=color, label="Histogramme"
    )
    subset_s.plot.kde(ax=ax2, color="black", label="Densite")
    ax2.set_title(f"{espece} - SE_L")
    ax2.set_xlabel("SE_L (mm)")
    ax2.legend(fontsize=7)
plt.suptitle("Histogramme et densite de SE_L par espece")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hist_density_SE_L_par_espece.png"), dpi=150)
plt.close()

#  Boxplots comparatifs
fig3, axes3 = plt.subplots(1, 4, figsize=(16, 5))
for ax3, var in zip(axes3, numeric_vars):
    data_by_espece: list[list[float]] = [
        df[df["Espece"] == esp][var].tolist() for esp in color_map.keys()
    ]
    bp = ax3.boxplot(
        data_by_espece, tick_labels=list(color_map.keys()), patch_artist=True
    )
    for patch, clr in zip(bp["boxes"], color_map.values()):
        patch.set_facecolor(clr)
        patch.set_alpha(0.6)
    ax3.set_title(var)
    ax3.set_ylabel("mm")
plt.suptitle("Boxplots comparatifs par espece")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "boxplots_especes.png"), dpi=150)
plt.close()
print("Graphe enregistre : boxplots_especes.png")

#  Nuage de points SE_L vs PE_L
fig4, ax4 = plt.subplots(figsize=(7, 5))
for espece, color in color_map.items():
    sub_df: pd.DataFrame = df[df["Espece"] == espece]
    ax4.scatter(
        sub_df["SE_L"],
        sub_df["PE_L"],
        c=color,
        label=espece,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.3,
    )
ax4.set_xlabel("SE_L (mm)")
ax4.set_ylabel("PE_L (mm)")
ax4.set_title("Nuage de points SE_L vs PE_L")
ax4.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scatter_SE_L_PE_L.png"), dpi=150)
plt.close()

#  Matrice des nuages de points
fig5, axes5 = plt.subplots(4, 4, figsize=(12, 12))
for i, var1 in enumerate(numeric_vars):
    for j, var2 in enumerate(numeric_vars):
        ax5 = axes5[i][j]
        if i == j:
            ax5.hist(df[var1].tolist(), bins=12, color="steelblue", alpha=0.7)
            ax5.set_xlabel(var1, fontsize=8)
        else:
            for espece, color in color_map.items():
                sub2: pd.DataFrame = df[df["Espece"] == espece]
                ax5.scatter(sub2[var2], sub2[var1], c=color, s=15, alpha=0.6)
            ax5.set_xlabel(var2, fontsize=8)
            ax5.set_ylabel(var1, fontsize=8)
        ax5.tick_params(labelsize=6)

legend_handles = [
    mlines.Line2D(
        [], [], marker="o", color="w", markerfacecolor=c, markersize=8, label=e
    )
    for e, c in color_map.items()
]
fig5.legend(handles=legend_handles, loc="upper right", fontsize=9)
plt.suptitle("Matrice des nuages de points", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scatter_matrix.png"), dpi=150)
plt.close()

# CHOIX DES AXES FACTORIELS

# Matrice de correlation
corr_matrix: pd.DataFrame = df[numeric_vars].corr()
print("\nMatrice de correlation :")
print(corr_matrix.round(3))

fig6, ax6 = plt.subplots(figsize=(6, 5))
im = ax6.imshow(corr_matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax6)
ax6.set_xticks(range(4))
ax6.set_xticklabels(numeric_vars)
ax6.set_yticks(range(4))
ax6.set_yticklabels(numeric_vars)
for i in range(4):
    for j in range(4):
        ax6.text(
            j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=10
        )
ax6.set_title("Matrice de correlation")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()

print(
    """
# INTERPRETATION :
# PE_L et PE_W sont fortement correlees (r > 0.9) => mesurent la taille du petale
# SE_L est moyennement correlee avec PE_L et PE_W
# SE_W est faiblement correlee avec les autres => variable plus independante
"""
)

X_raw: NDArray[np.float64] = df[numeric_vars].to_numpy(dtype=float)
X_mean: NDArray[np.float64] = X_raw.mean(axis=0)
X_std_arr: NDArray[np.float64] = X_raw.std(axis=0, ddof=0)
X: NDArray[np.float64] = (X_raw - X_mean) / X_std_arr
corr_np: NDArray[np.float64] = corr_matrix.to_numpy(dtype=float)
raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(corr_np)
sort_idx: NDArray[np.intp] = np.argsort(raw_eigenvalues)[::-1]
eigenvalues: NDArray[np.float64] = raw_eigenvalues[sort_idx].astype(np.float64)
eigenvectors: NDArray[np.float64] = raw_eigenvectors[:, sort_idx].astype(np.float64)
print("\n Valeurs propres")
for i, val in enumerate(eigenvalues):
    print(f"  lambda{i + 1} = {val:.4f}")

print("\n Vecteurs propres (colonnes = axes factoriels) ")
eigenvec_df = pd.DataFrame(
    eigenvectors, index=numeric_vars, columns=[f"PC{i + 1}" for i in range(4)]
)
print(eigenvec_df.round(4))
total_var: float = float(eigenvalues.sum())
variance_pct: NDArray[np.float64] = (eigenvalues / total_var) * 100
cumulative: NDArray[np.float64] = np.cumsum(variance_pct)

print("\n Variance expliquee par axe ")
for i in range(4):
    print(
        f"  PC{i + 1} : lambda={eigenvalues[i]:.4f}  |  "
        f"{variance_pct[i]:.2f}%  |  Cumule: {cumulative[i]:.2f}%"
    )

fig7, ax7 = plt.subplots(figsize=(6, 4))
ax7.bar(range(1, 5), eigenvalues, alpha=0.7, color="steelblue", label="Valeur propre")
ax7.axhline(y=1, color="red", linestyle="--", label="Seuil = 1 (Kaiser)")
ax7.plot(range(1, 5), cumulative, "o-", color="orange", label="Variance cumulee (%)")
for i in range(4):
    ax7.text(
        i + 1,
        float(eigenvalues[i]) + 0.05,
        f"{variance_pct[i]:.1f}%",
        ha="center",
        fontsize=9,
    )
ax7.set_xticks(range(1, 5))
ax7.set_xticklabels([f"PC{i}" for i in range(1, 5)])
ax7.set_title("Scree plot - Valeurs propres")
ax7.set_ylabel("Valeur propre / Variance cumulee (%)")
ax7.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scree_plot.png"), dpi=150)
plt.close()

print(
    """
# INTERPRETATION  :
# Chaque valeur propre lambda_k = variance portee par l'axe factoriel PC_k
# (variance totale dans l'espace standardise = p = nb de variables = 4)
#
# Critere de Kaiser : on retient les axes avec lambda > 1
# => PC1 et PC2 suffisent (variance cumulee > 94%)
# => Le s.e.v de projection est de dimension 2 (plan factoriel)
# => Ce resultat etait previsible car PE_L et PE_W sont fortement correlees :
#    les 4 variables "originales" n'apportent pas 4 dimensions independantes
#
# Pourquoi la matrice de correlation est diagonalisable ?
# => Elle est symetrique reelle => theoreme spectral garantit la diagonalisation
# => Ses vecteurs propres forment une base orthonormee de R^p
#
# Peut-on avoir un vecteur propre nul ?
# => NON : par definition Av = lambda*v avec v != 0
#
# Peut-on avoir une valeur propre nulle ?
# => Seulement si det(R) = 0 deux variables parfaitement colineaires (r=1)
# => r < 1 pour toutes les paires => toutes les valeurs propres > 0
"""
)

#  COORDONNÉES DES INDIVIDUS SUR LES AXES PRINCIPAUX
print("\n  Coordonnees des individus sur les axes principaux ")


coords_np: NDArray[np.float64] = X @ eigenvectors
coords_df: pd.DataFrame = pd.DataFrame(
    coords_np, index=df.index, columns=[f"PC{i + 1}" for i in range(4)]
)
matching_ids: list[str] = [
    str(i) for i in coords_df.index if "i027" in str(i).lower() or "027" in str(i)
]
ind: str | None = matching_ids[0] if matching_ids else None
if ind:
    pc1_val = float(coords_df.loc[ind, "PC1"])
    pc2_val = float(coords_df.loc[ind, "PC2"])
    print(f"\nCoordonnees de '{ind}' sur les deux premiers axes :")
    print(f"  PC1 = {pc1_val:.4f}")
    print(f"  PC2 = {pc2_val:.4f}")
else:
    print("\n Individu i027_VIRGINICA non trouve.")
    print("Individus disponibles :", list(coords_df.index[:5]))

pc1_series: pd.Series = coords_df["PC1"]
max_contrib_idx: str = str(pc1_series.abs().idxmax())
print(f"\n Fleur la plus contributrice sur PC1 :")
print(
    f"  {max_contrib_idx}  (PC1 = {float(coords_df.loc[max_contrib_idx, 'PC1']):.4f})"
)

# Qualite de representation sur le plan PC1-PC2
coords_sq: pd.DataFrame = coords_df**2
cos2_plan: pd.Series = (coords_sq["PC1"] + coords_sq["PC2"]) / coords_sq.sum(axis=1)  # type: ignore[type-arg]
print(f"\n Qualite de representation moyenne (plan PC1-PC2) :")
print(f"  cos2 moyen = {cos2_plan.mean():.4f} ({cos2_plan.mean() * 100:.1f}%)")

if ind:
    row_sq: pd.Series = coords_df.loc[ind] ** 2
    row_total: float = float(row_sq.sum())
    cos2_pc1: float = float(row_sq["PC1"]) / row_total
    cos2_pc2: float = float(row_sq["PC2"]) / row_total
    print(f"\nQualite de representation de '{ind}' :")
    print(f"  cos2(PC1)      = {cos2_pc1:.4f}")
    print(f"  cos2(PC2)      = {cos2_pc2:.4f}")
    print(
        f"  cos2(PC1+PC2)  = {cos2_pc1 + cos2_pc2:.4f} ({(cos2_pc1 + cos2_pc2) * 100:.1f}%)"
    )

print(
    """
# INTERPRETATION qualite de representation :
# cos2 proche de 1 => individu bien represente sur le plan factoriel
# cos2 proche de 0 => individu mal represente (proche du centre de gravite)
# cos2 moyen > 0.85 => projection globalement fiable pour l'echantillon
"""
)

#  Cercle de corrélation
loadings: NDArray[np.float64] = eigenvectors * np.sqrt(eigenvalues)

fig8, ax8 = plt.subplots(figsize=(6, 6))
circle_patch = mpatches.Circle(
    (0.0, 0.0), 1.0, fill=False, color="gray", linestyle="--"
)
ax8.add_patch(circle_patch)
for i, var in enumerate(numeric_vars):
    lx: float = float(loadings[i, 0])
    ly: float = float(loadings[i, 1])
    ax8.annotate(
        "",
        xy=(lx, ly),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    offset_x: float = 0.07 if lx >= 0 else -0.07
    offset_y: float = 0.07 if ly >= 0 else -0.07
    ax8.text(
        lx + offset_x,
        ly + offset_y,
        var,
        fontsize=11,
        color="darkblue",
        fontweight="bold",
    )
ax8.axhline(0.0, color="black", lw=0.7)
ax8.axvline(0.0, color="black", lw=0.7)
ax8.set_xlim(-1.2, 1.2)
ax8.set_ylim(-1.2, 1.2)
ax8.set_xlabel(f"PC1 ({variance_pct[0]:.1f}%)", fontsize=11)
ax8.set_ylabel(f"PC2 ({variance_pct[1]:.1f}%)", fontsize=11)
ax8.set_title("Cercle de correlation (Plan PC1-PC2)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cercle_correlation.png"), dpi=150)
plt.close()

#  Projection des individus sur le plan factoriel
fig9, ax9 = plt.subplots(figsize=(9, 6))
for espece, color in color_map.items():
    mask: pd.Series = df["Espece"] == espece
    ax9.scatter(
        coords_df.loc[mask, "PC1"],
        coords_df.loc[mask, "PC2"],
        c=color,
        label=espece,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.3,
        s=50,
    )
ax9.axhline(0.0, color="gray", lw=0.7)
ax9.axvline(0.0, color="gray", lw=0.7)
ax9.set_xlabel(f"PC1 ({variance_pct[0]:.1f}%)", fontsize=11)
ax9.set_ylabel(f"PC2 ({variance_pct[1]:.1f}%)", fontsize=11)
ax9.set_title("Projection des individus - 1er plan factoriel (ACP)")
ax9.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "projection_individus_plan_factoriel.png"), dpi=150
)
plt.close()

# Correlation entre PC1 et PC2
corr_pc1_pc2: float = float(
    np.corrcoef(coords_df["PC1"].to_numpy(), coords_df["PC2"].to_numpy())[0, 1]
)
print(f"\nCorrelation entre PC1 et PC2 : {corr_pc1_pc2:.6f}")
print(
    """
# INTERPRETATION : correlation PC1/PC2 = 0 (numeriquement)
# => Les composantes principales sont ORTHOGONALES (non correlees)
# => Propriete fondamentale de l'ACP : les axes decomposent la variance
#    sans redondance
"""
)

print(
    """
 INTERPRETATION DES DEUX PREMIERS AXES :
# PC1 (axe 1) : "taille globale de la fleur"
#   => Fortement correle avec PE_L, PE_W et SE_L (meme sens positif)
#   => Separe nettement SETOSA (tres petits petales => PC1 tres negatif)
#      des especes VERSICOLOR et VIRGINICA
#
# PC2 (axe 2) : "contraste sepale / petale"
#   => Correle positivement avec SE_W, negativement avec PE_L et PE_W
#   => Permet de distinguer VERSICOLOR et VIRGINICA qui se chevauchent sur PC1
#
 CARACTERISTIQUES DISCRIMINANTES DES 3 ESPECES :
# SETOSA     : PE_L et PE_W tres faibles, SE_W relativement grand
#              => Groupe isole a gauche sur PC1 (facilement separable)
# VERSICOLOR : Mesures intermediaires
#              => Groupe central, chevauchement partiel avec VIRGINICA
# VIRGINICA  : PE_L, PE_W et SE_L les plus grands
#              => Groupe a droite sur PC1, SE_W moyen
"""
)

print(f"Tous les fichiers sont dans : ./{OUTPUT_DIR}/")

log_file.close()
