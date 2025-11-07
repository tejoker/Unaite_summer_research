import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # requis pour les graphiques 3D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # nécéssaire pour activer le mode 3D

# --- Fonctions de génération des séries temporelles (issues de test3.py) ---
def generate_trend(T, slope=None, intercept=None, sigma=1.0, seed=None):
    """Génère une série avec tendance linéaire + bruit gaussien."""
    if seed is not None:
        np.random.seed(seed)
    # Tirer une pente aléatoire si non fournie
    if slope is None:
        slope = np.random.uniform(-0.1, 0.1)
    if intercept is None:
        intercept = np.random.normal(0, 1)
    t = np.arange(T)
    noise = np.random.normal(0, sigma, size=T)
    return intercept + slope * t + noise

def generate_seasonal(T, amplitude=None, period=None, phase=None, sigma=1.0, seed=None):
    """Génère une série saisonnière sinusoïdale + bruit."""
    if seed is not None:
        np.random.seed(seed)
    if amplitude is None:
        amplitude = np.random.uniform(0.5, 2.0)
    if period is None:
        period = np.random.uniform(T/8, T/2)  # période aléatoire entre T/8 et T/2
    if phase is None:
        phase = np.random.uniform(0, 2*np.pi)
    t = np.arange(T)
    noise = np.random.normal(0, sigma, size=T)
    return amplitude * np.sin(2 * np.pi * t / period + phase) + noise

def generate_random_walk(T, sigma=1.0, seed=None):
    """Génère une série en marche aléatoire (somme cumulative de bruits)."""
    if seed is not None:
        np.random.seed(seed)
    steps = np.random.normal(0, sigma, size=T)
    return np.cumsum(steps)

def generate_ar1(T, alpha=0.8, sigma=1.0, seed=None):
    """Génère une série suivant un processus AR(1)."""
    if seed is not None:
        np.random.seed(seed)
    y = np.zeros(T)
    # Initialisation
    y[0] = np.random.normal(0, sigma)
    # Générer récursivement : y[t] = alpha*y[t-1] + bruit
    for t in range(1, T):
        y[t] = alpha * y[t-1] + np.random.normal(0, sigma)
    return y

def generate_diverse_multivariate(D, T, seed=None):
    """Génère D séries temporelles de longueur T avec comportements variés."""
    if seed is not None:
        np.random.seed(seed)
    types = ['trend', 'seasonal', 'random_walk', 'ar1']
    Y = np.zeros((D, T))
    for i in range(D):
        ts_type = np.random.choice(types)      # choisir un type de série aléatoirement
        sigma = np.random.uniform(0.2, 1.0)    # écart-type du bruit aléatoire pour cette série
        if ts_type == 'trend':
            Y[i] = generate_trend(T, sigma=sigma)
        elif ts_type == 'seasonal':
            Y[i] = generate_seasonal(T, sigma=sigma)
        elif ts_type == 'random_walk':
            Y[i] = generate_random_walk(T, sigma=sigma)
        else:
            # Processus AR(1) avec coefficient alpha aléatoire
            alpha = np.random.uniform(0.5, 0.99)
            Y[i] = generate_ar1(T, alpha=alpha, sigma=sigma)
    return Y

# --- Fonction d'application de l'interaction spatio-temporelle (de test3.py) ---
def apply_spatiotemporal(Y, lag=1, Beta_mean=None, sigma_beta=0.05, seed=None):
    """
    Applique l'interaction spatio-temporelle à un ensemble de séries Y.
    - Y : numpy array de shape (D, T) contenant D séries de longueur T.
    - lag : décalage temporel pour l'influence (entier).
    - Beta_mean : matrice (D, D) des moyennes β par paire (i,j) (None pour aléatoire).
    - sigma_beta : écart-type du bruit ajouté à chaque β tiré.
    Retourne:
    - Y_spatio : numpy array (D, T) des séries après interaction spatio-temporelle.
    - diffs_k  : numpy array (D, T, lag+1) des différences ΔY[i,t,k] enregistrées.
    """
    if seed is not None:
        np.random.seed(seed)
    D, T = Y.shape
    # Construire Beta_mean aléatoirement si non fourni
    if Beta_mean is None:
        Beta_mean = np.random.uniform(0.2, 0.8, size=(D, D))
        np.fill_diagonal(Beta_mean, 0.0)  # pas d'auto-influence (diagonale à 0)
    Y_spatio = Y.copy()
    diffs_k = np.zeros((D, T, lag+1))
    # Parcourir le temps et les dimensions pour appliquer l'influence spatio-temporelle
    for t in range(T):
        for i in range(D):
            # Choisir une autre série j au hasard
            js = [j for j in range(D) if j != i]
            j = np.random.choice(js)
            # Appliquer l'effet spatio-temporel une fois qu'on a au moins 'lag' pas de temps
            if t >= lag:
                # Tirer β_ijt autour de la moyenne Beta_mean[i,j] avec un bruit sigma_beta
                beta_ijt = Beta_mean[i, j] + np.random.normal(0, sigma_beta)
                # Fusion: combiner Y[i,t] avec Y[j,t-lag] selon β_ijt
                Y_spatio[i, t] = (1 - beta_ijt) * Y[i, t] + beta_ijt * Y[j, t - lag]
            # Calculer et stocker les différences ΔY[i,t,k] = Y[j, t-k] - Y[i, t] pour k = 0..lag
            for k in range(lag+1):
                if t - k >= 0:
                    diffs_k[i, t, k] = Y[j, t - k] - Y[i, t]
                else:
                    diffs_k[i, t, k] = np.nan  # pas définie pour k > t
    return Y_spatio, diffs_k
def summarize_diffs(diffs_k):
    """
    Imprime en console des stats globales et par série/lag
    pour la matrice diffs_k de shape (D, T, lag+1).
    """
    D, T, lagp1 = diffs_k.shape
    print(f"Spatio-temporal differences : D={D}, T={T}, lags={lagp1-1}")
    all_vals = diffs_k[~np.isnan(diffs_k)]
    print(f"→ Global  : n={all_vals.size}, μ={all_vals.mean():.3f}, σ={all_vals.std():.3f}, min={all_vals.min():.3f}, max={all_vals.max():.3f}")
    for i in range(D):
        print(f" Série {i+1} :")
        for k in range(lagp1):
            vals = diffs_k[i, :, k]
            vals = vals[~np.isnan(vals)]
            print(f"  • lag={k:2d} — n={len(vals):4d}, μ={vals.mean():.3f}, σ={vals.std():.3f}, min={vals.min():.3f}, max={vals.max():.3f}")



# --- Paramètres globaux ---
D   = 10      # nombre de séries (dimensions)
T   = 100    # longueur de chaque série
lag = 2       # décalage temporel maximal considéré
seed = 2025   # graine aléatoire pour reproductibilité

# Générer la matrice Beta_mean si on veut la fixer manuellement (optionnel)
np.random.seed(seed)
Beta_mean = np.random.uniform(0.2, 0.8, size=(D, D))
np.fill_diagonal(Beta_mean, 0.0)  # pas d'auto-influence sur la diagonale
sigma_beta = 0.05  # écart-type du bruit sur β

# --- Génération des séries de base et application de l'interaction spatio-temporelle ---
Y_base = generate_diverse_multivariate(D, T, seed=seed)              # Générer D séries de base variées
Y_spatio, diffs_k = apply_spatiotemporal(Y_base, lag=lag, 
                                        Beta_mean=Beta_mean, 
                                        sigma_beta=sigma_beta, 
                                        seed=seed)                   # Appliquer l'interaction spatio-temporelle

# Couleurs distinctes pour chaque série
colors = plt.cm.tab10(np.linspace(0, 1, D))

# Création de la figure globale avec 2 lignes et 2 colonnes de sous-graphiques
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[2, 1])

# === 1) Histogramme 3D global des différences ===
ax_top = fig.add_subplot(gs[0, :], projection='3d')  # sous-graphe 3D occupant toute la 1ère ligne

# Préparer les données pour l'histogramme 3D : listes des offsets (t-k) et des valeurs ΔY correspondantes
offsets = []
diff_values = []
for i in range(D):
    for k_val in range(lag+1):
        for t in range(T):
            if t >= k_val:  # différence définie pour t-k >= 0
                val = diffs_k[i, t, k_val]
                if not np.isnan(val):
                    offsets.append(t - k_val)
                    diff_values.append(val)

# Calculer un histogramme 2D : X = t-k, Y = valeur ΔY
num_bins = 10  # par exemple, 10 intervalles pour ΔY
hist, xedges, yedges = np.histogram2d(offsets, diff_values, 
                                     bins=[T, num_bins],
                                     range=[[0, T], [min(diff_values), max(diff_values)]])

# Construire les barres 3D à partir de l'histogramme
xpos, ypos, zpos = [], [], []
dx, dy, dz = [], [], []
# Taille d'une bin en X et Y
bin_width_x = xedges[1] - xedges[0]  # ~1 ici
bin_width_y = yedges[1] - yedges[0]
# On peut réduire un peu la taille des barres pour un léger espacement
bar_width_x = 0.8 * bin_width_x
bar_width_y = 0.8 * bin_width_y

# Boucle sur chaque bin de l'histogramme 2D
for xi in range(len(xedges)-1):
    for yi in range(len(yedges)-1):
        count = hist[xi, yi]
        xpos.append(xedges[xi])
        ypos.append(yedges[yi])
        zpos.append(0)
        dx.append(bar_width_x)
        dy.append(bar_width_y)
        dz.append(count)

# # Appliquer un colormap en fonction de la hauteur des barres (fréquence)
# dz = np.array(dz)
# cmap = plt.cm.viridis
# min_h, max_h = dz.min(), dz.max()
# colors_bar = [cmap((h - min_h) / (max_h - min_h) if max_h > min_h else 0) for h in dz]
# Nombre de bins en X (t-k) et en Y (valeurs de ΔY)
nx = T        # ou un autre choix de binning sur t-k
ny = 10       # par exemple 10 bins pour ΔY

# Pré‐allocation
xpos, ypos, zpos = [], [], []
dx, dy, dz = [], [], []
bar_series_idx = []    # <–– on va y stocker l'indice i de la série

# largeur des bars
bin_width_x = 1
bin_width_y = (yedges[1] - yedges[0])
bar_width_x = 0.8 * bin_width_x
bar_width_y = 0.8 * bin_width_y

# Pour CHAQUE série i, on calcule son propre histogramme 2D
for i in range(D):
    # 1) Collecte des offsets (t-k) et des ΔY pour cette série
    offs_i, diffs_i = [], []
    for k_val in range(lag+1):
        for t in range(k_val, T):
            offs_i.append(t - k_val)
            diffs_i.append(diffs_k[i, t, k_val])

    # 2) Histogramme 2D pour la série i
    hist_i, xedges, yedges = np.histogram2d(
        offs_i, diffs_i,
        bins=[nx, ny],
        range=[[0, T], [min(diffs_i), max(diffs_i)]]
    )

    # 3) Pour chaque bin non vide, on enregistre la barre + la série i
    for xi in range(hist_i.shape[0]):
        for yi in range(hist_i.shape[1]):
            count = hist_i[xi, yi]
            if count > 0:
                xpos.append(xedges[xi])
                ypos.append(yedges[yi])
                zpos.append(0)
                dx.append(bar_width_x)
                dy.append(bar_width_y)
                dz.append(count)
                bar_series_idx.append(i)    # <–– ici on note i

# 4) On génère enfin les couleurs de barre à partir de bar_series_idx
colors_bar = [colors[i] for i in bar_series_idx]

# 5) On trace
ax_top.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors_bar, zsort='average')


# Tracé des barres 3D
ax_top.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors_bar, zsort='average')
ax_top.set_xlabel("t-k (décalage temporel)")
ax_top.set_ylabel("ΔY (valeur)")
ax_top.set_zlabel("Fréquence")
ax_top.set_title("Histogramme 3D des écarts ΔY en fonction de t-k")

# (Optionnel) ajuster les ticks pour plus de clarté
ax_top.set_xticks(range(0, T, 5))
# inclure 0 sur l'axe Y si 0 est dans la plage
if 0 >= yedges[0] and 0 <= yedges[-1]:
    ax_top.set_yticks([yedges[0], 0, yedges[-1]])
else:
    ax_top.set_yticks([yedges[0], yedges[-1]])

# === 2) Histogrammes superposés à k=0 (par série) ===
fixed_k = 0  # lag fixé (exemple: 0)
ax_hist = fig.add_subplot(gs[1, 0])
# Définir des bornes de bins communes à toutes les séries pour k=0
all_vals_k0 = diffs_k[:, :, fixed_k].flatten()
all_vals_k0 = all_vals_k0[~np.isnan(all_vals_k0)]
bins = np.linspace(all_vals_k0.min(), all_vals_k0.max(), 15)  # par ex. 15 intervalles
# Tracer un histogramme par série i
for i in range(D):
    data_i = diffs_k[i, :, fixed_k]
    data_i = data_i[~np.isnan(data_i)]
    ax_hist.hist(data_i, bins=bins, alpha=0.5, color=colors[i], label=f"Série {i+1}")
ax_hist.set_title(f"Distribution de ΔY à k={fixed_k} par série")
ax_hist.set_xlabel("Valeur de ΔY")
ax_hist.set_ylabel("Fréquence")
ax_hist.legend(loc='upper right')

# === 3) Évolution temporelle de ΔY pour une paire fixée (i,j) à k=1 ===
i_pair, j_pair = 0, 1        # exemple : série 1 influencée par série 2 (indices 0 et 1)
fixed_k2 = 1                # lag fixé (exemple: 1)
ax_time = fig.add_subplot(gs[1, 1])
# Calculer ΔY = Y[j,t-k] - Y[i,t] pour t >= k (en utilisant Y_base original)
t_vals = np.arange(T)
diff_pair = np.full(T, np.nan)
for t in range(fixed_k2, T):
    diff_pair[t] = Y_base[j_pair, t - fixed_k2] - Y_base[i_pair, t]
# Tracé en barres avec couleur variant selon le signe de ΔY
bar_colors = ['#%02x%02x%02x%02x' % tuple((colors[j_pair]*255).astype(int)) if d >= 0 
              else '#%02x%02x%02x%02x' % tuple((colors[i_pair]*255).astype(int)) 
              for d in diff_pair[fixed_k2:]]
# (Note: on convertit la couleur RGBA de numpy en code hex RGBA pour chaque barre)
ax_time.bar(t_vals[fixed_k2:], diff_pair[fixed_k2:], color=bar_colors)
ax_time.axhline(y=0, color='black', linewidth=0.8)  # ligne horizontale à 0
ax_time.set_title(f"Évolution de ΔY au cours du temps (séries {i_pair+1} et {j_pair+1}, k={fixed_k2})")
ax_time.set_xlabel("Temps t")
ax_time.set_ylabel(f"ΔY (série {j_pair+1} - série {i_pair+1})")
# Légende pour indiquer la signification des couleurs
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor=colors[j_pair], label=f"Série {j_pair+1} > Série {i_pair+1}"),
    Patch(facecolor=colors[i_pair], label=f"Série {i_pair+1} > Série {j_pair+1}")
]
ax_time.legend(handles=legend_elems, loc='upper right')

plt.tight_layout()
plt.show()

def plot_time_series(Y_base, Y_spatio, T, D):
    """
    Plot the generated time series data.
    
    Args:
        Y_base: Original time series data, shape (D, T)
        Y_spatio: Time series after spatiotemporal interaction, shape (D, T)
        T: Length of each time series
        D: Number of time series
    """
    colors = plt.cm.tab10(np.linspace(0, 1, D))
    t = np.arange(T)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot original time series
    for i in range(D):
        ax1.plot(t, Y_base[i], color=colors[i], label=f"Series {i+1}")
    ax1.set_title("Original Time Series")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.legend(loc='upper right', ncol=2)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot time series after spatiotemporal interaction
    for i in range(D):
        ax2.plot(t, Y_spatio[i], color=colors[i], label=f"Series {i+1}")
    ax2.set_title("Time Series After Spatiotemporal Interaction")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Value")
    ax2.legend(loc='upper right', ncol=2)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def identify_ts_impacts(Y_spatio, diffs_k, threshold_percentile=95, top_n=20):
    """
    Identifies impacts between time series - which series influences which,
    when, at what lag, and whether positive or negative.
    
    Args:
        Y_spatio: Time series after spatiotemporal interaction
        diffs_k: Differences matrix from apply_spatiotemporal
        threshold_percentile: Percentile for significant impact threshold
        top_n: Number of top impacts to display
    """
    D, T, lagp1 = diffs_k.shape
    lag = lagp1 - 1
    
    # Calculate threshold based on percentile of absolute differences
    all_diffs = diffs_k[~np.isnan(diffs_k)]
    threshold = np.percentile(np.abs(all_diffs), threshold_percentile)
    print(f"Significance threshold (percentile {threshold_percentile}): {threshold:.4f}")
    
    # Store impacts
    impacts = []
    
    for i in range(D):  # target series
        for j in range(D):  # source series
            if i == j:
                continue  # Skip self-influence
            
            for t in range(lag, T):
                for k in range(lagp1):
                    if t - k < 0:
                        continue
                    
                    diff_value = Y_spatio[j, t-k] - Y_spatio[i, t]
                    
                    if np.abs(diff_value) > threshold:
                        impacts.append({
                            'source': j,
                            'target': i,
                            'time': t,
                            'lag': k,
                            'value': diff_value,
                            'sign': 'positive' if diff_value > 0 else 'negative'
                        })
    
    # Display impacts
    print(f"\nFound {len(impacts)} significant impacts")
    if impacts:
        sorted_impacts = sorted(impacts, key=lambda x: abs(x['value']), reverse=True)
        
        print(f"\nTop {min(top_n, len(sorted_impacts))} most significant impacts:")
        print("-" * 80)
        print(f"{'SOURCE':^10} {'→':^5} {'TARGET':^10} {'TIME':^10} {'LAG':^10} {'IMPACT':^15} {'SIGN':^10}")
        print("-" * 80)
        
        for impact in sorted_impacts[:top_n]:
            print(f"Series {impact['source']+1:^8} → Series {impact['target']+1:^8} t={impact['time']:^8} k={impact['lag']:^8} {impact['value']:^15.4f} {impact['sign']:^10}")
        
        # Summarize by source-target pairs
        print("\nSummary by source-target pairs:")
        pair_counts = {}
        for impact in impacts:
            pair = (impact['source'], impact['target'])
            if pair not in pair_counts:
                pair_counts[pair] = {'count': 0, 'positive': 0, 'negative': 0}
            pair_counts[pair]['count'] += 1
            pair_counts[pair][impact['sign']] += 1
        
        sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1]['count'], reverse=True)
        
        print("-" * 80)
        print(f"{'SOURCE':^10} {'→':^5} {'TARGET':^10} {'COUNT':^10} {'POSITIVE':^10} {'NEGATIVE':^10}")
        print("-" * 80)
        
        for pair, stats in sorted_pairs:
            src, tgt = pair
            print(f"Series {src+1:^8} → Series {tgt+1:^8} {stats['count']:^10} {stats['positive']:^10} {stats['negative']:^10}")
    
    return impacts

if __name__ == "__main__":
    # … vos lignes de génération inchangées …
    Y_base = generate_diverse_multivariate(D, T, seed=seed)
    Y_spatio, diffs_k = apply_spatiotemporal(
        Y_base, lag=lag,
        Beta_mean=Beta_mean,
        sigma_beta=sigma_beta,
        seed=seed
    )

    # --- NOUVEAU BLOC POUR TOUTES LES PAIRES (i,j) ---
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            print(f"Série {i+1} par rapport à Série {j+1} :")
            for t in range(T):
                # pour chaque décalage k de 0 à lag, on ne garde
                # que ceux où t-k >= 0
                vals = [
                    Y_spatio[j, t-k] - Y_spatio[i, t]
                    for k in range(lag+1)
                    if t - k >= 0
                ]
                print(f"  t={t:2d} → {vals}")
            print()

    plot_time_series(Y_base, Y_spatio, T, D)
    impacts = identify_ts_impacts(Y_spatio, diffs_k, threshold_percentile=95, top_n=20)
    print(impacts)


