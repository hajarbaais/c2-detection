# Experiences (Hajar) — ameliorations P6/P6.7

Branche **non mergee dans main**. Experiences autour de la detection C2 :
correction des labels, fusion de features, detection au niveau paire, regle des trois.
Tout tourne en **PyTorch pur** (pas besoin de scikit-learn, qui est bloque sur certaines machines).

## Prerequis

### 1. Donnees (gitignorees — a recuperer sur le Drive)
| Fichier | Livrable | Ou le mettre |
|---|---|---|
| `sessions_pcap.parquet`, `events.parquet`, `events_pkts.parquet`, `splits/` | **P2** | un dossier, ex. `~/Desktop/P2_output` |
| `sessions_unified.parquet` (colonne `ctu13_broad_label`) | **P3** | n'importe ou, chemin passe en argument |
| `features_dns.parquet` | **P4** | `p4_features_dns_time/output/` (ou regenerer avec les scripts P4) |

### 2. Generer les features de sequences (une fois)
```bash
python p5_features_tls_seq/scripts/p5_features.py \
    --p2 <P2_output> --out p5_features_tls_seq/output --seq-len 64
```

### 3. Environnement
```bash
pip install torch pandas pyarrow numpy
```

## Lancer les experiences

```bash
P2=<P2_output>
UNI=<chemin>/sessions_unified.parquet

# 1) Fusion de features + correction des labels (DNS ~100 %, web ~10 % a 0,1 % FPR)
python -I experiments/exp1_label_fix_fusion.py --p2 $P2 --unified $UNI

# 2) Detection au niveau paire (web ~63 % a 0,5 % FPR, trickbot detecte)
python -I experiments/exp2_pair_beaconing.py --p2 $P2 --unified $UNI

# 3) Regle des trois (taille du jeu de test benin)
python -I experiments/exp3_regle_des_trois.py --p2 $P2
```
Sous Windows PowerShell : remplacer `$P2` / `$UNI` par les chemins complets.

## Resultats obtenus (seuil calibre sur grand ensemble benin)

| Perimetre | Recall | FPR | Note |
|---|---|---|---|
| **DNS (en clair)** | **100 %** | 0,09 % | apres correction des labels de convention |
| **Web (chiffre, niveau paire)** | **63 %** | 0,48 % | seuil tolerant (0,5 %) ; trickbot/dridex detectes |
| Web (chiffre, niveau paire) | 13 % | 0,09 % | seuil strict 0,1 % — mur du chiffre |

## Idees clefs
- **Labels** : CTU-13 marque "C2" tout le trafic d'une machine infectee (convention). Le drapeau
  `ctu13_broad_label` (P3) permet de retirer ce bruit -> DNS passe a 100 %.
- **Fusion de features** : un seul modele avec toutes les features (P4 + P5 + resume sequences).
- **Niveau paire** : le C2 chiffre se detecte sur la paire (machine <-> serveur) qui balise
  longtemps, pas sur une session isolee.
- **Regle des trois** : il faut >= 3 000 benins (~30 000 confortable) pour calibrer un seuil a 0,1 %.
