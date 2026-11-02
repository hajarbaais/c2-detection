"""P4.4 - Periodicite : retrouver le rythme cache (beaconing) malgre le jitter.

Methode "glisser-comparer" = autocorrelation (via FFT, rapide). On decoupe le temps de la
session en petites cases, on compte les evenements par case, puis on regarde a quel decalage
(lag) le signal "retombe sur lui-meme" -> ce decalage = la periode, sa hauteur = le score.

Features (par session) :
  dns_period_dominant   : l'intervalle de balise principal (secondes), NaN si pas de rythme
  dns_periodicity_score : force du rythme, 0 (aucun) a ~1 (horloge parfaite)

Robuste au jitter : un rythme 60s +/- bruit reste groupe autour des multiples de 60 -> le pic
est un peu plus bas mais toujours la.

Entrees : events.parquet + events_pkts.parquet (horodatages, comme P4.3)
Usage   : python -I p4_04_periodicity.py [chemin_dossier_output_P2]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_P2 = Path.home() / "Desktop" / "P2_livrables_v3" / "P2_livrables_v3" / "output"
P2_OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_P2
OUT_DIR = Path(__file__).resolve().parents[1] / "output"

MIN_EVENTS = 10       # il faut assez de points pour parler de rythme
MAX_BINS = 4096       # plafond pour rester rapide


def periodicity(times):
    """Retourne (periode_dominante_s, score 0..1) par autocorrelation du signal binne."""
    t = np.sort(np.asarray(times, dtype="float64"))
    n = len(t)
    if n < MIN_EVENTS:
        return np.nan, 0.0
    t = t - t[0]
    duration = t[-1]
    diffs = np.diff(t)
    med = np.median(diffs)
    if duration <= 0 or med <= 0:
        return np.nan, 0.0

    # resolution ~ moitie de l'intervalle typique, plafonnee
    n_bins = int(duration / (med / 2))
    n_bins = max(20, min(n_bins, MAX_BINS))
    bin_width = duration / n_bins

    counts = np.histogram(t, bins=n_bins, range=(0.0, duration))[0].astype("float64")
    x = counts - counts.mean()
    if not np.any(x):
        return np.nan, 0.0

    # autocorrelation par FFT, normalisee par le lag 0
    f = np.fft.rfft(x, n=2 * n_bins)
    acf = np.fft.irfft(f * np.conjugate(f))[:n_bins]
    acf /= acf[0]

    hi = n_bins // 2          # on cherche un pic entre lag 2 et la moitie
    if hi <= 2:
        return np.nan, 0.0
    seg = acf[2:hi]
    lag = int(np.argmax(seg)) + 2
    score = float(np.clip(acf[lag], 0.0, 1.0))
    return lag * bin_width, score


def _self_test():
    rng = np.random.default_rng(0)
    # rythme parfait de 60 s
    perfect = np.arange(0, 60 * 60, 60.0)
    _, s_perfect = periodicity(perfect)
    # rythme 60 s avec jitter +/- 15 s (comme un vrai C2)
    jittered = np.cumsum(60 + rng.uniform(-15, 15, 60))
    p_jit, s_jit = periodicity(jittered)
    # temps totalement aleatoires
    rand = np.sort(rng.uniform(0, 3600, 60))
    _, s_rand = periodicity(rand)
    print(f"[test] score  parfait={s_perfect:.2f}  jitter={s_jit:.2f} (periode~{p_jit:.0f}s)  aleatoire={s_rand:.2f}")
    assert s_perfect > 0.5, "le rythme parfait doit avoir un score eleve"
    assert s_jit > s_rand, "le rythme jitte doit battre l'aleatoire"


def main():
    _self_test()

    print(f"[1/5] lecture sessions + horodatages depuis {P2_OUT}")
    se = pd.read_parquet(P2_OUT / "sessions_pcap.parquet", columns=["session_id", "session_type", "label"])
    dns_sessions = set(se.loc[se.session_type == "dns", "session_id"])

    ev = pd.read_parquet(P2_OUT / "events.parquet", columns=["session_id", "event_type", "ts"])
    ts_dns = ev[(ev.event_type == "dns_query") & (ev.session_id.isin(dns_sessions))][["session_id", "ts"]]

    print("[2/5] lecture events_pkts.parquet (110 Mo)...")
    pk = pd.read_parquet(P2_OUT / "events_pkts.parquet", columns=["session_id", "ts", "direction"])
    ts_web = pk[pk.direction == "up"][["session_id", "ts"]]

    ts = pd.concat([ts_dns, ts_web], ignore_index=True)
    # ne garder que les sessions avec assez de points
    sizes = ts.groupby("session_id").size()
    keep = sizes[sizes >= MIN_EVENTS].index
    ts = ts[ts.session_id.isin(keep)]
    print(f"[3/5] {len(keep):,} sessions a analyser (>= {MIN_EVENTS} points)")

    print("[4/5] calcul de la periodicite par session...")
    rows = []
    for i, (sid, arr) in enumerate(ts.groupby("session_id")["ts"], 1):
        period, score = periodicity(arr.values)
        rows.append((sid, period, score))
        if i % 10000 == 0:
            print(f"       {i:,}/{len(keep):,}")
    per = pd.DataFrame(rows, columns=["session_id", "dns_period_dominant", "dns_periodicity_score"])

    # fusion dans features_dns.parquet (sessions non analysees -> score 0, periode NaN)
    feat_path = OUT_DIR / "features_dns.parquet"
    base = pd.read_parquet(feat_path)
    base = base.drop(columns=[c for c in per.columns if c != "session_id" and c in base.columns])
    merged = base.merge(per, on="session_id", how="left")
    merged["dns_periodicity_score"] = merged["dns_periodicity_score"].fillna(0.0)
    merged.to_parquet(feat_path, index=False)
    print(f"[5/5] {len(merged):,} sessions, {merged.shape[1]} colonnes -> {feat_path}")

    # verification : score de periodicite par famille (beaconing attendu eleve)
    chk = per.merge(se, on="session_id", how="left")
    top = chk.groupby(["label", "family"]).agg(
        n=("session_id", "size"),
        score_median=("dns_periodicity_score", "median"),
        periode_med_s=("dns_period_dominant", "median")).round(2)
    print("\nverification (score de periodicite par famille) :")
    print(top.sort_values("score_median", ascending=False).head(15).to_string())


if __name__ == "__main__":
    main()
