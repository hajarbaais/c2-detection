"""P5 (Imane) - Features TLS/HTTPS, scaler (train only), sequences de taille fixe.

Entrees (livrables P2 v3) : sessions_pcap.parquet, events.parquet, events_pkts.parquet
Sorties : features_tls.parquet, features_tls_scaled.parquet, scaler.pkl,
          features_seq.npz, seq_scaler.pkl

Usage : python p5_features.py --p2 path/to/P2_livrables_v3/output --out p5_output [--seq-len 64]
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TLS_TYPES = ["https", "http"]  # les sessions DNS sont gerees par P4 (Hajar)


# ----------------------------------------------------------------------------
# 1) Features tabulaires HTTPS / TLS (une ligne par session https/http)
# ----------------------------------------------------------------------------
def tls_version_num(s: pd.Series) -> pd.Series:
    """Convertit la version TLS en nombre (0 = inconnu). A verifier sur les vraies valeurs."""
    t = s.astype(str).str.lower()
    out = pd.Series(0, index=s.index, dtype="int8")
    for pat, val in [("1.0|0x0301|769", 1), ("1.1|0x0302|770", 2),
                     ("1.2|0x0303|771", 3), ("1.3|0x0304|772", 4)]:
        out[t.str.contains(pat, regex=True)] = val
    return out


def rarity_from_train(col: pd.Series, is_train: pd.Series) -> pd.Series:
    """-log(frequence dans le train). Valeur jamais vue en train -> rarete maximale."""
    freq = col[is_train].value_counts(normalize=True)
    max_rar = -np.log(1.0 / (is_train.sum() + 1))
    rar = -np.log(col.map(freq))
    rar = rar.fillna(max_rar)
    rar[col.isna()] = 0.0  # pas de JA3/JA4 (ex: HTTP clair) -> 0, avec drapeau has_*
    return rar


def build_tabular(sess: pd.DataFrame) -> pd.DataFrame:
    s = sess[sess["session_type"].isin(TLS_TYPES)].copy()
    is_train = s["split"].eq("train")

    f = pd.DataFrame(index=s.index)
    # --- volumetrie
    orig, resp = s["orig_bytes"].fillna(0), s["resp_bytes"].fillna(0)
    pk = (s["orig_pkts"].fillna(0) + s["resp_pkts"].fillna(0))
    f["log_orig_bytes"] = np.log1p(orig)
    f["log_resp_bytes"] = np.log1p(resp)
    f["up_down_ratio"] = np.log1p(orig) - np.log1p(resp)  # log du ratio montee/descente
    f["avg_pkt_size"] = (orig + resp) / pk.replace(0, np.nan)
    f["log_duration"] = np.log1p(s["duration"].clip(lower=0).fillna(0))
    f["log_n_conn"] = np.log1p(s["n_events"].fillna(0))
    f["log_n_conn_no_payload"] = np.log1p(s["n_conn_no_payload"].fillna(0))
    f["conn_rate"] = s["n_events"] / s["duration"].clip(lower=1)  # connexions / seconde

    # --- empreintes TLS
    f["has_ja3"] = s["ja3"].notna().astype("int8")
    f["ja3_rarity"] = rarity_from_train(s["ja3"], is_train)
    f["ja4_rarity"] = rarity_from_train(s["ja4"], is_train)
    f["tls_version"] = tls_version_num(s["tls_version"])
    # certificat : ignore si le certificat a ete remplace par mitmproxy (README 6.3)
    selfsig = s["cert_self_signed"].fillna(False).astype(bool)
    f["cert_self_signed"] = (selfsig & ~s["tls_intercepted"].fillna(False).astype(bool)).astype("int8")
    # JA3S volontairement NON utilise (fortement biaise par mitmproxy)

    # --- reputation : nb de machines distinctes qui contactent le domaine (TRAIN seulement)
    key = s["server_name"].fillna(s["dst_ip_hash"])  # fallback : IP hachee, seulement comme cle de comptage
    rep = s.loc[is_train].groupby(key[is_train])["src_ip_hash"].nunique()
    f["log_domain_reputation"] = np.log1p(key.map(rep).fillna(0))

    f = f.replace([np.inf, -np.inf], np.nan).fillna(0)
    meta = s[["session_id", "label", "family", "session_type", "split", "unseen_strict"]]
    return pd.concat([meta, f], axis=1).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2) Sequences : (N, L, F) avec masque. DNS -> evenements, HTTPS/HTTP -> paquets
# ----------------------------------------------------------------------------
SEQ_FEATS = ["log_dt", "log_size", "log_resp_size", "direction", "is_dns", "is_nxdomain"]


def to_seconds(ts: pd.Series) -> pd.Series:
    if np.issubdtype(ts.dtype, np.datetime64):
        return ts.astype("datetime64[ns]").astype("int64") / 1e9
    return ts.astype(float)


def build_steps(sess, events, pkts) -> pd.DataFrame:
    idx = pd.Series(np.arange(len(sess)), index=sess["session_id"].values)
    stype = sess.set_index("session_id")["session_type"]

    # DNS : un pas = une requete
    e = events[events["session_id"].map(stype).eq("dns")]
    a = pd.DataFrame({
        "i": e["session_id"].map(idx).values, "seq": e["seq"].values, "ts": to_seconds(e["ts"]).values,
        "size": e["size"].fillna(0).values, "resp_size": e["resp_size"].fillna(0).values,
        "direction": 0.0, "is_dns": 1.0, "is_nx": (e["rcode"] == 3).astype(float).values})

    # HTTPS/HTTP : un pas = un paquet avec charge utile (sinon on retombe sur les connexions)
    b = pd.DataFrame({
        "i": pkts["session_id"].map(idx).values, "seq": pkts["seq"].values, "ts": to_seconds(pkts["ts"]).values,
        "size": pkts["size"].fillna(0).values, "resp_size": 0.0,
        "direction": np.where(pkts["direction"].eq("up"), 1.0, -1.0), "is_dns": 0.0, "is_nx": 0.0})
    have_pkts = set(b["i"].dropna().astype(int))
    e2 = events[events["session_id"].map(stype).isin(TLS_TYPES)]
    e2 = e2[~e2["session_id"].map(idx).isin(have_pkts)]
    c = pd.DataFrame({
        "i": e2["session_id"].map(idx).values, "seq": e2["seq"].values, "ts": to_seconds(e2["ts"]).values,
        "size": e2["size"].fillna(0).values, "resp_size": e2["resp_size"].fillna(0).values,
        "direction": 0.0, "is_dns": 0.0, "is_nx": 0.0})
    return pd.concat([a, b, c], ignore_index=True).dropna(subset=["i"]).astype({"i": int})


def build_sequences(sess, events, pkts, L):
    st = build_steps(sess, events, pkts).sort_values(["i", "seq"], kind="stable").reset_index(drop=True)
    dt = st["ts"].diff().where(st["i"].eq(st["i"].shift()), 0.0).clip(lower=0).fillna(0)
    st["rank"] = st.groupby("i").cumcount()
    st["log_dt"], st["log_size"], st["log_resp_size"] = np.log1p(dt), np.log1p(st["size"]), np.log1p(st["resp_size"])
    st = st[st["rank"] < L]

    N, F = len(sess), len(SEQ_FEATS)
    X = np.zeros((N, L, F), dtype=np.float32)
    mask = np.zeros((N, L), dtype=bool)
    vals = st[["log_dt", "log_size", "log_resp_size", "direction", "is_dns", "is_nx"]].to_numpy(np.float32)
    X[st["i"].values, st["rank"].values] = vals
    mask[st["i"].values, st["rank"].values] = True
    return X, mask


def scale_sequences(X, mask, is_train):
    """Scaler sur les pas valides des sessions train, colonnes continues 0..2 seulement."""
    sc = StandardScaler().fit(X[is_train][:, :, :3][mask[is_train]])
    X[:, :, :3] = np.where(mask[:, :, None], sc.transform(X[:, :, :3].reshape(-1, 3)).reshape(len(X), -1, 3), 0)
    return X, sc


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2", required=True, help="dossier contenant sessions_pcap.parquet, events*.parquet")
    ap.add_argument("--out", default="p5_output")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--no-seq", action="store_true", help="ne calcule que features_tls + scaler (pas besoin de events*.parquet)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sess = pd.read_parquet(os.path.join(a.p2, "sessions_pcap.parquet")).reset_index(drop=True)

    # ---- features tabulaires + scaler (fit train only)
    tab = build_tabular(sess)
    feat_cols = [c for c in tab.columns if c not in
                 ["session_id", "label", "family", "session_type", "split", "unseen_strict"]]
    tab.to_parquet(os.path.join(a.out, "features_tls.parquet"), index=False)

    train = tab["split"].eq("train")
    scaler = StandardScaler().fit(tab.loc[train, feat_cols])
    joblib.dump({"scaler": scaler, "columns": feat_cols}, os.path.join(a.out, "scaler.pkl"))
    scaled = tab.copy()
    scaled[feat_cols] = scaler.transform(tab[feat_cols])
    scaled.to_parquet(os.path.join(a.out, "features_tls_scaled.parquet"), index=False)
    print(f"features_tls: {len(tab)} sessions, {len(feat_cols)} features")

    if a.no_seq:
        return

    # ---- sequences
    events = pd.read_parquet(os.path.join(a.p2, "events.parquet"),
                             columns=["session_id", "seq", "ts", "size", "resp_size", "rcode"])
    pkts = pd.read_parquet(os.path.join(a.p2, "events_pkts.parquet"),
                           columns=["session_id", "seq", "ts", "direction", "size"])
    # on ne garde que les L premiers pas de chaque session (economise beaucoup de RAM)
    events = events.groupby("session_id", sort=False).head(a.seq_len)
    pkts = pkts.groupby("session_id", sort=False).head(a.seq_len)
    print(f"events: {len(events)} lignes, pkts: {len(pkts)} lignes apres troncature")
    X, mask = build_sequences(sess, events, pkts, a.seq_len)
    is_train = sess["split"].eq("train").to_numpy()
    X, seq_sc = scale_sequences(X, mask, is_train)
    joblib.dump({"scaler": seq_sc, "features": SEQ_FEATS[:3]}, os.path.join(a.out, "seq_scaler.pkl"))
    np.savez_compressed(
        os.path.join(a.out, "features_seq.npz"),
        X=X, mask=mask, session_id=sess["session_id"].to_numpy(str),
        label=sess["label"].to_numpy(np.int8), split=sess["split"].to_numpy(str),
        session_type=sess["session_type"].to_numpy(str), feature_names=np.array(SEQ_FEATS))
    print(f"features_seq: X{X.shape}, {mask.sum(1).mean():.1f} pas valides en moyenne")


if __name__ == "__main__":
    main()
