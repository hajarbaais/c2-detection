"""P7 - Tableau de bord Streamlit de la démo C2 (Aicha, Sprint 3).

Dépôt d'un PCAP -> tableau des sessions (alertes en rouge) -> détail + graphe SHAP -> chronomètre.

Lancement (depuis la racine du dépôt) :
    streamlit run p7_app_demo/app/app.py

Trois sources de données :
  - « Déposer un PCAP »   : C2Pipeline.run_pcap()            (attend extract/sessionize de P7.1)
  - « Sorties P2 »        : C2Pipeline.run_from_p2_output()  (données + modèles du Drive)
  - « Démo »              : données simulées, aucune dépendance (développement, secours en soutenance)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO = APP_DIR.parents[1]
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(REPO / "p7_app_demo"))

import dashboard_lib as lib  # noqa: E402

UPLOAD_DIR = Path(tempfile.gettempdir()) / "c2-uploads"
C2_COLOR, NORMAL_COLOR = "#c0392b", "#2e6da4"

st.set_page_config(page_title="Détection C2 — démo", page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------- pipeline (chargé une seule fois)
@st.cache_resource(show_spinner="Chargement des modèles…")
def get_pipeline():
    """C2Pipeline est instancié UNE fois par serveur (contrainte < 5 s : pas de rechargement des modèles)."""
    from pipeline import C2Pipeline
    return C2Pipeline()


def pipeline_or_error():
    try:
        return get_pipeline(), None
    except Exception as e:   # dépendance absente (sklearn, torch…), modèle manquant…
        return None, f"{type(e).__name__} : {e}"


def run_timed(fn, *args, **kwargs) -> dict:
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    res["wall_s"] = time.perf_counter() - t0
    return res


# ---------------------------------------------------------------- barre latérale : source
st.sidebar.title("🛡️ Détection C2")
source = st.sidebar.radio("Source des données",
                          ["Déposer un PCAP", "Sorties P2 déjà calculées", "Démo (données simulées)"],
                          index=2)

if source == "Déposer un PCAP":
    up = st.sidebar.file_uploader("Capture réseau", type=["pcap", "pcapng", "cap"],
                                  help=f"{lib.MAX_UPLOAD_MB} Mo max. Le fichier est seulement lu, jamais exécuté.")
    if up is not None and st.sidebar.button("Analyser", type="primary"):
        pipe, err = pipeline_or_error()
        if err:
            st.sidebar.error(f"Pipeline indisponible — {err}")
        else:
            path = None
            try:
                path = lib.save_upload(up.getvalue(), UPLOAD_DIR)
                with st.spinner("Analyse de la capture…"):
                    st.session_state["res"] = run_timed(pipe.run_pcap, path)
                    st.session_state["res_label"] = f"PCAP déposé ({up.size / 1e6:.1f} Mo)"
            except ValueError as e:
                st.sidebar.error(str(e))
            except NotImplementedError:
                st.sidebar.warning("L'analyse d'un PCAP n'est pas encore branchée dans le pipeline "
                                   "(P7.1 : extract / sessionize). Utiliser « Sorties P2 » ou « Démo ».")
            except Exception as e:
                st.sidebar.error(f"Erreur pendant l'analyse — {type(e).__name__} : {e}")
            finally:
                if path is not None:
                    path.unlink(missing_ok=True)   # rien ne reste sur le disque

elif source == "Sorties P2 déjà calculées":
    p2 = st.sidebar.text_input("Dossier output de P2", os.environ.get("C2_P2_DIR", ""),
                               help="Contient sessions_pcap.parquet (Drive de l'équipe).")
    limit = st.sidebar.number_input("Sessions (0 = toutes)", min_value=0, value=5000, step=1000)
    if st.sidebar.button("Analyser", type="primary"):
        if not p2 or not (Path(p2) / "sessions_pcap.parquet").exists():
            st.sidebar.error("sessions_pcap.parquet introuvable dans ce dossier.")
        else:
            pipe, err = pipeline_or_error()
            if err:
                st.sidebar.error(f"Pipeline indisponible — {err}")
            else:
                try:
                    with st.spinner("Analyse des sessions…"):
                        st.session_state["res"] = run_timed(pipe.run_from_p2_output, Path(p2),
                                                            limit=int(limit) or None)
                        st.session_state["res_label"] = f"Sorties P2 ({p2})"
                except Exception as e:
                    st.sidebar.error(f"Erreur pendant l'analyse — {type(e).__name__} : {e}")

else:
    n = st.sidebar.slider("Nombre de sessions simulées", 50, 2000, 300, step=50)
    if st.sidebar.button("Générer", type="primary") or "res" not in st.session_state:
        st.session_state["res"] = run_timed(lib.demo_result, n=n)
        st.session_state["res_label"] = "Démo (données simulées)"

res = st.session_state.get("res")
st.title("Détection de trafic Command & Control")
if res is None:
    st.info("Choisir une source dans la barre latérale puis cliquer sur « Analyser ».")
    st.stop()

if res.get("simulated"):
    st.warning("**Données simulées** : domaines fictifs (.example / .test), scores et explications inventés. "
               "Sert à présenter l'interface sans les données ni les modèles.", icon="🧪")
st.caption(f"Source : {st.session_state.get('res_label', '')}")

table = lib.sessions_table(res)
alerts = lib.alerts_by_session(res["alerts"])
timings = res["timings"]

# ---------------------------------------------------------------- chronomètre
st.subheader("⏱️ Chronomètre")
total = float(timings.get("total", res.get("wall_s", 0.0)))
ok = lib.latency_ok(timings)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sessions analysées", f"{len(table):,}".replace(",", " "))
c2.metric("Alertes", f"{int(table['alerte'].sum()):,}".replace(",", " "))
c3.metric("Temps de traitement", f"{total:.2f} s",
          delta=f"{'✓' if ok else '✗'} objectif < {lib.LATENCY_TARGET_S:.0f} s",
          delta_color="normal" if ok else "inverse")
c4.metric("Chargement des modèles", f"{timings.get('load_models', 0):.2f} s", help="Fait une seule fois au démarrage.")

tf = lib.timings_frame(timings)
if not tf.empty:
    chart = alt.Chart(tf).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("secondes:Q", title="secondes"),
        y=alt.Y("étape:N", sort=None, title=None),
        color=alt.condition("datum.une_fois", alt.value("#9aa5b1"), alt.value(NORMAL_COLOR)),
        tooltip=["étape", "secondes"],
    ).properties(height=34 * len(tf))
    st.altair_chart(chart, width="stretch")
    st.caption("En gris : fait une seule fois au démarrage, non compté dans le temps de traitement.")

# ---------------------------------------------------------------- tableau des sessions
st.subheader("📋 Sessions")
f1, f2, f3 = st.columns([1, 2, 2])
only_alerts = f1.toggle("Alertes seulement", value=False)
types = f2.multiselect("Type", sorted(table["type"].dropna().unique()), placeholder="Tous")
query = f3.text_input("Rechercher (destination ou session)", "")

view = table
if only_alerts:
    view = view[view["alerte"]]
if types:
    view = view[view["type"].isin(types)]
if query:
    q = query.lower()
    view = view[view["destination"].fillna("").str.lower().str.contains(q, regex=False)
                | view["session_id"].str.lower().str.contains(q, regex=False)]
view = view.reset_index(drop=True)

event = st.dataframe(
    view.style.apply(lib.row_style, axis=1),
    hide_index=True, width="stretch", height=380,
    on_select="rerun", selection_mode="single-row",
    column_config={
        "alerte": st.column_config.CheckboxColumn("alerte"),
        "iforest_score": st.column_config.NumberColumn("score IF", format="%.3f"),
        "ae_score": st.column_config.NumberColumn("score AE", format="%.3f"),
    },
)
st.caption(f"{len(view)} session(s) affichée(s) — lignes rouges = alertes. Cliquer une ligne pour le détail.")

# ---------------------------------------------------------------- détail + SHAP
st.subheader("🔍 Détail de la session")
rows = event.selection.rows if event and event.selection else []
if rows:
    selected = view.iloc[rows[0]]
elif view["alerte"].any():
    selected = view[view["alerte"]].iloc[0]
    st.caption("Aucune ligne sélectionnée : affichage de la première alerte.")
elif len(view):
    selected = view.iloc[0]
else:
    st.info("Aucune session avec ces filtres.")
    st.stop()

alert = alerts.get(selected["session_id"])
left, right = st.columns([2, 3])
with left:
    if selected["alerte"]:
        st.error(f"**ALERTE** — sévérité {selected['sévérité'] or '?'}", icon="🚨")
    else:
        st.success("Session normale : aucun détecteur n'a alerté.", icon="✅")
    info = {"Session": selected["session_id"], "Type": selected["type"],
            "Destination": selected["destination"] or "—", "Détecteurs": selected["détecteurs"] or "—"}
    for c in lib.SCORE_COLUMNS:
        if c in selected and pd.notna(selected[c]):
            info[c] = f"{selected[c]:.4f}"
    if alert:
        info["IP source (hachée)"] = alert.get("src_ip_hash", "—")
        info["MITRE ATT&CK"] = ", ".join(f"{m['technique_id']} {m['name']}" for m in alert.get("mitre") or []) or "—"
    st.table(pd.DataFrame({"": list(info.values())}, index=list(info.keys())))
    if alert:
        with st.expander("Alerte JSON (format SIEM, P7.3)"):
            st.json(alert)

with right:
    st.markdown("**Pourquoi ? — explication SHAP**")
    shap = lib.shap_frame(alert)
    if not selected["alerte"]:
        st.caption("Les explications sont calculées pour les sessions en alerte.")
    elif shap.empty:
        st.info("Explication SHAP pas encore disponible : `top_features` est vide "
                "(calcul SHAP en cours de développement, P7.2).")
    else:
        bars = alt.Chart(shap).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("impact:Q", title="impact sur la décision (SHAP)"),
            y=alt.Y("name:N", sort=None, title=None),
            color=alt.Color("sens:N", title=None,
                            scale=alt.Scale(domain=["vers C2", "vers normal"], range=[C2_COLOR, NORMAL_COLOR]),
                            legend=alt.Legend(orient="bottom")),
            tooltip=[alt.Tooltip("name:N", title="feature"), alt.Tooltip("value:Q", title="valeur"),
                     alt.Tooltip("impact:Q", title="impact", format="+.3f")],
        )
        rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#666").encode(x="x:Q")
        st.altair_chart((bars + rule).properties(height=40 * len(shap) + 40), width="stretch")
        st.caption("Barre rouge : la valeur de cette feature pousse vers « C2 ». Bleue : vers « normal ».")

# ---------------------------------------------------------------- export
if res["alerts"]:
    st.download_button("⬇️ Télécharger les alertes (JSON)", json.dumps(res["alerts"], indent=2, default=str),
                       file_name="alertes_c2.json", mime="application/json")
