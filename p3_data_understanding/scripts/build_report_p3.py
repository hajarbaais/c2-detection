r"""
P3 - Genere le rapport final (un seul fichier HTML) a partir des sorties deja produites.

Usage (depuis P2_livrables_v3\P2_livrables_v3), APRES quality.py et p3_suite.py :
    python build_report_p3.py

Lit    : p3_output\quality_report.txt, suite_report.txt, labels_a_verifier.csv, graphs\*.png
Ecrit  : p3_output\Rapport_P3.html   (s'ouvre dans le navigateur ; Ctrl+P -> "Enregistrer en PDF")

Les chiffres du resume viennent de ta propre execution de quality.py (25/10/2026).
Les parties "A completer" sont a remplir par toi apres lecture des graphiques.
"""
import base64
import html
import os
import sys

import pandas as pd

OUT = "p3_output"
need = ["quality_report.txt", "suite_report.txt", "labels_a_verifier.csv", "extra_report.txt"]
manque = [n for n in need if not os.path.exists(os.path.join(OUT, n))]
if manque:
    sys.exit("Fichiers manquants dans p3_output : " + ", ".join(manque) +
             "\nLance d'abord : python quality.py  puis  python p3_suite.py")


def lire(n):
    with open(os.path.join(OUT, n), encoding="utf-8") as f:
        return f.read()


def pre(t):
    return "<pre>" + html.escape(t) + "</pre>"


def img(nom, legende, regarder, comm="................................................"):
    p = os.path.join(OUT, "graphs", nom)
    if not os.path.exists(p):
        return f"<p><em>Graphique {nom} absent (installe matplotlib puis relance p3_suite.py).</em></p>"
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    return (f"<figure><img src='data:image/png;base64,{b64}' alt='{html.escape(legende)}'>"
            f"<figcaption><b>{html.escape(legende)}</b><br>Ce qu'il faut regarder : {html.escape(regarder)}"
            f"<br><span class='comm'><b>Commentaire de P3 :</b> {html.escape(comm)}</span>"
            f"</figcaption></figure>")


# ---------------------------------------------------------------- labels verifies
lab = pd.read_csv(os.path.join(OUT, "labels_a_verifier.csv"), encoding="utf-8-sig")
lab["verdict_manuel"] = lab["verdict_manuel"].fillna("").astype(str).str.strip().str.lower()


def norm(v):
    if v == "":
        return "non rempli"
    if v.startswith("v"):
        return "verifie"
    if v.startswith("d"):
        return "douteux"
    return "autre"


lab["verdict"] = lab["verdict_manuel"].map(norm)
tab = lab.groupby(["dataset", "verdict"]).size().unstack(fill_value=0)
nb_vide = int((lab["verdict"] == "non rempli").sum())
alerte = (f"<p class='warn'>{nb_vide} sessions sur {len(lab)} n'ont pas encore de verdict manuel. "
          "Remplis la colonne <code>verdict_manuel</code> de <code>labels_a_verifier.csv</code> "
          "(verifie / douteux) puis relance ce script.</p>") if nb_vide else ""
douteux = lab[lab["verdict"] == "douteux"][["session_id", "dataset", "family", "label", "verdict_auto", "commentaire"]]
bloc_douteux = (douteux.to_html(index=False, border=0, classes="t") if len(douteux)
                else "<p>Aucune session marquee douteuse.</p>")
auto = lab.assign(avis=lab["verdict_auto"].str.split(" :").str[0]).groupby(["dataset", "avis"]).size().unstack(fill_value=0)

# ---------------------------------------------------------------------- HTML
CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:30px auto;padding:0 16px;color:#222;line-height:1.5}
h1{border-bottom:3px solid #1f4e79;padding-bottom:6px;color:#1f4e79} h2{color:#1f4e79;margin-top:34px}
pre{background:#f4f4f4;padding:10px;overflow-x:auto;font-size:12px;border-radius:4px}
table.t{border-collapse:collapse;font-size:13px;margin:8px 0} table.t th,table.t td{border:1px solid #ccc;padding:4px 9px}
table.t th{background:#e8eef5} figure{margin:18px 0;border:1px solid #ddd;padding:10px;border-radius:4px}
figure img{max-width:100%} figcaption{font-size:13px;margin-top:6px}
.warn{background:#fff3cd;padding:8px 12px;border-left:4px solid #e0a800}
.ok{background:#e6f4ea;padding:8px 12px;border-left:4px solid #2e7d32}
.todo{color:#888} .comm{display:block;margin-top:4px;color:#1a1a1a} details{margin:8px 0} summary{cursor:pointer;color:#1f4e79}
"""

H = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Rapport P3</title>
<style>{CSS}</style></head><body>
<h1>Rapport P3 &mdash; Harmonisation, fusion &amp; qualite</h1>
<p>Projet &laquo; Detection de trafic C2 (DNS / HTTPS) &raquo; &middot; Controle de <code>sessions_pcap.parquet</code> (P2, v3)
&middot; 25/10/2026</p>

<h2>1. Objectif et role de P3</h2>
<p>P3 est le controle entre P2 (qui produit les sessions) et P4/P5/P6 (qui calculent les features et entrainent les modeles).
P3 <b>ne modifie pas</b> les donnees : il controle, explique et previent. Toute vraie erreur est signalee a P2.</p>

<h2>2. Decision : pas de fusion avec les CSV du CIC</h2>
<div class="warn">A faire valider par l'equipe / l'encadrant. Resultat de <code>check_csv.py</code> sur les vrais CSV :
........................</div>
<table class="t"><tr><th>Raison</th><th>Preuve</th></tr>
<tr><td>P2 a deja extrait les PCAP de DNS-EXF et DoHBrw</td><td>86 572 et 87 226 sessions dans sessions_pcap.parquet</td></tr>
<tr><td>Labels de P2 = labels de P1 et du CIC</td><td>compare_p1_labels.txt : 185 863 sessions identiques ; rapport P1 : 1 167 269 / 1 167 269 flux DoHBrw</td></tr>
<tr><td>Un CSV decrit des flux, pas des sessions</td><td>rapport P1 : des centaines de flux vers le meme site = une session</td></tr>
<tr><td>CSV <code>aggregate</code> = doubles comptes</td><td>rapport P1, sections 6 et 11</td></tr>
<tr><td>Features du CIC &ne; features du projet</td><td>P4 et P5 recalculent les leurs</td></tr>
<tr><td>Volume deja atteint</td><td>107 903 benins en test (seuil 100 000)</td></tr></table>

<h2>3. Resume des controles</h2>
<div class="ok">0 doublon &middot; 0 incoherence (duree negative, fin avant debut, label hors 0/1) &middot; 0 IP en clair &middot;
0 benin dans &laquo; unseen &raquo; &middot; 0 pcap present dans deux jeux (pas de fuite train/val/test).</div>
<table class="t"><tr><th>Jeu</th><th>Benin</th><th>C2</th></tr>
<tr><td>train</td><td>56 154</td><td>3 343</td></tr><tr><td>val</td><td>11 957</td><td>1 244</td></tr>
<tr><td>test</td><td><b>107 903</b></td><td>4 157</td></tr><tr><td>unseen</td><td>0</td><td>221</td></tr>
<tr><td><b>Total</b></td><td>176 014</td><td>8 965 (4,85 %)</td></tr></table>

<h2>4. Valeurs manquantes</h2>
<p>Normales : <code>orig_bytes</code>, <code>ja3</code>, <code>server_name</code>&hellip; vides a 100 % sur le DNS (champs HTTPS/TLS) ;
<code>parent_domain</code> vide sur HTTP/HTTPS ; <code>capture_start</code> vide pour 1 469 sessions Stratosphere (horodatages relatifs).
Rien n'est impute ni supprime.</p>
<p><b>A signaler :</b> sur HTTPS, <code>server_name</code> est vide pour 5,8 a 7 % des benins mais pour 100 % (CTU-13), 99,9 %
(Stratosphere C2) et 53 % (DoHBrw C2) des C2 ; <code>ja3</code> vide : 0 % (DoHBrw C2) a 76,5 % (CTU-13) selon le dataset.
Ces cases vides sont liees au label ou a la source : ne pas les utiliser comme features.</p>

<h2>5. Valeurs extremes</h2>
<table class="t"><tr><th>Constat</th><th>Verdict</th></tr>
<tr><td>jusqu'a 7 156 evenements par session DNS</td><td>reel : tunnels d'exfiltration (heavy video / exe)</td></tr>
<tr><td>duree maximale 3 600 s en DNS</td><td>normal : fenetre d'1 h de P2</td></tr>
<tr><td>duree 27,5 h en HTTPS (firefox-adguard)</td><td>plausible : la capture entiere dure 350 h</td></tr></table>
<p>Rien n'est supprime. 183 sessions sont marquees <code>is_extreme_events</code> (au-dessus du 99,9e centile de leur type).</p>

<h2>6. Desequilibre et pieges de labels</h2>
<ul>
<li>Seulement <b>42</b> vraies sessions de tunnel DNS (DNS-EXF) ; sur 2 302 sessions &laquo; DNS C2 &raquo;, <b>2 260</b> viennent de CTU-13
(DNS banal de la machine infectee, C2 par convention) &rarr; colonne <code>ctu13_broad_label</code> pour tester avec / sans.</li>
<li>Test : 7 sessions C2 DNS-EXF et 39 DoHBrw seulement &rarr; recall instable.</li>
<li>val = 1 244 C2, unseen = 221 C2 : petits jeux, donner des intervalles de confiance.</li>
<li>Sources a une seule classe (CTU-13, Stratosphere) : le modele peut apprendre la source plutot que le C2.</li>
<li>1 425 sessions Stratosphere a 6 evenements, tres repetitives (845 dans un seul pcap de test).</li>
<li>2 478 sessions Stratosphere avec certificat mitmproxy : ignorer certificat / auto-signe / JA3S.</li>
<li><code>no_tls_seen</code> : 6 % des benins mais 0 % / 22,5 % / 76,5 % des C2 selon le dataset &rarr; artefact de capture.</li></ul>

<h2>7. Verification des labels (echantillon, graine 42)</h2>
<p><b>Methode.</b> 354 sessions sont tirees au hasard (20 au plus par dataset et par famille, graine 42). Une regle automatique
compare chaque session a ce qu'on sait de sa source ; elle verifie la <i>coherence</i> du label, elle ne prouve pas qu'il est exact.
Seules les 15 sessions signalees &laquo; A REGARDER &raquo; ont ete examinees une par une avec des criteres explicites
(<code>p3_prefill_labels.py</code>) ; les 339 autres sont confirmees par la regle automatique uniquement.</p>
<p><b>Contexte des familles.</b> Les sessions C2 du dataset Stratosphere viennent de captures de machines infectees par
<b>Dridex</b> et <b>TrickBot</b>, deux malwares bancaires qui contactent regulierement leur serveur de commande (balisage) en HTTPS,
souvent sur des ports atypiques (4443, 4743, 449). Le label C2 vient donc du pcap (malware connu), pas d'une analyse de chaque
session : un rythme irregulier (CV de 1 a 1,9) n'invalide pas le label.</p>
{alerte}
<p>Avis automatique (coherence avec les regles de P1/P2) :</p>{auto.to_html(border=0, classes="t")}
<p>Verdict manuel de P3 :</p>{tab.to_html(border=0, classes="t")}
<p>Sessions jugees douteuses :</p>{bloc_douteux}

<h2>8. Graphiques de controle</h2>
{img("1_duree.png", "Duree des sessions (echelle log)", "les C2 ont-ils des durees tres differentes des benins ? Pic a 3 600 s = fenetre d'1 h.",
 "Les benins forment deux groupes (environ 0,1 s et 100 s) et les C2 s'etalent sur presque toute l'echelle : la duree seule ne separe pas les classes. "
 "Un petit groupe de C2 a environ 10 microsecondes (log10 = -5) est tres different du reste : sessions quasi instantanees, a signaler a P2 pour verification. "
 "Le graphique n'affiche que les durees strictement positives.")}
{img("2_n_events.png", "Nombre d'evenements par session (echelle log)", "les C2 ont-ils beaucoup plus d'evenements (tunnels DNS) ? Queue vers 7 000.",
 "La grande majorite des sessions, benignes comme C2, n'a que 1 ou 2 evenements. Les tunnels DNS a des milliers d'evenements existent mais sont tres rares, "
 "donc n_events seul detecterait peu de C2 : il faut des features de rythme (P4) et pas seulement des comptes.")}
{img("3_orig_bytes.png", "Octets envoyes, HTTPS/HTTP (echelle log)", "chevauchement benin / C2 : si fort, ce volume seul ne suffit pas.",
 "Les benins sont centres autour de 1 000 a 2 000 octets avec une longue queue. Les C2 sont peu nombreux, avec un pic net vers 400 octets : "
 "des sessions de taille quasi identique, typiques d'un balisage automatique. Ce pic est un bon indice mais il vient surtout de quelques familles (pas un signal general).")}
{img("4_resp_bytes.png", "Octets recus, HTTPS/HTTP (echelle log)", "meme lecture que le graphique precedent.",
 "Meme constat cote reception : les benins ont une large distribution et les C2 un pic etroit vers 2 800 octets (reponses de taille presque fixe du serveur). "
 "Le chevauchement reste fort en dehors de ce pic.")}
{img("5_desequilibre.png", "Benin / C2 par dataset (echelle log)", "sources a une seule classe et tres peu de C2 DNS-EXF.",
 "CTU-13 et Stratosphere-C2 ne contiennent que du C2, Stratosphere-normal que du benin : un modele peut deviner le label en reconnaissant la source "
 "(99,82 % d'accuracy avec le seul nom du dataset). DNS-EXF n'a que 42 C2 pour 86 530 benins et DoHBrw 300 pour 86 926 : a evaluer separement.")}
{img("6_taille_dns.png", "Taille des requetes DNS", "les tunnels ont-ils des requetes plus grandes ? (les noms sont tronques : on utilise size, pas len(query)).",
 "Les benins sont groupes entre 65 et 95 octets. Les C2 ont un pic massif a environ 307 octets, mais c'est un graphique par evenement : "
 "il est domine par quelques tunnels de plusieurs milliers de requetes (DNS-EXF heavy). Voir le graphique 8 pour la vue par session.")}
{img("7_entropie_dns.png", "Entropie des noms DNS, par session", "chevauchement fort benin / C2 : moyenne 2,08 contre 2,24 bits, l'entropie seule ne suffit pas (noms tronques).",
 "Aucune separation nette : les deux classes couvrent la meme plage (0 a 4 bits). Les C2 montrent des pics discrets (noms repetes) et un groupe a 0 bit, "
 "mais la moyenne est proche (2,24 contre 2,08). Comme P2 tronque les noms, l'entropie calculee est sous-estimee : P4 devra le savoir avant d'en faire une feature cle.")}
{img("8_taille_dns_session.png", "Taille des requetes DNS, par session", "le graphique 6 est dessine par evenement et domine par quelques tunnels de 7 000 requetes ; ici chaque session compte pour une.",
 "En vue par session, benins et C2 se superposent presque completement (65 a 95 octets) ; seul un petit groupe de C2 isole vers 307 octets se detache. "
 "La taille des requetes ne detecte donc que ces tunnels-la.")}

<h2>8b. Controles complementaires (p3_extra.py)</h2>
<table class="t"><tr><th>Controle</th><th>Resultat</th><th>Lecture</th></tr>
<tr><td>Doublons de contenu (35 colonnes, hors identifiants et horodatages)</td><td>0 session identique</td><td>Pas de doublon cache, meme sans compter session_id.</td></tr>
<tr><td>Machine presente dans plusieurs jeux</td><td>16 machines sur 54 (DNS-EXF, DoHBrw benin, CTU-13, Stratosphere normal)</td>
<td>Documente par P2 : l'unite de decoupage est le <b>pcap</b> (la machine seulement pour DoHBrw malveillant). Cela ecarte la fuite par fichier mais pas par machine : a faire valider (la fiche demande « une meme machine jamais des deux cotes »).</td></tr>
<tr><td>Raccourci de source</td><td>99,82 % d'accuracy en predisant le label majoritaire du dataset</td>
<td>CTU-13 et Stratosphere sont a une seule classe : un modele peut apprendre la source. Evaluer aussi par dataset.</td></tr></table>

<h2>9. Colonnes ajoutees par P3</h2>
<table class="t"><tr><th>Colonne</th><th>Sens</th><th>Usage</th></tr>
<tr><td>source</td><td>toujours pcap</td><td>information</td></tr>
<tr><td>no_tls_seen</td><td>HTTPS sans JA3 (6 774)</td><td>valeurs manquantes seulement, pas une feature</td></tr>
<tr><td>share_no_payload</td><td>part de connexions sans donnee</td><td>a verifier par dataset (section A1 du rapport detaille)</td></tr>
<tr><td>ctu13_broad_label</td><td>DNS banal CTU-13 etiquete C2 (2 260)</td><td>filtre avec / sans</td></tr>
<tr><td>is_extreme_events</td><td>au-dessus du p99,9</td><td>filtre optionnel</td></tr></table>
<p><b>Interdit comme features :</b> src_ip_hash, dst_ip_hash, session_key, qname_truncated, qname_truncated_share.</p>

<h2>10. Conclusion et recommandations</h2>
<div class="ok"><b>Conclusion.</b> La base <code>sessions_unified.parquet</code> (184 979 sessions, 176 014 benignes dont 107 903 en test) est
exploitable : aucun doublon (d'identifiant ou de contenu), aucune incoherence, aucune IP en clair, volume benin au-dessus du seuil de 100 000.
Les risques ne sont pas dans la proprete des donnees mais dans leur <b>composition</b> : sources a une seule classe, tres peu de vrais tunnels DNS,
labels CTU-13 par convention, et des signaux simples (taille, entropie, duree) qui se chevauchent fortement entre benin et C2.</div>
<table class="t"><tr><th>Pour</th><th>Recommandation</th></tr>
<tr><td><b>P2</b></td><td>Valider le decoupage : 16 machines sur 54 sont dans plusieurs jeux (l'unite est le pcap, la fiche demande la machine).
Verifier les sessions C2 d'environ 10 microsecondes. Envisager de ne pas tronquer les noms DNS (entropie sous-estimee).</td></tr>
<tr><td><b>P4</b></td><td>Ne pas compter sur l'entropie ou la taille des requetes seules : les distributions se chevauchent. Privilegier le rythme
(intervalles, periodicite). Ne pas utiliser <code>qname_truncated</code>, <code>src_ip_hash</code>, <code>dst_ip_hash</code>, <code>session_key</code>.</td></tr>
<tr><td><b>P5</b></td><td>Ne pas utiliser comme features <code>server_name</code> vide, <code>ja3</code> vide, <code>no_tls_seen</code> : ils dependent de la source, pas du C2.
Ignorer certificat, auto-signe et JA3S sur les 2 478 sessions Stratosphere interceptees. Calibrer le seuil de 0,1 % sur le test (107 903 benins).</td></tr>
<tr><td><b>P6</b></td><td>Evaluer <b>par dataset</b> en plus du global (99,82 % de label devinable par la source). Tester avec et sans <code>ctu13_broad_label</code>
(7 sessions DNS C2 en test sans CTU-13). Donner des intervalles de confiance : val = 1 244 C2, unseen = 221 C2. Surveiller les 1 425 sessions TrickBot/Dridex repetitives.</td></tr>
<tr><td><b>P7</b></td><td>Expliquer qu'une alerte DNS peut venir d'une convention de label (CTU-13) et non d'un vrai tunnel. Ne pas annoncer un recall global sans le detail par famille.</td></tr></table>

<h2>11. Limites</h2>
<ul><li>Le controle automatique des labels verifie la coherence avec les regles, il ne remplace pas le jugement humain.</li>
<li>Pour DoHBrw, la verification des labels est faible : benins et C2 sont tous en HTTPS sur le port 443, la regle ne les
distingue pas ; le label est herite du pcap.</li>
<li>Aucune donnee corrigee : toute anomalie reelle est renvoyee a P2.</li>
<li>Les resultats de recall/FPR dependent des lots P4 a P7.</li></ul>

<h2>Annexe : sorties detaillees</h2>
<details><summary>quality.py (controle qualite complet)</summary>{pre(lire("quality_report.txt"))}</details>
<details><summary>p3_suite.py (partie 6, labels, graphiques)</summary>{pre(lire("suite_report.txt"))}</details>
<details><summary>p3_extra.py (doublons de contenu, fuite de machines, raccourci de source, entropie)</summary>{pre(lire("extra_report.txt"))}</details>
</body></html>"""

dest = os.path.join(OUT, "Rapport_P3.html")
with open(dest, "w", encoding="utf-8") as f:
    f.write(H)
print("Ecrit :", dest)
