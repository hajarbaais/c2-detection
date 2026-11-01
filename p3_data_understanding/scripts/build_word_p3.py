r"""
Genere le rapport Word du lot P3 : p3_output\Rapport_P3.docx

Usage (depuis P2_livrables_v3\P2_livrables_v3), apres quality.py, p3_suite.py et p3_extra.py :
    python build_word_p3.py

Necessite : python -m pip install python-docx
Dans Word : clic droit sur le sommaire -> "Mettre a jour les champs" pour afficher les numeros de page.
"""
import os
import sys

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUT = "p3_output"
G = os.path.join(OUT, "graphs")
DEST = os.path.join(OUT, "Rapport_P3.docx")

# ------------------------------------------------------------------ constantes
AUTEUR = "[Nom et prénom]"
DATE = "25 octobre 2026"
TEAL = RGBColor(0x0B, 0x6B, 0x78)
INK = RGBColor(0x12, 0x23, 0x2B)
GREY = RGBColor(0x4A, 0x5F, 0x68)
HEAD_FILL = "0B6B78"
ALT_FILL = "EEF4F5"
NOTE_FILL = "FBE9CC"
OK_FILL = "E0F1E7"

manque = [n for n in ("1_duree.png", "2_n_events.png", "3_orig_bytes.png", "4_resp_bytes.png",
                      "5_desequilibre.png", "6_taille_dns.png", "7_entropie_dns.png",
                      "8_taille_dns_session.png") if not os.path.exists(os.path.join(G, n))]
if manque:
    sys.exit("Graphiques manquants dans p3_output\\graphs : " + ", ".join(manque) +
             "\nLance d'abord : python p3_suite.py  puis  python p3_extra.py")

doc = Document()

# ------------------------------------------------------------------ page et styles
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.3)
sec.top_margin, sec.bottom_margin = Cm(2.2), Cm(2.2)

st = doc.styles
n = st["Normal"]
n.font.name = "Calibri"
n.font.size = Pt(11)
n.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
n.paragraph_format.space_after = Pt(6)
n.paragraph_format.line_spacing = 1.15

for name, size, before, after in (("Heading 1", 18, 20, 8), ("Heading 2", 13.5, 14, 4), ("Heading 3", 11.5, 10, 3)):
    h = st[name]
    h.font.name = "Calibri"
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    h.font.size = Pt(size)
    h.font.bold = True
    h.font.color.rgb = TEAL if name != "Heading 3" else INK
    h.paragraph_format.space_before = Pt(before)
    h.paragraph_format.space_after = Pt(after)
    h.paragraph_format.keep_with_next = True


# ------------------------------------------------------------------ outils
def fill(cell, hex_):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_)
    tcPr.append(shd)


def borders(table, color="C9D6D9"):
    tblPr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:space"), "0")
        e.set(qn("w:color"), color)
        b.append(e)
    tblPr.append(b)


def cell_margins(table, top=60, bottom=60, left=90, right=90):
    tblPr = table._tbl.tblPr
    m = OxmlElement("w:tblCellMar")
    for k, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        e = OxmlElement(f"w:{k}")
        e.set(qn("w:w"), str(v))
        e.set(qn("w:type"), "dxa")
        m.append(e)
    tblPr.append(m)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    e = OxmlElement("w:tblHeader")
    e.set(qn("w:val"), "true")
    trPr.append(e)


def no_split(row):
    trPr = row._tr.get_or_add_trPr()
    e = OxmlElement("w:cantSplit")
    e.set(qn("w:val"), "true")
    trPr.append(e)


def run_fmt(run, size=None, bold=None, italic=None, color=None):
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def para(text="", bold=False, italic=False, size=None, color=None, align=None, after=None, style=None):
    p = doc.add_paragraph(style=style)
    if text:
        r = p.add_run(text)
        run_fmt(r, size, bold, italic, color)
    if align:
        p.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def rich(parts, style=None, after=None):
    """parts = liste de (texte, gras). Permet un paragraphe avec des passages en gras."""
    p = doc.add_paragraph(style=style)
    for t, b in parts:
        r = p.add_run(t)
        r.bold = b
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def bullets(items):
    for it in items:
        if isinstance(it, tuple):
            p = doc.add_paragraph(style="List Bullet")
            r = p.add_run(it[0]); r.bold = True
            p.add_run(it[1])
        else:
            doc.add_paragraph(it, style="List Bullet")
        doc.paragraphs[-1].paragraph_format.space_after = Pt(2)


def numbered(items):
    for it in items:
        p = doc.add_paragraph(style="List Number")
        if isinstance(it, tuple):
            r = p.add_run(it[0]); r.bold = True
            p.add_run(it[1])
        else:
            p.add_run(it)
        p.paragraph_format.space_after = Pt(2)


def table(headers, rows, widths, num_cols=(), font=9.5, total_row=False):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    borders(t)
    cell_margins(t)
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.width = Cm(widths[i])
        fill(c, HEAD_FILL)
        p = c.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(h)
        run_fmt(r, font, True, color=RGBColor(0xFF, 0xFF, 0xFF))
        if i in num_cols:
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    repeat_header(t.rows[0])
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        no_split(t.rows[-1])
        last = total_row and ri == len(rows) - 1
        for i, v in enumerate(row):
            c = cells[i]
            c.width = Cm(widths[i])
            if last:
                fill(c, "DCEFF1")
            elif ri % 2 == 1:
                fill(c, ALT_FILL)
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            r = p.add_run(str(v))
            run_fmt(r, font, bold=last)
            if i in num_cols:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def caption(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    run_fmt(r, 9.5, italic=True, color=GREY)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_together = True
    return p


def box(text_parts, fill_hex=NOTE_FILL):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell_margins(t, 100, 100, 160, 160)
    c = t.rows[0].cells[0]
    c.width = Cm(16.4)
    fill(c, fill_hex)
    p = c.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    for txt, b in text_parts:
        r = p.add_run(txt)
        r.bold = b
        r.font.size = Pt(10.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def figure(fname, num, titre, commentaire, conclusion):
    doc.add_picture(os.path.join(G, fname), width=Cm(14.2))
    pic = doc.paragraphs[-1]
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic.paragraph_format.keep_with_next = True
    pic.paragraph_format.space_after = Pt(2)
    cp = caption(f"Figure {num} : {titre}")
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp.paragraph_format.keep_with_next = True
    para(commentaire, after=3)
    rich([("Lecture : ", True), (conclusion, False)], after=10)


def field(paragraph, instr):
    r = paragraph.add_run()
    for kind, text in (("begin", None), (None, instr), ("separate", None), ("txt", "1"), ("end", None)):
        if kind in ("begin", "separate", "end"):
            e = OxmlElement("w:fldChar")
            e.set(qn("w:fldCharType"), kind)
            r._r.append(e)
        elif kind == "txt":
            t = OxmlElement("w:t"); t.text = text
            r._r.append(t)
        else:
            e = OxmlElement("w:instrText")
            e.set(qn("xml:space"), "preserve")
            e.text = text
            r._r.append(e)


# ------------------------------------------------------------------ pied de page
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fp.add_run("Détection de trafic C2 (DNS / HTTPS) · Rapport du lot P3 · page ")
run_fmt(r, 9, color=GREY)
field(fp, "PAGE")
for rr in fp.runs:
    rr.font.size = Pt(9)
    rr.font.color.rgb = GREY

# ================================================================== PAGE DE GARDE
for _ in range(5):
    para("", after=14)
para("PROJET DE FIN D'ÉTUDES · CRISP-DM", bold=True, size=11, color=TEAL, after=10)
para("Détection de trafic C2 (DNS / HTTPS)", bold=True, size=28, color=INK, after=4)
para("Rapport du lot P3", bold=True, size=20, color=TEAL, after=10)
para("Contrôle qualité et analyse exploratoire de la base de sessions", size=14, color=GREY, after=40)
p = para("", after=2)
r = p.add_run("Réalisé par : "); run_fmt(r, 11, True)
r = p.add_run(AUTEUR); run_fmt(r, 11)
p = para("", after=2)
r = p.add_run("Date : "); run_fmt(r, 11, True)
r = p.add_run(DATE); run_fmt(r, 11)
p = para("", after=2)
r = p.add_run("Phase CRISP-DM : "); run_fmt(r, 11, True)
r = p.add_run("Data Understanding / Data Preparation"); run_fmt(r, 11)
doc.add_page_break()

# ================================================================== SOMMAIRE
para("Sommaire", bold=True, size=18, color=TEAL, after=8)
p = doc.add_paragraph()
field(p, 'TOC \\o "1-2" \\h \\z \\u')
para("Si le sommaire est vide ou sans numéros de page : clic droit dessus, puis « Mettre à jour les champs ».",
     italic=True, size=9, color=GREY)
doc.add_page_break()

# ================================================================== 1. RÉSUMÉ
doc.add_heading("1. Résumé", 1)
para("Dans ce rapport, je présente le travail réalisé sur la base de sessions issue des captures réseau (PCAP) "
     "du projet de détection de trafic C2. Mon rôle a été de contrôler la qualité de cette base avant qu'elle soit "
     "utilisée pour calculer les features et entraîner les modèles.")
para("La base contient 184 979 sessions : 176 014 sessions normales (bénignes) et 8 965 sessions malveillantes (C2), "
     "soit 4,85 % de C2. Le jeu de test contient 107 903 sessions bénignes, ce qui dépasse le seuil de 100 000 fixé pour "
     "mesurer un taux de fausses alertes inférieur à 0,1 %.")
box([("Conclusion principale : ", True),
     ("la base est propre et exploitable. Je n'ai trouvé aucun doublon, aucune valeur impossible, aucune adresse IP en clair "
      "et aucune fuite de fichier entre les jeux d'entraînement, de validation et de test. Les risques identifiés tiennent à la "
      "composition des données : peu de vrais tunnels DNS, des sources qui ne contiennent qu'une seule classe, et des mesures "
      "simples (durée, taille, entropie) qui séparent mal le trafic normal du trafic malveillant.", False)], OK_FILL)
para("Les principales actions que j'ai menées sont les suivantes :")
bullets([
    "contrôle de l'intégrité de la base et ajout de 5 colonnes de contrôle pour les lots suivants ;",
    "détection des doublons, des valeurs incohérentes, des valeurs manquantes et des valeurs extrêmes ;",
    "vérification de l'anonymisation des adresses IP ;",
    "mesure du déséquilibre entre classes et du volume de sessions bénignes ;",
    "production de 8 graphiques d'analyse exploratoire ;",
    "vérification de la fiabilité des labels sur un échantillon de 354 sessions ;",
    "rédaction de recommandations pour les lots P2, P4, P5, P6 et P7.",
])

# ================================================================== 2. CONTEXTE
doc.add_heading("2. Contexte et objectifs", 1)
doc.add_heading("2.1 Le projet", 2)
para("Le projet vise à détecter du trafic de commande et contrôle (C2) dans des échanges DNS et HTTPS, en n'utilisant que "
     "les métadonnées du réseau (jamais le contenu déchiffré). Les objectifs de réussite sont un recall d'au moins 98 %, un taux "
     "de fausses alertes inférieur à 0,1 % sur au moins 100 000 sessions bénignes, et une alerte en moins de 5 secondes.")
doc.add_heading("2.2 Place de mon travail dans la chaîne", 2)
para("Le projet est découpé en sept lots. Les captures réseau sont d'abord rangées (P1), puis transformées en sessions "
     "labellisées (P2). Mon lot (P3) intervient ensuite : il contrôle la qualité de la base de sessions avant que les features "
     "soient calculées (P4, P5), que les modèles soient entraînés (P6) et que la démonstration soit construite (P7).")
doc.add_heading("2.3 Objectifs de mon travail", 2)
numbered([
    "Garantir que la base de sessions est cohérente, sans doublon et sans fuite de données personnelles.",
    "Repérer et documenter les valeurs manquantes et les valeurs extrêmes.",
    "Mesurer le déséquilibre entre trafic normal et trafic malveillant, et vérifier le volume de sessions bénignes.",
    "Vérifier la fiabilité des labels sur un échantillon.",
    "Produire des graphiques d'exploration pour comprendre les données.",
    "Signaler aux autres lots les pièges à éviter.",
])

# ================================================================== 3. DONNÉES
doc.add_heading("3. Données d'entrée", 1)
para("La base de départ est le fichier sessions_pcap.parquet, produit par le lot P2 à partir des captures PCAP. "
     "Elle contient 184 979 sessions décrites par 36 colonnes (identifiants, type de session, source, label, famille, durée, "
     "nombre d'événements, volumes, empreintes TLS, jeu d'appartenance, etc.). Les événements détaillés de chaque session sont dans un "
     "second fichier, events.parquet.")
para("Une session regroupe les échanges entre une machine et un service. Elle est de type DNS (demandes de noms de sites), "
     "HTTP ou HTTPS. Chaque session porte un label : 0 pour le trafic normal, 1 pour le trafic C2.")
doc.add_heading("3.1 Sources de données", 2)
table(["Source", "Bénin", "C2", "Contenu"],
      [["cic-bell-dns-exf-2021", "86 530", "42", "DNS : exfiltration de données par tunnel"],
       ["cira-cic-dohbrw-2020", "86 926", "300", "HTTPS : DNS sur HTTPS, bénin et malveillant"],
       ["ctu13", "0", "4 906", "Botnets réels (DNS, HTTP, HTTPS)"],
       ["stratosphere-https-c2", "0", "3 717", "HTTPS de machines infectées (Dridex, TrickBot, Cobalt Strike)"],
       ["stratosphere-normal", "2 558", "0", "Machine propre"],
       ["Total", "176 014", "8 965", "184 979 sessions"]],
      [4.4, 2.0, 2.0, 8.0], num_cols=(1, 2), total_row=True)
caption("Tableau 1 : répartition des sessions par source et par label.")
doc.add_heading("3.2 Types de sessions", 2)
table(["Type", "Bénin", "C2", "Total"],
      [["DNS", "87 319", "2 302", "89 621"],
       ["HTTP", "262", "2 565", "2 827"],
       ["HTTPS", "88 433", "4 098", "92 531"]],
      [4.0, 3.0, 3.0, 3.0], num_cols=(1, 2, 3))
caption("Tableau 2 : répartition des sessions par type.")
doc.add_heading("3.3 Jeux d'entraînement, de validation, de test", 2)
para("Les sessions sont réparties en quatre jeux : train (entraînement), val (réglages), test (mesure finale) et unseen "
     "(familles de malwares jamais vues, pour tester la généralisation). Chaque fichier PCAP appartient à un seul jeu, "
     "afin d'éviter qu'un même enregistrement serve à entraîner et à tester.")

# ================================================================== 4. DÉMARCHE
doc.add_heading("4. Démarche et outils", 1)
para("J'ai travaillé en Python avec les bibliothèques pandas (analyse de tableaux) et matplotlib (graphiques). "
     "Chaque contrôle est programmé dans un script, ce qui rend l'analyse reproductible : en relançant les scripts dans l'ordre, "
     "on obtient les mêmes résultats. Le tirage aléatoire de l'échantillon de labels utilise une graine fixe (42).")
table(["Script", "Rôle", "Résultats produits"],
      [["quality.py", "Contrôle qualité complet, ajout des 5 colonnes", "sessions_unified.parquet, quality_report.txt"],
       ["p3_suite.py", "Analyse des colonnes ajoutées, échantillon de labels, graphiques 1 à 6",
        "suite_report.txt, labels_a_verifier.csv, graphs/"],
       ["p3_extra.py", "Doublons de contenu, machines communes aux jeux, risque de raccourci, graphiques 7 et 8",
        "extra_report.txt, graphs/"],
       ["p3_prefill_labels.py", "Application des critères de vérification des labels", "labels_a_verifier.csv"],
       ["build_report_p3.py", "Rapport HTML de synthèse", "Rapport_P3.html"],
       ["build_word_p3.py", "Rapport Word (ce document)", "Rapport_P3.docx"]],
      [3.6, 7.0, 5.8])
caption("Tableau 3 : scripts développés et résultats produits.")
para("Ordre d'exécution : quality.py, puis p3_suite.py, puis p3_extra.py, puis p3_prefill_labels.py, puis les scripts de rapport.")

# ================================================================== 5. INTÉGRITÉ
doc.add_heading("5. Contrôle de l'intégrité de la base", 1)
para("Avant d'analyser les données, j'ai vérifié que la base de travail (sessions_unified.parquet) reprend fidèlement "
     "celle livrée par P2 et qu'elle est cohérente avec les fichiers associés.")
table(["Contrôle", "Résultat"],
      [["Nombre de lignes identique à celui du fichier de P2", "Conforme : 184 979 des deux côtés"],
       ["Colonnes d'origine (36) inchangées, même ordre de sessions", "Conforme : aucune colonne modifiée"],
       ["Nombre d'événements (n_events) égal au nombre réel d'événements", "Conforme : 184 979 sessions sur 184 979"],
       ["Sessions sans événements, événements sans session", "Conforme : 0 et 0"],
       ["Listes train, val, test, unseen identiques à la colonne split", "Conforme : 4 jeux identiques, aucune session sans jeu"],
       ["Adresses IP anonymisées", "Conforme : code de 16 caractères pour toutes les sessions, aucune valeur vide"],
       ["Cohérence entre famille et label", "Conforme : toute famille « benign » a le label 0, et inversement"]],
      [8.4, 8.0])
caption("Tableau 4 : contrôles d'intégrité de la base.")
doc.add_heading("5.1 Colonnes ajoutées", 2)
para("J'ai ajouté cinq colonnes de contrôle, calculées à partir des colonnes existantes. J'ai recalculé les trois colonnes "
     "dérivées par une méthode indépendante, avec un résultat identique.")
table(["Colonne", "Définition", "Usage"],
      [["source", "Origine des sessions (pcap pour toutes)", "Information"],
       ["no_tls_seen", "Session HTTPS sans empreinte JA3 : 6 774 sessions", "Repérer les captures sans poignée de main TLS visible"],
       ["share_no_payload", "Part des connexions sans données (HTTP et HTTPS)", "Distinguer les scans et connexions vides"],
       ["ctu13_broad_label", "Session DNS de CTU-13 étiquetée C2 : 2 260 sessions", "Tester les modèles avec et sans ces sessions"],
       ["is_extreme_events", "Nombre d'événements au-dessus du 99,9e centile de son type : 183 sessions", "Filtre optionnel"]],
      [3.6, 7.2, 5.6])
caption("Tableau 5 : colonnes ajoutées.")
doc.add_heading("5.2 Cohérence avec les labels du lot P1", 2)
para("Les labels de la base concordent avec ceux de la table de correspondance du lot P1 pour 185 863 sessions. "
     "Les 218 différences ont toutes la même origine : pour les captures d'attaque de DNS-EXF, 203 sessions de bruit "
     "Windows (microsoft.com, windowsupdate.com, etc.) ont été retirées au lieu d'être comptées comme C2, et 15 sessions vers le "
     "domaine d'exfiltration, trouvées dans une capture bénigne, ont été retirées. Ce choix rend les labels plus justes. "
     "Il explique aussi le faible nombre de sessions C2 en DNS-EXF (42).")

# ================================================================== 6. QUALITÉ
doc.add_heading("6. Contrôle qualité", 1)
doc.add_heading("6.1 Doublons", 2)
para("J'ai recherché les doublons à trois niveaux : l'identifiant de session, la ligne entière, et le contenu (35 colonnes "
     "comparées, en ignorant les identifiants et les horodatages). Ce dernier test permet de détecter des sessions identiques "
     "qui n'ont pas le même identifiant.")
table(["Test", "Résultat", "Attendu"],
      [["Identifiants de session en double", "0", "0"],
       ["Lignes entièrement identiques", "0", "0"],
       ["Sessions au contenu identique à une autre", "0", "0"]],
      [9.0, 3.7, 3.7], num_cols=(1, 2))
caption("Tableau 6 : recherche de doublons.")
doc.add_heading("6.2 Cohérence des valeurs", 2)
para("Aucune valeur impossible n'a été trouvée : pas de label autre que 0 ou 1, pas de durée négative, pas de fin de session "
     "antérieure à son début, pas de session sans événement, pas de session bénigne dans le jeu unseen, et aucun fichier PCAP "
     "présent dans plusieurs jeux.")
doc.add_heading("6.3 Anonymisation des adresses IP", 2)
para("Pour respecter la conformité (loi 09-08 et RGPD), les adresses IP ne doivent pas apparaître en clair. J'ai vérifié "
     "l'absence de colonnes d'IP en clair (src_ip, dst_ip) et contrôlé que chaque session porte un code anonymisé de "
     "16 caractères pour la machine source et la destination, sans valeur vide. Ces codes servent à identifier une machine de "
     "façon cohérente sans révéler son adresse.")
doc.add_heading("6.4 Valeurs manquantes", 2)
para("Les cases vides sont, pour la plupart, normales : elles dépendent du type de session. Je n'ai rien imputé ni supprimé.")
table(["Type", "server_name", "ja3", "cert_self_signed", "orig_bytes"],
      [["DNS", "100 %", "100 %", "100 %", "100 %"],
       ["HTTP", "100 %", "100 %", "100 %", "0 %"],
       ["HTTPS", "11,0 %", "7,3 %", "35,4 %", "0 %"]],
      [3.0, 3.4, 3.0, 3.8, 3.2], num_cols=(1, 2, 3, 4))
caption("Tableau 7 : part de valeurs manquantes par type de session.")
para("Le DNS n'a ni octets envoyés ni empreinte TLS, et le HTTP n'a pas de nom de site (SNI) : ces vides sont attendus. "
     "La colonne capture_start est vide pour 1 469 sessions Stratosphere, dont les horodatages sont relatifs.")
rich([("Point d'attention : ", True),
      ("sur HTTPS, le nom de site (server_name) manque pour 100 % des sessions C2 de CTU-13, 99,9 % de celles de Stratosphere et "
       "53 % de celles de DoHBrw, contre 5,8 à 7 % des sessions bénignes. L'empreinte JA3 manque pour 76,5 % des C2 de CTU-13, "
       "22,5 % de ceux de Stratosphere et 0 % de ceux de DoHBrw. Ces cases vides dépendent donc de la source et non du "
       "comportement : elles ne doivent pas servir de critère de détection.", False)])
doc.add_heading("6.5 Valeurs extrêmes", 2)
para("J'ai repéré les valeurs extrêmes sans les supprimer, car elles correspondent à des comportements réels.")
table(["Constat", "Interprétation"],
      [["Jusqu'à 7 156 événements dans une session DNS", "Réel : tunnels d'exfiltration (DNS-EXF, familles « heavy »)"],
       ["Durée maximale de 3 600 s en DNS", "Normal : fenêtre d'une heure appliquée par le lot P2"],
       ["Session HTTPS de 27,5 heures", "Plausible : la capture complète dure environ 350 heures"],
       ["183 sessions au-dessus du 99,9e centile de leur type", "Marquées par la colonne is_extreme_events, conservées"]],
      [7.4, 9.0])
caption("Tableau 8 : valeurs extrêmes observées.")

# ================================================================== 7. DÉSÉQUILIBRE
doc.add_heading("7. Déséquilibre et volume de sessions", 1)
para("Le trafic malveillant représente 4,85 % des sessions. Ce déséquilibre est normal pour de la détection d'intrusion, "
     "mais il impose de choisir des métriques adaptées (recall, taux de fausses alertes) plutôt que la simple exactitude.")
table(["Jeu", "Bénin", "C2", "Remarque"],
      [["train", "56 154", "3 343", ""],
       ["val", "11 957", "1 244", "Petit jeu C2 : métriques instables"],
       ["test", "107 903", "4 157", "Seuil de 100 000 bénins atteint"],
       ["unseen", "0", "221", "Familles jamais vues, petit jeu"],
       ["Total", "176 014", "8 965", "4,85 % de C2"]],
      [3.0, 3.0, 3.0, 7.4], num_cols=(1, 2), total_row=True)
caption("Tableau 9 : répartition par jeu.")
para("Deux risques en découlent. D'abord, le très faible nombre de vrais tunnels DNS : seulement 42 sessions C2 en DNS-EXF "
     "(7 dans le jeu de test). Ensuite, 2 260 des 2 302 sessions DNS étiquetées C2 viennent de CTU-13, où toute la machine infectée "
     "est considérée comme C2, y compris son trafic DNS ordinaire (google.com, yahoo.com, etc.). Dans le jeu de test, il reste "
     "1 811 sessions DNS C2 avec CTU-13, mais seulement 7 sans. La colonne ctu13_broad_label permet de tester avec et sans.")
rich([("Risque de raccourci : ", True),
      ("en prédisant simplement le label majoritaire de la source, on obtient 99,82 % d'exactitude. CTU-13 et Stratosphere "
       "ne contiennent qu'une seule classe : un modèle pourrait apprendre à reconnaître la source au lieu du comportement malveillant. "
       "L'évaluation doit donc aussi être faite source par source.", False)])

# ================================================================== 8. GRAPHIQUES
doc.add_heading("8. Analyse exploratoire", 1)
para("J'ai produit huit graphiques. Sur tous, le bleu représente le trafic normal et le rouge le trafic malveillant. "
     "Sur les axes en échelle logarithmique, chaque pas vers la droite multiplie la valeur par 10.")
figure("1_duree.png", 1, "durée des sessions (échelle log)",
       "Les sessions normales forment deux groupes, autour de 0,1 seconde et de 100 secondes. Les sessions C2 s'étalent sur "
       "presque toute l'échelle. Un petit groupe de sessions C2 dure environ 10 microsecondes.",
       "la durée seule ne sépare pas les classes. Le groupe de 10 microsecondes est atypique et sera signalé au lot P2.")
figure("2_n_events.png", 2, "nombre d'événements par session (échelle log)",
       "La grande majorité des sessions, normales comme malveillantes, comptent 1 ou 2 événements.",
       "les tunnels à plusieurs milliers d'événements sont trop rares pour que ce critère suffise. Des mesures de rythme sont nécessaires.")
figure("3_orig_bytes.png", 3, "octets envoyés en HTTPS et HTTP (échelle log)",
       "Les sessions normales sont centrées autour de 1 000 à 2 000 octets, avec une longue traîne. Les sessions C2 présentent un pic "
       "étroit autour de 400 octets (valeur estimée sur le graphique).",
       "ce pic traduit des messages de taille presque constante, typiques d'un balisage automatique, mais il provient de quelques familles seulement.")
figure("4_resp_bytes.png", 4, "octets reçus en HTTPS et HTTP (échelle log)",
       "Même constat côté réponses : large distribution pour le trafic normal, pic étroit autour de 2 800 octets pour le C2 "
       "(valeur estimée).",
       "en dehors de ce pic, les deux classes se chevauchent fortement.")
figure("5_desequilibre.png", 5, "sessions bénignes et C2 par source (échelle log)",
       "Trois sources ne contiennent qu'une seule classe : Stratosphere-normal (bénin), Stratosphere-C2 et CTU-13 (C2). "
       "DNS-EXF ne compte que 42 sessions C2 face à 86 530 sessions bénignes.",
       "le risque est qu'un modèle reconnaisse la source plutôt que le comportement.")
figure("6_taille_dns.png", 6, "taille des requêtes DNS, une valeur par événement",
       "Les requêtes normales sont groupées entre 65 et 95 octets. Les requêtes C2 présentent un pic massif autour de 307 octets.",
       "ce graphique est dominé par quelques tunnels de plus de 7 000 requêtes chacun : il est à lire avec la figure 8.")
figure("7_entropie_dns.png", 7, "entropie des noms DNS, moyenne par session",
       "L'entropie mesure le caractère aléatoire d'un nom. Les deux classes couvrent la même plage (0 à 4 bits). La moyenne est de "
       "2,24 bits pour le C2 et de 2,08 bits pour le trafic normal.",
       "l'entropie seule ne sépare pas les classes. Les noms DNS étant tronqués dans les captures, la valeur est de plus sous-estimée.")
figure("8_taille_dns_session.png", 8, "taille des requêtes DNS, une valeur par session",
       "Avec une valeur par session, les deux distributions se superposent presque complètement. Seul un petit groupe de sessions C2 "
       "se détache autour de 307 octets.",
       "la taille des requêtes ne détecte que ces tunnels-là.")
box([("Enseignement général : ", True),
     ("aucune mesure simple ne sépare nettement le trafic normal du trafic malveillant. Les lots suivants devront s'appuyer sur "
      "des signaux plus fins, comme la régularité du rythme des connexions.", False)], OK_FILL)

# ================================================================== 9. LABELS
doc.add_heading("9. Vérification des labels", 1)
doc.add_heading("9.1 Méthode", 2)
para("Un label est la réponse attendue pour une session (0 ou 1). Comme il vient de la source des données, je ne peux pas le "
     "prouver de façon absolue : je vérifie sa cohérence avec ce que la session fait réellement.")
para("J'ai tiré au hasard 354 sessions (au plus 20 par source et par famille, graine 42). Une règle automatique compare chaque "
     "session à ce que l'on sait de sa source. Les cas que la règle signalait comme douteux ont ensuite été examinés un par un, avec "
     "des critères écrits dans le script p3_prefill_labels.py.")
doc.add_heading("9.2 Force du contrôle selon la source", 2)
table(["Source", "Ce qui est contrôlé", "Force"],
      [["DNS-EXF", "Le C2 doit viser le domaine d'exfiltration cicresearch.ca, le bénin ne doit pas. Conforme pour les 42 sessions C2.", "Forte"],
       ["Stratosphere", "Le label vient du PCAP (malware connu ou machine propre). Les sites bénins sont des services courants.", "Moyenne"],
       ["CTU-13", "Toute la machine infectée est C2 par convention des auteurs, y compris son DNS ordinaire.", "Moyenne"],
       ["DoHBrw", "Bénin et C2 sont tous en HTTPS sur le port 443 : la règle ne les distingue pas, le label est hérité du PCAP.", "Faible"]],
      [3.0, 11.2, 2.2])
caption("Tableau 10 : force de la vérification selon la source.")
doc.add_heading("9.3 Résultats", 2)
para("Sur les 354 sessions, aucune n'est marquée douteuse. La règle automatique a jugé 339 sessions cohérentes (dont 65 de CTU-13 "
     "qui relèvent d'une convention de label) et signalé 15 cas, tous examinés individuellement.")
bullets([
    ("13 sessions C2 de Stratosphere (Dridex et TrickBot) : ", "labels cohérents. Ces malwares bancaires contactent régulièrement leur "
     "serveur de commande en HTTPS, souvent sur des ports inhabituels (4443, 4743, 449), sans nom de site. Leur rythme irrégulier "
     "montrait que le critère de départ était trop strict, pas que le label était faux."),
    ("2 sessions bénignes de Stratosphere sans nom de site : ", "l'une est une connexion vide d'un seul événement, sans handshake TLS visible "
     "(artefact de capture). L'autre est une session longue (7 événements, 2 805 s) sur une machine propre, sans signe de "
     "malveillance : je l'ai jugée cohérente."),
])
para("Une règle de vérification a été corrigée en cours de route : le HTTP en clair n'a pas de nom de site par construction, "
     "ce qui n'est donc pas une anomalie.")

# ================================================================== 10. RISQUES
doc.add_heading("10. Risques identifiés", 1)
table(["Risque", "Constat", "Conséquence possible"],
      [["Peu de vrais tunnels DNS", "42 sessions C2 en DNS-EXF (7 en test)", "Recall DNS instable"],
       ["Labels CTU-13 par convention", "2 260 sessions DNS banales étiquetées C2", "Modèle qui apprend du DNS ordinaire"],
       ["Sources à une seule classe", "99,82 % d'exactitude avec la seule source", "Modèle qui reconnaît la source"],
       ["Petits jeux de validation", "val : 1 244 C2 ; unseen : 221 C2", "Métriques instables"],
       ["Sessions très répétitives", "1 425 sessions TrickBot / Dridex à 6 événements", "Scores gonflés"],
       ["Certificats interceptés", "2 478 sessions Stratosphere avec certificat de proxy", "Certificat et JA3S trompeurs"],
       ["Valeurs manquantes liées à la source", "server_name, ja3, no_tls_seen", "Signal artificiel"],
       ["Découpage par fichier", "16 machines sur 54 présentes dans plusieurs jeux", "Fuite possible par machine"],
       ["Noms DNS tronqués", "Captures limitées à 96 octets par paquet", "Entropie sous-estimée"]],
      [4.2, 6.4, 5.8])
caption("Tableau 11 : risques et conséquences.")
para("Sur le découpage : le lot P2 découpe par fichier PCAP, et par machine uniquement pour DoHBrw malveillant. Ce choix écarte la "
     "fuite par fichier, mais une même machine bénigne peut apparaître dans l'entraînement et le test. Je le signale pour validation.")

# ================================================================== 11. RECOMMANDATIONS
doc.add_heading("11. Recommandations pour les lots suivants", 1)
table(["Lot", "Recommandation"],
      [["P2", "Valider le découpage (machines communes à plusieurs jeux). Vérifier les sessions C2 d'environ 10 microsecondes. "
              "Envisager de ne plus tronquer les noms DNS."],
       ["P4", "Ne pas compter sur l'entropie ou la taille des requêtes seules. Privilégier le rythme (écarts entre requêtes, "
              "régularité). Colonnes interdites : src_ip_hash, dst_ip_hash, session_key, qname_truncated."],
       ["P5", "Ne pas utiliser server_name vide, ja3 vide ou no_tls_seen comme features. Ignorer certificat, auto-signé et JA3S sur les "
              "2 478 sessions interceptées. Régler le seuil à 0,1 % sur le jeu de test."],
       ["P6", "Évaluer aussi source par source. Tester avec et sans ctu13_broad_label. Fournir des intervalles de confiance. "
              "Surveiller les sessions TrickBot / Dridex répétitives."],
       ["P7", "Expliquer qu'une alerte DNS peut venir d'une convention de label (CTU-13) et non d'un vrai tunnel. Ne pas annoncer un "
              "recall global sans le détail par famille."]],
      [1.8, 14.6])
caption("Tableau 12 : recommandations par lot.")

# ================================================================== 12. LIMITES
doc.add_heading("12. Limites", 1)
bullets([
    "La vérification des labels contrôle la cohérence, pas l'exactitude absolue. Un label faux mais cohérent avec les règles passerait inaperçu.",
    "339 des 354 verdicts reposent sur la règle automatique. Seuls 15 cas ont été examinés individuellement.",
    "Pour DoHBrw, la vérification est faible : trafic normal et C2 ont le même aspect côté règle.",
    "Je n'ai modifié aucune donnée. Toute anomalie réelle est renvoyée au lot concerné.",
    "Les valeurs lues sur les graphiques (400 et 2 800 octets, 10 microsecondes) sont des estimations visuelles.",
    "Les performances de détection (recall, fausses alertes) dépendent des lots P4 à P7 et ne sont pas mesurées ici.",
])

# ================================================================== 13. CONCLUSION
doc.add_heading("13. Conclusion", 1)
para("Mon travail a établi que la base de sessions est de bonne qualité : sans doublon, sans incohérence, anonymisée, "
     "cohérente avec les fichiers de découpage et au-dessus du seuil de 100 000 sessions bénignes dans le jeu de test.")
para("Il a aussi mis en évidence que les difficultés à venir viennent de la composition des données plutôt que de leur propreté : "
     "peu de vrais tunnels DNS, des sources à classe unique, des labels par convention et des mesures simples qui se chevauchent. "
     "Les recommandations ci-dessus permettent aux lots suivants d'en tenir compte dès le calcul des features et l'évaluation des modèles.")

# ================================================================== ANNEXES
doc.add_heading("Annexe A. Livrables", 1)
table(["Fichier", "Contenu"],
      [["p3_output/sessions_unified.parquet", "Base de sessions contrôlée (184 979 lignes, 41 colonnes)"],
       ["p3_output/quality_report.txt", "Rapport de contrôle qualité détaillé"],
       ["p3_output/suite_report.txt", "Colonnes ajoutées, échantillon de labels, graphiques"],
       ["p3_output/extra_report.txt", "Doublons de contenu, machines communes, raccourci de source, entropie"],
       ["p3_output/labels_a_verifier.csv", "Échantillon de 354 sessions avec verdicts et commentaires"],
       ["p3_output/graphs/", "8 graphiques d'analyse exploratoire"],
       ["p3_output/Rapport_P3.html et Rapport_P3.docx", "Rapports de synthèse"]],
      [7.2, 9.2])

doc.add_heading("Annexe B. Glossaire", 1)
table(["Terme", "Définition"],
      [["C2", "Serveur de commande et contrôle, qui envoie des ordres aux machines infectées."],
       ["Bénin", "Normal, sans danger."],
       ["Session", "Ensemble des échanges réseau entre une machine et un service. Une ligne de la base."],
       ["Label", "Réponse attendue pour une session : 0 (normal) ou 1 (malveillant)."],
       ["PCAP", "Enregistrement brut du trafic réseau."],
       ["Parquet", "Format de fichier tabulaire adapté aux gros volumes."],
       ["DNS", "Service qui traduit un nom de site en adresse. Un tunnel DNS cache des données dans ces demandes."],
       ["HTTPS", "Navigation web chiffrée."],
       ["Balisage (beacon)", "Contact régulier d'un malware avec son serveur pour demander des ordres."],
       ["Entropie", "Mesure du caractère aléatoire d'un texte."],
       ["JA3 / JA3S", "Empreintes de la négociation TLS du client et du serveur."],
       ["SNI / server_name", "Nom du site demandé dans une connexion HTTPS."],
       ["Fuite de données", "Présence des mêmes données dans l'entraînement et le test, qui rend les résultats trop favorables."],
       ["FPR", "Taux de fausses alertes : part des sessions normales signalées à tort. Objectif : moins de 0,1 %."],
       ["Recall", "Part des sessions malveillantes effectivement détectées. Objectif : au moins 98 %."]],
      [4.0, 12.4])

doc.core_properties.title = "Rapport du lot P3 - Contrôle qualité de la base de sessions"
doc.core_properties.subject = "Détection de trafic C2 (DNS / HTTPS)"
doc.save(DEST)
print("Ecrit :", DEST)
