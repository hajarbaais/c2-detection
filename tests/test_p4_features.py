"""P4 tâche 6 - Tests unitaires des features DNS & temporelles (scripts de Hajar).

Deux niveaux :
  1. fonctions élémentaires (entropie, sous-domaine, ratio de chiffres, CV, périodicité)
     testées sur des valeurs connues à la main (« mot connu », « série périodique artificielle ») ;
  2. intégration : les 5 scripts tournent sur une mini-sortie P2 synthétique (conftest.make_p2)
     et chaque feature doit avoir la valeur attendue sur des sessions construites exprès.

Les tests marqués xfail(strict=True) documentent des ANOMALIES trouvées dans le code P4 :
ils « échouent comme prévu » tant que le bug existe, et deviennent rouges (XPASS) dès qu'il est
corrigé -> il suffit alors de retirer le marqueur.
"""
import math
import re

import numpy as np
import pytest

from conftest import REPO

FEATURE_DICT = REPO / "p4_features_dns_time" / "feature_dict.md"
FORBIDDEN = {"qname_truncated", "qname_truncated_share", "label", "family", "src_ip_hash", "dst_ip_hash"}


# =========================================================================== #
# 1. Fonctions élémentaires
# =========================================================================== #
class TestShannonEntropy:
    @pytest.mark.parametrize("word, expected", [
        ("", 0.0), ("aaaa", 0.0), ("ab", 1.0), ("aabb", 1.0), ("abcd", 2.0), ("abcdefgh", 3.0),
        ("www", 0.0), ("google", -(2 * (2 / 6) * math.log2(2 / 6) + 2 * (1 / 6) * math.log2(1 / 6))),
    ])
    def test_known_words(self, p4, word, expected):
        assert p4["01"].shannon_entropy(word) == pytest.approx(expected, abs=1e-9)

    def test_order_does_not_matter(self, p4):
        f = p4["01"].shannon_entropy
        assert f("abcabc") == pytest.approx(f("cbacba"))

    def test_bounded_by_log2_alphabet(self, p4):
        s = "abcdefghijklmnopqrstuvwxyz234567" * 3          # 32 symboles équiprobables
        assert p4["01"].shannon_entropy(s) == pytest.approx(5.0)

    def test_encoded_string_higher_than_word(self, p4):
        f = p4["01"].shannon_entropy
        assert f("mzxw6ytboi4tqmrsgq3dk") > 3.5 > f("mail")


class TestSubdomain:
    @pytest.mark.parametrize("query, parent, expected", [
        ("x9f2k.google.com", "google.com", "x9f2k"),
        ("a.b.example.co.uk", "example.co.uk", "ab"),           # points retirés
        ("google.com", "google.com", ""),                       # requête sur le parent lui-même
        ("WWW.Google.COM.", "google.com", "www"),                # casse et point final
        ("24125.89cufV7pr2a7", "cicresearch.ca", "2412589cufv7pr2a7"),  # nom tronqué DNS-EXF
        (None, "google.com", ""),
        ("", "google.com", ""),
        ("abc.evil.com", None, "abcevilcom"),
    ])
    def test_cases(self, p4, query, parent, expected):
        assert p4["01"].subdomain(query, parent) == expected


class TestDigitRatio:
    @pytest.mark.parametrize("text, expected", [
        ("ab12", 0.5), ("1234", 1.0), ("abcd", 0.0), ("", 0.0), ("a-1.b", 1 / 3), (None, 0.0), ("---", 0.0),
    ])
    def test_cases(self, p4, text, expected):
        assert p4["02"].digit_ratio(text) == pytest.approx(expected)


class TestCoefficientOfVariation:
    def test_periodic_is_zero(self, p4):
        iat = np.diff([0, 60, 120, 180, 240])
        assert p4["03"].cv(iat.std(), iat.mean()) == pytest.approx(0.0)

    def test_poisson_close_to_one(self, p4):
        iat = np.random.default_rng(0).exponential(60, 20_000)
        assert p4["03"].cv(iat.std(), iat.mean()) == pytest.approx(1.0, abs=0.05)

    def test_zero_mean_returns_zero(self, p4):
        assert p4["03"].cv(0.0, 0.0) == 0.0


class TestPeriodicity:
    @pytest.mark.parametrize("period", [5.0, 60.0, 3600.0])
    def test_perfect_beacon_any_scale(self, p4, period):
        dominant, score = p4["04"].periodicity(np.arange(50) * period)
        assert score > 0.9
        assert dominant == pytest.approx(period, rel=0.02)

    def test_jitter_10pct_keeps_period(self, p4):
        rng = np.random.default_rng(1)
        t = np.cumsum(60 * (1 + 0.1 * rng.uniform(-1, 1, 80)))
        dominant, score = p4["04"].periodicity(t)
        assert score > 0.5
        assert dominant == pytest.approx(60, rel=0.1)

    def test_jitter_25pct_beats_random(self, p4):
        rng = np.random.default_rng(2)
        jittered = np.cumsum(60 + rng.uniform(-15, 15, 80))
        _, s_jit = p4["04"].periodicity(jittered)
        s_rand = max(p4["04"].periodicity(np.cumsum(rng.exponential(60, 80)))[1] for _ in range(10))
        assert s_jit > s_rand

    def test_random_has_low_score(self, p4):
        rng = np.random.default_rng(3)
        scores = [p4["04"].periodicity(np.cumsum(rng.exponential(60, 80)))[1] for _ in range(20)]
        assert max(scores) < 0.3

    def test_order_independent(self, p4):
        t = np.arange(30) * 10.0
        assert p4["04"].periodicity(t[::-1]) == p4["04"].periodicity(t)

    @pytest.mark.parametrize("times", [np.arange(9) * 60.0, np.zeros(20), np.array([])])
    def test_degenerate_inputs(self, p4, times):
        dominant, score = p4["04"].periodicity(times)
        assert math.isnan(dominant) and score == 0.0


# =========================================================================== #
# 2. Intégration : les 5 scripts sur une mini-sortie P2
# =========================================================================== #
def test_p4_01_02_03_05_run_without_error(p4_run):
    _, errors = p4_run
    for k in ("01", "02", "03", "05"):
        assert errors[k] is None, f"p4_{k} a levé {errors[k]!r}"


@pytest.mark.xfail(strict=True, reason="BUG p4_04 : la vérification finale fait groupby('family') alors que "
                                        "sessions_pcap est lu sans la colonne 'family' -> KeyError. "
                                        "Correctif : ajouter 'family' dans columns=[...] ligne 'se = pd.read_parquet'.")
def test_p4_04_runs_without_error(p4_run):
    _, errors = p4_run
    assert errors["04"] is None, repr(errors["04"])


def test_one_row_per_session(p4_run):
    feats, _ = p4_run
    assert feats.index.is_unique


def test_every_dns_session_present(p4_run):
    feats, _ = p4_run
    for sid in ["beacon_dns", "exfil_dns", "benign_dns", "single_dns", "winA0", "winA1", "winB0"]:
        assert sid in feats.index


def test_web_session_without_payload_is_absent(p4_run):
    """Comportement documenté : P6 / P7 doivent faire une jointure GAUCHE depuis sessions_pcap."""
    feats, _ = p4_run
    assert "empty_web" not in feats.index


def test_no_forbidden_column(p4_run):
    feats, _ = p4_run
    assert not (set(feats.columns) & FORBIDDEN)


def test_entropy_exfil_vs_benign(p4_run):
    f, _ = p4_run
    assert f.loc["exfil_dns", "dns_entropy_mean"] > 4.0          # base32 aléatoire ~ 4,7 bits/car.
    assert f.loc["benign_dns", "dns_entropy_mean"] < 2.0
    assert f.loc["single_dns", "dns_entropy_max"] == 0.0         # requête sur le parent : pas de sous-domaine


def test_query_shape(p4_run):
    f, _ = p4_run
    assert f.loc["exfil_dns", "dns_pct_txt"] == 1.0
    assert f.loc["benign_dns", "dns_pct_txt"] == 0.0
    assert f.loc["exfil_dns", "dns_pct_nxdomain"] == pytest.approx(10 / 30)
    assert f.loc["exfil_dns", "dns_unique_ratio"] == 1.0          # tout est unique
    assert f.loc["beacon_dns", "dns_unique_ratio"] == pytest.approx(1 / 60)
    assert f.loc["benign_dns", "dns_unique_ratio"] == pytest.approx(4 / 6)
    assert f.loc["exfil_dns", "dns_qname_len_mean"] == len("x" * 40 + ".tunnel.net")
    assert f.loc["benign_dns", "dns_n_levels_mean"] == pytest.approx((3 + 3 + 3 + 2 + 3 + 3) / 6)


def test_interarrival(p4_run):
    f, _ = p4_run
    assert f.loc["beacon_dns", "dns_iat_cv"] < 0.1
    assert f.loc["beacon_dns", "dns_iat_mean"] == pytest.approx(60, rel=0.05)
    assert f.loc["benign_dns", "dns_iat_cv"] > 0.8
    assert np.isnan(f.loc["single_dns", "dns_iat_mean"])          # 1 événement : pas d'écart


def test_periodicity_dns_and_web(p4_run):
    f, _ = p4_run
    assert f.loc["beacon_dns", "dns_periodicity_score"] > 0.5
    assert f.loc["beacon_dns", "dns_period_dominant"] == pytest.approx(60, rel=0.1)
    assert f.loc["beacon_web", "dns_period_dominant"] == pytest.approx(30, rel=0.05)   # via events_pkts 'up'
    assert f.loc["single_dns", "dns_periodicity_score"] == 0.0
    assert np.isnan(f.loc["single_dns", "dns_period_dominant"])


def test_windows_single_window_key(p4_run):
    f, _ = p4_run
    row = f.loc["beacon_dns"]
    assert (row["dns_persist_hours"], row["dns_span_hours"], row["dns_coverage"]) == (1, 1, 1.0)
    assert row["dns_events_per_hour"] == 60 and row["dns_max_events_6h"] == 60


def test_windows_multi_hour_key_same_pcap(p4_run):
    f, _ = p4_run
    assert f.loc["winA1", "dns_persist_hours"] >= 2


@pytest.mark.xfail(strict=True, reason="BUG p4_05 : regroupe par session_key seule ; la même clé existe dans "
                                        "plusieurs pcaps (16 471 clés sur les vraies données) et window_idx "
                                        "est relatif à chaque pcap. Correctif : groupby(['file', 'session_key']).")
def test_windows_do_not_mix_pcaps(p4_run):
    f, _ = p4_run
    assert f.loc["winB0", "dns_persist_hours"] == 1          # B n'a qu'une fenêtre
    assert f.loc["winA0", "dns_events_per_hour"] == 3        # A : 6 requêtes / 2 h, pas 9 / 2


# =========================================================================== #
# 3. Dictionnaire des features
# =========================================================================== #
def _documented_features():
    text = FEATURE_DICT.read_text(encoding="utf-8")
    return set(re.findall(r"^\|\s*`([a-z0-9_]+)`", text, flags=re.M))


def test_feature_dict_exists():
    assert FEATURE_DICT.exists()


def test_every_produced_feature_is_documented(p4_run):
    feats, _ = p4_run
    missing = set(feats.columns) - _documented_features()
    assert not missing, f"features non documentées dans feature_dict.md : {sorted(missing)}"


def test_no_obsolete_feature_in_dict(p4_run):
    feats, _ = p4_run
    obsolete = _documented_features() - set(feats.columns)
    assert not obsolete, f"features documentées mais plus produites : {sorted(obsolete)}"
