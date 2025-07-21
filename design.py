# design.py
import streamlit as st
from typing import Optional, Union
import numpy as np

# ──────────────────────────────────────────────────────────────
# 1.  Feuille de style globale
# ──────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
        <style>
        /* ============================== */
        /*  Palette par défaut = light    */
        /* ============================== */
        :root{
            --bg:       #FFFFFF;
            --text:     #222222;
            --primary:  #13795B;
            --accent:   #F28C28;
            --card-bg:  #FFFFFF;
            --card-shadow: rgba(0,0,0,0.06);
            --delta-pos:#30c48d;
            --delta-neg:#ff6961;
            --delta-neu:#cfcfcf;
        }
        /* =========  Polices  ========== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Poppins:wght@500;700&display=swap');
        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
        h1, h2, h3, h4              { font-family: 'Poppins', sans-serif; }

        /* =========  Fond général  ===== */
        body{ background: var(--bg); color: var(--text); }

        /* =========  Header & Sidebar == */
        header[data-testid="stHeader"]{ background: var(--primary); }
        header h1{ color:#fff; }

        .sidebar .sidebar-content{ background: color-mix(in srgb, var(--primary) 10%, transparent); }

        /* =========  Cards ============== */
        .kpi-card{
            background: var(--card-bg);
            border-radius:12px;
            padding:1rem 1.2rem;
            box-shadow:0 2px 6px var(--card-shadow);
            margin-bottom:0.75rem;
            border:1px solid #ffffff20;
        }
        .kpi-card h3{ margin:0; font-size:0.95rem; font-weight:600; color:var(--text); }
        .kpi-value{ font-size:1.9rem; font-weight:700; color:var(--primary); margin:0.2rem 0; }
        .kpi-delta{ font-size:0.85rem; font-weight:600; }
        .kpi-delta.pos{ color:var(--delta-pos); }
        .kpi-delta.neg{ color:var(--delta-neg); }
        .kpi-delta.neu{ color:var(--delta-neu); }
        .kpi-sub{ font-size:0.75rem; color:var(--delta-neu); }

        /* =========  Divers ============= */
        footer, .css-1q8dd3e{ visibility:hidden; }  /* menu & footer ST */
        </style>
        """,
        unsafe_allow_html=True,
    )

def card(
    container,
    title: str,
    value: Union[str, float, int],
    *,
    delta: Optional[float] = None,
    unit: str = "",
    icon: str = "📈",
    subtext: Optional[str] = None,
    value_fmt: str = "{:,.2f}",      # format num si value est numérique
    delta_fmt: str = "{:+.2f}",      # format num pour delta
    invert_colors: bool = False,     # si une baisse est "bonne" (ex: inflation)
) -> None:
    """
    Affiche un KPI card responsive.

    Parameters
    ----------
    container : Streamlit container (col, st, etc.)
    title     : Titre de l’indicateur.
    value     : Valeur affichée (num ou str).
    delta     : Variation numérique (ex: % vs mois‑1). None = pas d’affichage.
    unit      : Chaîne suffixe ("%", "pts", "FCFA", ...).
    icon      : Emoji/tiny icon.
    subtext   : Ligne d’info (période, source...).
    value_fmt : Format si value numérique.
    delta_fmt : Format si delta numérique.
    invert_colors : Passe le vert sur négatif (utile pour inflation).
    """
    # formater la valeur
    if isinstance(value, (int, float, np.number)):
        value_str = value_fmt.format(value)
    else:
        value_str = str(value)

    # delta
    delta_html = ""
    if delta is not None:
        if isinstance(delta, (int, float, np.number)):
            delta_str = delta_fmt.format(delta)
            sign = "pos" if delta > 0 else "neg" if delta < 0 else "neu"
            # inversion couleur si baisse = bonne nouvelle
            if invert_colors and sign != "neu":
                sign = "pos" if sign == "neg" else "neg"
            arrow = "▲" if delta > 0 else "▼" if delta < 0 else "▬"
            delta_html = (
                f"<div class='kpi-delta {sign}'>{arrow} {delta_str}{unit}</div>"
            )
        else:
            # delta déjà formatté texte
            delta_html = f"<div class='kpi-delta neu'>{delta}{unit}</div>"

    # subtext
    sub_html = f"<div class='kpi-sub'>{subtext}</div>" if subtext else ""

    container.markdown(
        f"""
        <div class='kpi-card'>
          <h3>{icon} {title}</h3>
          <div class='kpi-value'>{value_str}{unit}</div>
          {delta_html}
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True
    )
