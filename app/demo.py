"""Démo Gradio de vérification de signature.

Lancement (depuis la racine du projet) :
    python app/demo.py            # local
    python app/demo.py --share    # lien public temporaire, pour montrer à quelqu'un
"""
import argparse
import html
import inspect
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from predict import SignatureVerifier  # noqa: E402

verifier = SignatureVerifier(ROOT / "models" / "best_model_v3.pt")

ROTATION_CHOICES = [
    ("Désactivée : signatures dans le même sens (recommandé)", "off"),
    ("Quarts de tour : 0°, 90°, 180°, 270°", "quarter"),
    ("Tous les angles : pas de 15° (moins fiable)", "full"),
]
ROTATION_NAMES = {"off": "désactivée", "quarter": "quarts de tour", "full": "tous les angles"}

# --------------------------------------------------------------------------- style
CSS = """
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; }

.sv-header { padding: 26px 8px 8px; }
.sv-header h1 { font-size: 1.9rem; font-weight: 800; margin: 0 0 4px; letter-spacing: -0.02em; }
.sv-header p { margin: 0 0 12px; opacity: .75; font-size: 1.02rem; }
.sv-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.sv-chip { font-size: .78rem; font-weight: 600; padding: 4px 11px; border-radius: 999px;
           background: rgba(99,102,241,.14); color: #6366f1; border: 1px solid rgba(99,102,241,.30); }
.sv-chip.warn { background: rgba(217,119,6,.14); color: #d97706; border-color: rgba(217,119,6,.35); }

.sv-section { font-weight: 700; font-size: 1.02rem; margin: 14px 2px 2px; }
.sv-section span { font-weight: 400; opacity: .65; font-size: .88rem; margin-left: 6px; }

.sv-card { border-radius: 16px; padding: 20px 22px; border: 1px solid; margin-top: 4px; }
.sv-card.ok   { background: rgba(22,163,74,.10);  border-color: rgba(22,163,74,.40); }
.sv-card.ko   { background: rgba(220,38,38,.10);  border-color: rgba(220,38,38,.40); }
.sv-card.warn { background: rgba(217,119,6,.10);  border-color: rgba(217,119,6,.40); }
.sv-card.idle { background: rgba(148,163,184,.10); border-color: rgba(148,163,184,.35); border-style: dashed; }

.sv-head { display: flex; align-items: center; gap: 14px; margin-bottom: 18px; }
.sv-icon { width: 46px; height: 46px; border-radius: 50%; display: flex; align-items: center;
           justify-content: center; font-size: 1.5rem; font-weight: 800; color: #fff; flex: none; }
.ok .sv-icon { background: #16a34a; } .ko .sv-icon { background: #dc2626; } .warn .sv-icon { background: #d97706; }
.sv-title { font-size: 1.35rem; font-weight: 800; line-height: 1.2; }
.ok .sv-title { color: #16a34a; } .ko .sv-title { color: #dc2626; } .warn .sv-title { color: #d97706; }
.sv-sub { opacity: .75; font-size: .95rem; margin-top: 2px; }

.sv-track-wrap { margin: 4px 4px 6px; }
.sv-track { position: relative; height: 12px; border-radius: 999px; background: rgba(220,38,38,.28); }
.sv-zone { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px 0 0 999px; background: rgba(22,163,74,.55); }
.sv-thr { position: absolute; top: -6px; bottom: -6px; width: 2px; background: currentColor; opacity: .55; }
.sv-marker { position: absolute; top: 50%; width: 20px; height: 20px; margin: -10px 0 0 -10px; border-radius: 50%;
             border: 3px solid #fff; box-shadow: 0 1px 4px rgba(0,0,0,.4); }
.ok .sv-marker { background: #16a34a; } .ko .sv-marker { background: #dc2626; }
.sv-scale { display: flex; justify-content: space-between; font-size: .76rem; opacity: .6; margin: 8px 2px 16px; }

.sv-metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
.sv-metric { background: rgba(148,163,184,.12); border-radius: 12px; padding: 10px 12px; }
.sv-metric span { display: block; font-size: .74rem; opacity: .65; text-transform: uppercase; letter-spacing: .04em; }
.sv-metric b { font-size: 1.2rem; }
.sv-note { margin-top: 12px; font-size: .85rem; opacity: .7; }
.sv-idle-text { text-align: center; opacity: .65; padding: 12px 0; }

.sv-footer { text-align: center; font-size: .82rem; opacity: .6; padding: 18px 0 8px; }

@media (max-width: 640px) { .sv-metrics { grid-template-columns: repeat(2, 1fr); } }
"""

THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue", neutral_hue="slate", radius_size="lg")

HEADER = """
<div class="sv-header">
  <h1>Vérification de signature</h1>
  <p>Comparez une signature à des références authentiques grâce à un réseau siamois.</p>
  <div class="sv-chips">
    <span class="sv-chip">Réseau siamois · ResNet18</span>
    <span class="sv-chip">Entraîné sur CEDAR</span>
    <span class="sv-chip warn">Prototype : ne pas utiliser pour sécuriser des opérations réelles</span>
  </div>
</div>
"""

FOOTER = """
<div class="sv-footer">
  Les images sont traitées en mémoire et ne sont pas conservées.
  Erreur mesurée sur le jeu de test : environ 27 % pour les falsifications expertes, 16 % pour les signatures d'autres personnes.
</div>
"""

HELP = """
**Comment ça marche.** Le modèle transforme chaque signature en une empreinte numérique, puis mesure la
*distance* entre la signature à vérifier et chaque référence. Plus la distance est petite, plus elles se ressemblent.
La signature est acceptée si la distance moyenne est **inférieure au seuil**.

**Le seuil.** Plus il est bas, plus le système est strict : il rejette davantage de fausses signatures, mais aussi
davantage de vraies. Le seuil par défaut équilibre les deux types d'erreur sur le jeu de test.

**Pour de bons résultats.**
- Utiliser des références **authentiques** et, si possible, deux plutôt qu'une.
- Photographier la feuille de près, sans ombre, avec la signature seule.
- Garder le **même sens** pour toutes les signatures. La tolérance à la rotation existe, mais elle augmente le
  risque d'accepter une falsification et n'a pas été évaluée.
"""


# --------------------------------------------------------------------------- rendu du résultat
def card_idle():
    return ('<div class="sv-card idle"><div class="sv-idle-text">'
            "Le résultat apparaîtra ici après la vérification.</div></div>")


def card_message(text, kind="warn"):
    return (f'<div class="sv-card {kind}"><div class="sv-head"><div class="sv-icon">!</div>'
            f'<div><div class="sv-title">À vérifier</div><div class="sv-sub">{html.escape(text)}</div>'
            "</div></div></div>")


def card_result(res):
    ok = res["accepted"]
    kind = "ok" if ok else "ko"
    icon = "✓" if ok else "✕"
    title = "Signature authentique" if ok else "Signature rejetée"
    sub = ("Elle ressemble aux signatures de référence." if ok
           else "Elle diffère des références : falsification possible.")

    dist, thr = res["distance"], res["threshold"]
    scale = max(1.0, dist * 1.15, thr * 2.0)  # l'axe va de 0 à `scale`
    pos_dist = min(100.0, dist / scale * 100.0)
    pos_thr = min(100.0, thr / scale * 100.0)

    note = ""
    if res.get("rotation", "off") != "off":
        angles = ", ".join(f"{a}°" for a in res["angles"])
        note = (f'<div class="sv-note">Tolérance à la rotation : {ROTATION_NAMES[res["rotation"]]} '
                f"· meilleur angle par référence : {angles}</div>")

    return f"""
<div class="sv-card {kind}">
  <div class="sv-head">
    <div class="sv-icon">{icon}</div>
    <div><div class="sv-title">{title}</div><div class="sv-sub">{sub}</div></div>
  </div>
  <div class="sv-track-wrap">
    <div class="sv-track">
      <div class="sv-zone" style="width:{pos_thr:.1f}%"></div>
      <div class="sv-thr" style="left:{pos_thr:.1f}%"></div>
      <div class="sv-marker" style="left:{pos_dist:.1f}%"></div>
    </div>
  </div>
  <div class="sv-scale"><span>Plus proche</span><span>Trait fin : seuil de décision</span><span>Plus différent</span></div>
  <div class="sv-metrics">
    <div class="sv-metric"><span>Distance</span><b>{dist:.3f}</b></div>
    <div class="sv-metric"><span>Seuil</span><b>{thr:.3f}</b></div>
    <div class="sv-metric"><span>Score indicatif</span><b>{res["similarity"]:.0f} %</b></div>
    <div class="sv-metric"><span>Références</span><b>{res["n_references"]}</b></div>
  </div>
  {note}
</div>
"""


def verify_signatures(ref_1, ref_2, candidate, threshold, rotation):
    refs = [r for r in (ref_1, ref_2) if r is not None]
    if not refs:
        return card_message("Ajoutez au moins une signature de référence authentique.")
    if candidate is None:
        return card_message("Ajoutez la signature à vérifier.")
    try:
        res = verifier.verify(candidate, refs, threshold=threshold, rotation=rotation)
    except Exception as exc:  # image illisible, etc.
        return card_message(f"Analyse impossible : {exc}")
    return card_result(res)


# --------------------------------------------------------------------------- interface
def image_input(label):
    return gr.Image(label=label, type="numpy", image_mode="RGBA",
                    sources=["upload", "clipboard"], height=230)


# Gradio 6 attend theme/css dans launch(), les versions précédentes dans Blocks()
STYLE = {"theme": THEME, "css": CSS}
BLOCKS_KW = {k: v for k, v in STYLE.items() if k in inspect.signature(gr.Blocks.__init__).parameters}
LAUNCH_KW = {k: v for k, v in STYLE.items() if k in inspect.signature(gr.Blocks.launch).parameters}

with gr.Blocks(title="Vérification de signature", **BLOCKS_KW) as demo:
    gr.HTML(HEADER)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gr.HTML('<div class="sv-section">1 · Signatures de référence'
                    "<span>authentiques, une ou deux</span></div>")
            ref_1 = image_input("Référence 1")
            ref_2 = image_input("Référence 2 (facultative)")

        with gr.Column(scale=1):
            gr.HTML('<div class="sv-section">2 · Signature à vérifier</div>')
            candidate = image_input("Signature à vérifier")
            with gr.Accordion("Réglages avancés", open=False):
                threshold = gr.Slider(0.2, 0.9, value=verifier.threshold, step=0.01,
                                      label="Seuil de décision (plus bas = plus strict)")
                rotation = gr.Radio(ROTATION_CHOICES, value="off", label="Tolérance à la rotation")
            run = gr.Button("Vérifier la signature", variant="primary", size="lg")
            clear = gr.Button("Tout effacer", variant="secondary")

    gr.HTML('<div class="sv-section">3 · Résultat</div>')
    result = gr.HTML(card_idle())

    with gr.Accordion("Comment lire le résultat ?", open=False):
        gr.Markdown(HELP)
    gr.HTML(FOOTER)

    run.click(verify_signatures, [ref_1, ref_2, candidate, threshold, rotation], result)
    clear.click(lambda: (None, None, None, card_idle()), None, [ref_1, ref_2, candidate, result])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true", help="lien public temporaire")
    demo.launch(share=p.parse_args().share, **LAUNCH_KW)