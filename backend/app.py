"""
Flask entry point. Does exactly two things:
  1. serves the static React build (frontend/dist)
  2. answers chat queries by running the agent over the ML tools

Run:  python app.py   (needs ANTHROPIC_API_KEY, e.g. in backend/.env)
"""

import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import data
from model_store import get_model
from agent import run_agent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend", "dist"))

# Load ANTHROPIC_API_KEY from backend/.env regardless of the working directory.
load_dotenv(os.path.join(BASE_DIR, ".env"))

# static_folder=None: we serve the built SPA ourselves via the catch-all below,
# so Flask's implicit static route can't shadow deep client-side routes.
app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/api/*": {"origins": "*"}})   # for the Vite dev origin


ASSUMPTION_NOTES = [
    "This is an MVP on synthetic data — the 2000 leads, 15 offers, and every "
    "conversion outcome are randomly generated to demonstrate the approach.",
    "Expected value (EV) = P(convert) x payout. Payout is excluded from the "
    "conversion model (the consumer never sees it) and only scales EV.",
    "Conversion is driven by how well the offer matches the lead's intent, the "
    "effort the offer demands (commitment level), engagement/recency, and "
    "acquisition source.",
    "Sending is budget-constrained (blasting everyone damages sender reputation), "
    "so revenue is framed as the best use of a K-send budget: the model's top-K "
    "leads vs a random K.",
]


def flywheel_enabled():
    """Feature flag for the learning-flywheel simulation. Off by default — set
    ENABLE_FLYWHEEL=true (or 1/yes/on) in the environment to turn it on."""
    return os.environ.get("ENABLE_FLYWHEEL", "false").strip().lower() in (
        "1", "true", "yes", "on")


@app.route("/api/config")
def config():
    """Feature flags the SPA reads on load (e.g. whether to show the flywheel tab
    as live or as 'coming soon')."""
    return jsonify({"flywheel_enabled": flywheel_enabled()})


@app.route("/api/assumptions")
def assumptions():
    summary = data.catalog_summary()
    summary["notes"] = ASSUMPTION_NOTES
    return jsonify(summary)


# Web-facing clamps, tighter than flywheel.CAPS, so a browser click can't queue a
# minute-long compute. simulate() clamps again to the module's hard caps.
_FLY_WEB = {"rounds": (4, 36), "budget": (50, 800), "ensemble": (2, 12),
            "warmstart": (0, 1500), "retrain_every": (1, 6)}


@app.route("/api/flywheel", methods=["POST"])
def flywheel_run():
    """Run the learning-flywheel + bandit simulation and return its series.
    Pure ML/simulation — needs no ANTHROPIC_API_KEY, so it works even when
    /api/chat can't. Gated behind the ENABLE_FLYWHEEL feature flag."""
    if not flywheel_enabled():
        return jsonify({"error": "The flywheel simulation is not enabled on this "
                                 "server (set ENABLE_FLYWHEEL=true)."}), 503
    import flywheel
    body = request.get_json(silent=True) or {}
    cfg = {}
    for k, (lo, hi) in _FLY_WEB.items():
        if body.get(k) is not None:
            try:
                cfg[k] = min(max(int(body[k]), lo), hi)
            except (TypeError, ValueError):
                return jsonify({"error": f"'{k}' must be an integer"}), 400
    if body.get("postback") is not None:
        try:
            cfg["postback"] = min(max(float(body["postback"]), 0.1), 1.0)
        except (TypeError, ValueError):
            return jsonify({"error": "'postback' must be a number"}), 400
    if body.get("seed") is not None:
        try:
            cfg["seed"] = int(body["seed"])
        except (TypeError, ValueError):
            return jsonify({"error": "'seed' must be an integer"}), 400
    cfg["reps"] = 1                       # no seed-averaging over the web (latency)
    try:
        return jsonify(flywheel.simulate(cfg))
    except Exception as e:
        app.logger.exception("flywheel failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    messages = payload.get("messages")
    if not messages:
        return jsonify({"error": "Request body must include a non-empty 'messages' list."}), 400
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return jsonify({"error": "ANTHROPIC_API_KEY is not set on the server."}), 503
    try:
        return jsonify(run_agent(messages))
    except Exception as e:
        app.logger.exception("chat failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


# --- SPA: serve a real file if it exists, else index.html (client routing) ---
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def spa(path):
    full = os.path.join(DIST_DIR, path)
    if path and os.path.isfile(full):
        return send_from_directory(DIST_DIR, path)
    index = os.path.join(DIST_DIR, "index.html")
    if os.path.isfile(index):
        return send_from_directory(DIST_DIR, "index.html")
    return ("Frontend not built yet. Run `npm install && npm run build` in "
            "frontend/, or use the Vite dev server.", 200)


def _startup():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n[WARNING] ANTHROPIC_API_KEY is not set — /api/chat will return an "
              "error until it is. Set it in backend/.env or the environment.\n")
    data.warm()
    get_model()   # train once / load from model.pkl


_startup()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
