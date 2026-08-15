"""Entrypoint: Flask web service + the Discord bot running in a thread."""
from __future__ import annotations

import threading

from flask import Flask, jsonify

import config
import main

app = Flask(__name__)


@app.route("/")
def index():
    return jsonify(
        status="ok",
        bot=main.bot.is_ready(),
        user=str(main.bot.user) if main.bot.user else None,
    )


def run_bot():
    main.bot.run(config.DISCORD_TOKEN)


if __name__ == "__main__":
    if not config.DISCORD_TOKEN:
        raise SystemExit("Set DISCORD_TOKEN in .env (see .env.example).")
    threading.Thread(target=run_bot, daemon=True).start()
    app.run(host="0.0.0.0", port=8080)