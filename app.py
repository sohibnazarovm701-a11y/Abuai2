import os
from flask import Flask, request, jsonify, send_from_directory
from groq import Groq

# ============================================================
# Калиди Groq-ро дар ҳамин ҷо гузоред (ё дар муҳити сервер).
# Ҳоло ин танҳо ҷойгиркунанда аст — баъдтар худатон иваз кунед.
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "ВОРИД_КАРДАНИ_КАЛИД")

MODEL = "openai/gpt-oss-120b"
SYSTEM_PROMPT = (
    "Ту ёрдамчии AI бо номи Abiau ҳастӣ. "
    "Ба забоне ҷавоб деҳ, ки корбар менависад. "
    "Агар бо тоҷикӣ навиштанд, бо тоҷикӣ ҷавоб деҳ. "
    "Ҷавобҳоро равшан ва кӯтоҳ нигоҳ дор."
)

app = Flask(__name__)
client = Groq(api_key=GROQ_API_KEY)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    history = data.get("messages", [])

    # Танҳо 20 паёми охиринро мегирем ва сохтори онҳоро тафтиш мекунем
    clean = [
        {"role": m["role"], "content": str(m["content"])[:4000]}
        for m in history[-20:]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    if not clean:
        return jsonify({"error": "Паём холӣ аст."}), 400

    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + clean,
            temperature=0.7,
            max_tokens=1024,
        )
        return jsonify({"reply": res.choices[0].message.content})
    except Exception as e:
        print("Хатои Groq:", e)
        return jsonify({"error": "Аз Groq ҷавоб нагирифтем. Калидро тафтиш кунед."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
