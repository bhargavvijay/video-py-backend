from flask import Flask, request, jsonify
from transformers import pipeline
from collections import defaultdict

app = Flask(__name__)

# Load the summarizer once when the server starts
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@app.route('/summarize', methods=['POST'])
def summarize_transcripts():
    data = request.get_json()

    transcripts = data.get("transcripts", {})
    roles = data.get("roles", {})

    role_texts = defaultdict(str)
    for speaker, text in transcripts.items():
        role = roles.get(speaker, "unknown")
        role_texts[role] += " " + text

    summaries_by_role = {}

    for role, text in role_texts.items():
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        summaries = summarizer(chunks, max_length=100, min_length=30, do_sample=False)
        final_summary = "\n".join([s['summary_text'] for s in summaries])
        summaries_by_role[role] = final_summary

    return jsonify(summaries_by_role)

if __name__ == '__main__':
    app.run(debug=True)