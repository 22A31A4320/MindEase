from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def start():
    return render_template("start.html")

@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    if request.method == "POST":
        mood = request.form.get("mood")
        sleep = int(request.form.get("sleep"))
        energy = request.form.get("energy")
        anxiety = request.form.get("anxiety")
        motivation = request.form.get("motivation")
        concentration = request.form.get("concentration")
        social = request.form.get("social")
        selfcare = request.form.get("selfcare")
        outlook = request.form.get("outlook")

        # Scoring logic
        scores = {
            "mood": {"happy": 10, "neutral": 7, "sad": 4, "irritable": 3, "anxious": 2},
            "energy": {"high": 10, "medium": 7, "low": 4, "verylow": 2},
            "anxiety": {"never": 10, "sometimes": 7, "often": 4, "always": 2},
            "motivation": {"very": 10, "somewhat": 6, "not": 3},
            "concentration": {"easily": 10, "sometimes": 6, "hardly": 3},
            "social": {"daily": 10, "weekly": 7, "rarely": 4, "never": 2},
            "selfcare": {"daily": 10, "weekly": 7, "rarely": 4},
            "outlook": {"positive": 10, "neutral": 6, "negative": 3}
        }

        total_score = (
            scores["mood"][mood]
            + scores["energy"][energy]
            + scores["anxiety"][anxiety]
            + scores["motivation"][motivation]
            + scores["concentration"][concentration]
            + scores["social"][social]
            + scores["selfcare"][selfcare]
            + scores["outlook"][outlook]
        )

        # Sleep score
        if 7 <= sleep <= 8:
            total_score += 10
        elif 5 <= sleep < 7 or 8 < sleep <= 9:
            total_score += 7
        else:
            total_score += 3

        # Determine analysis
        if total_score >= 75:
            status = "🌈 Excellent Mental Health"
            suggestion = "You seem balanced, energetic, and emotionally strong! Keep maintaining your positive habits and self-care routine."
        elif 55 <= total_score < 75:
            status = "💪 Moderate Mental Health"
            suggestion = "You're doing fairly well, but you could benefit from relaxation activities like meditation, journaling, or nature walks."
        else:
            status = "😔 Signs of Stress"
            suggestion = "Consider adjusting your sleep, connecting socially, or talking to a counselor for emotional support."

        return render_template("result.html", score=total_score, status=status, suggestion=suggestion)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
