from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from twilio.twiml.messaging_response import MessagingResponse
from sklearn.ensemble import RandomForestClassifier
from math import exp, factorial

app = Flask(__name__)

# ----------------------
# ML MODEL (pré-entraîné)
# ----------------------
X_train = np.array([[1.8,1.0],[2.2,0.9],[1.3,1.4],[0.8,1.9],[1.6,1.2],[2.4,0.7]])
y_train = np.array([1,1,0,0,1,1])
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train,y_train)

# ----------------------
# Fonctions bot
# ----------------------
def xg_from_odds(odds):
    return round(2.5/odds,2)

def poisson(lmbda,k):
    return (lmbda**k * exp(-lmbda))/factorial(k)

def best_score(xg_h,xg_a):
    scores = {}
    for h in range(5):
        for a in range(5):
            scores[f"{h}-{a}"] = poisson(xg_h,h)*poisson(xg_a,a)
    return max(scores,key=scores.get)

def btts_prob(xg_h,xg_a):
    return "YES" if (1-exp(-xg_h))*(1-exp(-xg_a))>0.5 else "NO"

def kelly(prob,odds):
    return max(0,((prob*odds)-1)/(odds-1))*100

def double_chance(prob):
    if prob>0.6:
        return "1X"
    elif prob<0.4:
        return "X2"
    else:
        return "12"

def value_bet(prob,odds):
    return (prob*odds)-1

# ----------------------
# WhatsApp endpoint
# ----------------------
@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    msg = request.form.get("Body").strip().split("\n")
    try:
        teams = msg[0].split()
        odds = list(map(float,msg[1].split()))
        home,away = teams[0],teams[1]
        oh,od,oa = odds

        xg_home=xg_from_odds(oh)
        xg_away=xg_from_odds(oa)
        prob_home=model.predict_proba([[xg_home,xg_away]])[0][1]
        fav = home if prob_home>0.5 else away

        reply=f"""
⚽ {home} vs {away}
🏆 Favori : {fav}
🔐 Double chance : {double_chance(prob_home)}
🎯 Score probable : {best_score(xg_home,xg_away)}
⚽ BTTS : {btts_prob(xg_home,xg_away)}
💰 Mise conseillée : {kelly(prob_home,oh):.1f}% bankroll
💎 Value Bet : {'YES' if value_bet(prob_home,oh)>0 else 'NO'}
⚠️ Jeu responsable
"""
    except:
        reply="❌ Format invalide. Exemple:\nArsenal Chelsea\n1.85 3.60 4.20"

    resp = MessagingResponse()
    resp.message(reply)
    return str(resp)

# ----------------------
# Dashboard Flask
# ----------------------
@app.route("/")
def dashboard():
    df=pd.read_csv("data/sample_matches.csv")
    # Ajouter backtesting simple
    df['xG_home'] = df['odds_home'].apply(xg_from_odds)
    df['xG_away'] = df['odds_away'].apply(xg_from_odds)
    df['prob_home'] = df.apply(lambda r: model.predict_proba([[r.xG_home,r.xG_away]])[0][1],axis=1)
    df['Fav'] = df.apply(lambda r: r.home if r.prob_home>0.5 else r.away,axis=1)
    return render_template("dashboard.html", tables=[df.to_html(classes='data', index=False)])

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)
