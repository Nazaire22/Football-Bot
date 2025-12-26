from flask import Flask, render_template, request
from telegram import Bot, Update
from telegram.ext import CommandHandler, MessageHandler, Filters, Updater
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from math import exp, factorial

app = Flask(__name__)

# ============================
# Initialisation du modèle
# ============================
X_train = np.array([[1.8,1.0],[2.2,0.9],[1.3,1.4],[0.8,1.9],[1.6,1.2],[2.4,0.7]])
y_train = np.array([1,1,0,0,1,1])
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train,y_train)

# ============================
# Fonction de calcul xG
# ============================
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

# ============================
# Fonction pour Telegram
# ============================
TELEGRAM_TOKEN = 8237989275:AAHwDZ_xSzFNOHQwBub7hGowlCxTm4d2HMc

updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
dispatcher = updater.dispatcher

def start(update, context):
    update.message.reply_text("Salut, je suis ton bot de pronostics football ! Envoie-moi un match pour obtenir un pronostic.")

def handle_message(update, context):
    message = update.message.text.strip().split("\n")
    try:
        teams = message[0].split()
        odds = list(map(float,message[1].split()))
        home,away = teams[0],teams[1]
        oh,od,oa = odds

        xg_home=xg_from_odds(oh)
        xg_away=xg_from_odds(oa)
        prob_home=model.predict_proba([[xg_home,xg_away]])[0][1]
        fav = home if prob_home>0.5 else away

        response = f"""
⚽ {home} vs {away}
🏆 Favori : {fav}
🔐 Double chance : {double_chance(prob_home)}
🎯 Score probable : {best_score(xg_home,xg_away)}
⚽ BTTS : {btts_prob(xg_home,xg_away)}
💰 Mise conseillée : {kelly(prob_home,oh):.1f}% bankroll
💎 Value Bet : {'YES' if value_bet(prob_home,oh)>0 else 'NO'}
⚠️ Jeu responsable
"""
        update.message.reply_text(response)
    except:
        update.message.reply_text("❌ Format invalide. Exemple :\nArsenal Chelsea\n1.85 3.60 4.20")

# ============================
# Ajout des handlers Telegram
# ============================
start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)

message_handler = MessageHandler(Filters.text, handle_message)
dispatcher.add_handler(message_handler)

# ============================
# Démarrage du bot
# ============================
if __name__ == "__main__":
    updater.start_polling()
    app.run(host="0.0.0.0", port=5000)