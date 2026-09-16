import urllib.request
import json

TOKEN = "REDACTED_TELEGRAM_TOKEN"

def send_alert():
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    req = urllib.request.urlopen(url)
    data = json.loads(req.read().decode('utf-8'))

    if not data.get("result"):
        print("❌ Telegram पर बॉट को खोलकर एक बार 'Hi' भेजें!")
        return

    last_msg = data["result"][-1]["message"]
    chat_id = last_msg["chat"]["id"]
    user_name = last_msg["chat"].get("first_name", "User")
    print(f"✅ Subscriber Found: {user_name} | Chat ID: {chat_id}")

    alert_text = (
        "🚨 *IMD / CYCLOVISION AI EMERGENCY RED ALERT*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "🌀 *Active Storm:* CYCLONE-X (BOB-02)\n"
        "⚠️ *Stage:* Very Severe Cyclonic Storm (VSCS)\n"
        "💨 *Max Sustained Wind:* 165 km/h (Gusts: 185 km/h)\n"
        "📍 *Target Landfall:* Visakhapatnam - Puri Coast\n"
        "🌊 *Storm Surge:* 3.8 Meters Inundation Risk\n"
        "⏱️ *Impact Window:* Next 18 Hours\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "🛡️ *Action Required:* Evacuate coastal lowlands immediately. Move to designated concrete shelters.\n"
        "📞 *State Disaster Helpline:* 1070"
    )

    send_url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": alert_text, "parse_mode": "Markdown"}).encode('utf-8')
    req_send = urllib.request.Request(send_url, data=payload, headers={'Content-Type': 'application/json'})
    urllib.request.urlopen(req_send)
    print("🚀 BOOM! Alert successfully delivered to your Telegram phone!")

if __name__ == "__main__":
    send_alert()
