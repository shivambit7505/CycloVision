import time, json, urllib.request, os, subprocess
from datetime import datetime

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "REDACTED_TELEGRAM_TOKEN")
LAST_ALERT_TIME = None
COOLDOWN_SECONDS = 300  # 5-Minute Anti-Spam Cooldown (Slide 30)

def dispatch_sentinel_alert(storm_name, wind_kmph, stage, landfall_time):
    global LAST_ALERT_TIME
    now = datetime.now()
    
    # Check Hysteresis/Cooldown (Slide 30)
    if LAST_ALERT_TIME and (now - LAST_ALERT_TIME).total_seconds() < COOLDOWN_SECONDS:
        print(f"[{now.strftime('%H:%M:%S')}] ⏳ Alert suppressed by anti-spam cooldown window.")
        return
        
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
        req = urllib.request.urlopen(url)
        data = json.loads(req.read().decode('utf-8'))
        if data.get("result"):
            chat_id = data["result"][-1]["message"]["chat"]["id"]
            msg = (
                f"🚨 *[CYCLOVISION V2 AUTONOMOUS SENTINEL]*\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"🌀 *Storm:* {storm_name}\n"
                f"⚠️ *Stage:* {stage}\n"
                f"💨 *Wind:* {wind_kmph} km/h\n"
                f"⏱️ *Landfall:* {landfall_time}\n"
                f"📍 *Sector:* Visakhapatnam - Puri Coast\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"🛡️ *Status:* Validated Multi-Channel Trigger"
            )
            send_url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
            payload = json.dumps({"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}).encode('utf-8')
            urllib.request.urlopen(urllib.request.Request(send_url, data=payload, headers={'Content-Type': 'application/json'}))
            LAST_ALERT_TIME = now
            print(f"[{now.strftime('%H:%M:%S')}] 📱 TELEGRAM NOTIFICATION DELIVERED!")
    except Exception as e:
        print("Telegram error:", e)

    # Laptop Audio Siren
    try:
        import winsound
        winsound.Beep(1400, 300)
        cmd = 'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'Cyclone Red Alert Detected. Evacuate coastal lowlands immediately.\')"'
        subprocess.Popen(cmd, shell=True)
        print(f"[{now.strftime('%H:%M:%S')}] 🔊 LAPTOP SPEAKER SIREN & VOICE BROADCAST ACTIVATED!")
    except Exception:
        pass

if __name__ == "__main__":
    print("🌀 CycloVision V2 Sentinel Running Single Verification Sweep...")
    dispatch_sentinel_alert("CYCLONE-X (BOB-02)", 165.0, "Very Severe Cyclonic Storm (VSCS)", "Within 18h")
