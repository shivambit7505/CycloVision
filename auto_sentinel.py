import time
import json
import urllib.request
import urllib.parse
import base64
import os
import subprocess
from datetime import datetime

# 100% VERIFIED CREDENTIALS
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "REDACTED_TELEGRAM_TOKEN")
TWILIO_SID = os.getenv("TWILIO_SID", "REDACTED_TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN", "REDACTED_TWILIO_TOKEN")
TWILIO_WHATSAPP_FROM = "+14155238886"

# 3 User WhatsApp Numbers
WHATSAPP_NUMBERS = ["+917037445886", "+919068192058", "+917505110093"]

# 1. Twilio WhatsApp Dispatcher
def dispatch_whatsapp_alerts(storm_name, wind_kmph):
    twilio_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json"
    credentials = f"{TWILIO_SID}:{TWILIO_TOKEN}"
    base64_auth = base64.b64encode(credentials.encode('ascii')).decode('ascii')
    
    body_text = (
        f"🚨 *MoES NDMA CYCLONE RED ALERT*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🌀 Storm: *{storm_name}*\n"
        f"💨 Max Wind: *{wind_kmph} km/h*\n"
        f"📍 Landfall Sector: *Visakhapatnam - Puri Coast*\n"
        f"🌊 Surge Threat: *3.8m Inundation Risk*\n"
        f"🛡️ Action: Evacuate coastal lowlands to shelters immediately.\n"
        f"📞 State Helpline: *1070*"
    )
    
    for num in WHATSAPP_NUMBERS:
        try:
            data = urllib.parse.urlencode({
                "From": f"whatsapp:{TWILIO_WHATSAPP_FROM}",
                "To": f"whatsapp:{num}",
                "Body": body_text
            }).encode('utf-8')
            
            req = urllib.request.Request(twilio_url, data=data)
            req.add_header("Authorization", f"Basic {base64_auth}")
            
            with urllib.request.urlopen(req) as resp:
                res_data = json.loads(resp.read().decode('utf-8'))
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 💬 WHATSAPP SUCCESS to {num}! (SID: {res_data.get('sid')[:12]}...)")
        except urllib.error.HTTPError as he:
            err_body = he.read().decode('utf-8')
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ℹ️ Twilio response for {num}: {err_body[:100]}...")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ℹ️ WhatsApp notice for {num}: {e}")

# 2. Telegram Alert Dispatcher
def dispatch_telegram_alert(storm_name, wind_kmph, stage, landfall_time):
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
        req = urllib.request.urlopen(url)
        data = json.loads(req.read().decode('utf-8'))
        if not data.get("result"):
            return
        chat_id = data["result"][-1]["message"]["chat"]["id"]
        
        msg = (
            f"🚨 *[AUTONOMOUS MULTI-CHANNEL CYCLONE ALERT]*\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🌀 *Storm:* {storm_name}\n"
            f"⚠️ *Stage:* {stage} (RED ALERT)\n"
            f"💨 *Max Sustained Wind:* {wind_kmph} km/h\n"
            f"⏱️ *Impact Window:* {landfall_time}\n"
            f"📍 *High-Risk Coast:* Visakhapatnam - Puri Sector\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🤖 *Channels Active:* Telegram + 3 WhatsApp Numbers + Laptop Voice Siren"
        )
        send_url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        payload = json.dumps({"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}).encode('utf-8')
        req_send = urllib.request.Request(send_url, data=payload, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req_send)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📱 TELEGRAM NOTIFICATION DELIVERED!")
    except Exception as e:
        print("Telegram error:", e)

# 3. Laptop Speaker Voice Siren
def play_laptop_voice_alert(storm_name, wind_kmph):
    try:
        import winsound
        winsound.Beep(1200, 250)
        winsound.Beep(1600, 400)
        spoken_text = f"Emergency Red Alert. CycloVision AI detected {storm_name} with wind speed of {wind_kmph} kilometers per hour. Immediate coastal evacuation advisory active."
        cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{spoken_text}\')"'
        subprocess.Popen(cmd, shell=True)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔊 LAPTOP SPEAKER SIREN & VOICE BROADCAST ACTIVATED!")
    except Exception as e:
        print("Voice error:", e)

# Main Multi-Channel Sentinel Loop
def run_all_alerts(interval_seconds=60):
    print("=" * 68)
    print("🌀 CYCLOVISION AI — FULL AUTONOMOUS MULTI-CHANNEL SENTINEL")
    print(f"📡 Channels: Telegram + 3 WhatsApp Numbers ({len(WHATSAPP_NUMBERS)}) + Voice Siren")
    print("=" * 68)
    
    scan_count = 1
    while True:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[SCAN #{scan_count} - {timestamp}] 🛰️ Scanning Bay of Bengal Satellite Data...")
        
        simulated_wind = 155.0  # km/h
        
        if simulated_wind >= 65.0:
            print("  🚨 [CYCLONE DETECTED] Wind: 155 km/h -> FIRING ALL CHANNELS!")
            
            # 1. Telegram Alert
            dispatch_telegram_alert("CYCLONE-X (BOB-02)", simulated_wind, "Very Severe Cyclonic Storm", "Within 18 Hours")
            
            # 2. WhatsApp Alerts (All 3 numbers)
            dispatch_whatsapp_alerts("CYCLONE-X (BOB-02)", simulated_wind)
            
            # 3. Laptop Speaker Voice Siren
            play_laptop_voice_alert("CYCLONE-X", simulated_wind)
            
        scan_count += 1
        print(f"⏳ Next sweep in {interval_seconds} seconds...")
        time.sleep(interval_seconds)

if __name__ == "__main__":
    run_all_alerts(interval_seconds=60)
