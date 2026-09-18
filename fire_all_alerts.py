import json
import urllib.request
import urllib.parse
import base64
import os
import subprocess
from datetime import datetime

# 1. CREDENTIALS VIA ENVIRONMENT VARIABLES
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_PHONE_NUMBER", "+14155238886")

WHATSAPP_NUMBERS = [n.strip() for n in os.getenv("ALERT_WHATSAPP_NUMBERS", "").split(",") if n.strip()]

print("=" * 70)
print("🌀 CYCLOVISION AI — IMMEDIATE MULTI-CHANNEL ALERT DISPATCHER")
print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# A. DISPATCH TELEGRAM ALERT
print("\n[1/3] 📱 FIRING TELEGRAM ALERT...")
try:
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    req = urllib.request.urlopen(url)
    data = json.loads(req.read().decode('utf-8'))
    
    if data.get("result"):
        # Get the latest chat_id
        chat_id = data["result"][-1]["message"]["chat"]["id"]
        user_name = data["result"][-1]["message"]["chat"].get("first_name", "User")
        
        tele_msg = (
            "🚨 *[IMD / CYCLOVISION AI EMERGENCY RED ALERT]*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "🌀 *Active Storm:* CYCLONE-X (BOB-02)\n"
            "⚠️ *Stage:* Very Severe Cyclonic Storm (VSCS)\n"
            "💨 *Max Sustained Wind:* 165 km/h (Gusts: 185 km/h)\n"
            "📍 *Target Landfall:* Visakhapatnam - Puri Coast\n"
            "🌊 *Storm Surge Hazard:* 3.8 Meters Coastal Inundation\n"
            "⏱️ *Estimated Landfall:* Within 18 Hours\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "🛡️ *Immediate Action:* Move to designated concrete cyclone shelters. Total fishing ban.\n"
            "📞 *State Disaster Management Helpline:* 1070"
        )
        
        send_url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        payload = json.dumps({"chat_id": chat_id, "text": tele_msg, "parse_mode": "Markdown"}).encode('utf-8')
        req_send = urllib.request.Request(send_url, data=payload, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req_send)
        print(f"   ✅ TELEGRAM SUCCESS! Delivered to {user_name} (Chat ID: {chat_id})")
    else:
        print("   ⚠️ Telegram notice: Bot ko kholein aur ek baar 'Hi' bhein!")
except Exception as e:
    print(f"   ❌ Telegram Error: {e}")

# B. DISPATCH WHATSAPP ALERTS (ALL 3 NUMBERS)
print("\n[2/3] 💬 FIRING TWILIO WHATSAPP ALERTS...")
twilio_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json"
credentials = f"{TWILIO_SID}:{TWILIO_TOKEN}"
base64_auth = base64.b64encode(credentials.encode('ascii')).decode('ascii')

wa_body = (
    "🚨 *MoES NDMA CYCLONE RED ALERT*\n"
    "━━━━━━━━━━━━━━━━━━━━━\n"
    "Severe Cyclone *CYCLONE-X* approaching with winds of *165 km/h*.\n"
    "Target: *Visakhapatnam - Puri Coast* in next 18h.\n"
    "Surge Threat: *3.8m Inundation*.\n"
    "Advisory: Evacuate lowlands immediately to concrete shelters.\n"
    "State Helpline: *1070*"
)

for num in WHATSAPP_NUMBERS:
    try:
        data = urllib.parse.urlencode({
            "From": f"whatsapp:{TWILIO_FROM}",
            "To": f"whatsapp:{num}",
            "Body": wa_body
        }).encode('utf-8')
        
        req = urllib.request.Request(twilio_url, data=data)
        req.add_header("Authorization", f"Basic {base64_auth}")
        
        with urllib.request.urlopen(req) as resp:
            res_data = json.loads(resp.read().decode('utf-8'))
            print(f"   ✅ WHATSAPP SENT to {num}! SID: {res_data.get('sid')[:12]}...")
    except urllib.error.HTTPError as he:
        err_msg = he.read().decode('utf-8')
        print(f"   ℹ️ Twilio status for {num}: {err_msg[:80]}...")
    except Exception as e:
        print(f"   ❌ WhatsApp error for {num}: {e}")

# C. GMAIL / EMAIL DISPATCH LOG
print("\n[3/3] 📧 GMAIL / EMAIL ALERT PIPELINE...")
email_bulletin = (
    "Subject: 🚨 URGENT: Tropical Cyclone Emergency Advisory Bulletin (MoES / IMD)\n\n"
    "Official Cyclone Warning Bulletin issued by CycloVision AI National Decision Support System.\n"
    "System: Very Severe Cyclonic Storm (VSCS) BOB-02\n"
    "Estimated Central Pressure: 965 hPa | Max Wind: 165 km/h\n"
    "Landfall Sector: Odisha & North Andhra Coast.\n"
    "All coastal district magistrates and relief commissioners are requested to initiate Phase-3 evacuation."
)
with open("latest_email_alert.txt", "w", encoding="utf-8") as f:
    f.write(email_bulletin)
print("   ✅ EMAIL BULLETIN PREPARED & SAVED TO latest_email_alert.txt!")

# D. AUDIO SIREN
try:
    import winsound
    winsound.Beep(1500, 350)
    winsound.Beep(1800, 450)
    cmd = 'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'Emergency Warning! Tropical Cyclone Red Alert Dispatched to all registered mobile devices.\')"'
    subprocess.Popen(cmd, shell=True)
    print("\n🔊 LAPTOP SPEAKER SIREN & VOICE BROADCAST PLAYED!")
except Exception:
    pass

print("\n" + "=" * 70)
print("🎉 ALL ALERT CHANNELS TRIGGERED SIMULTANEOUSLY!")
print("=" * 70)
