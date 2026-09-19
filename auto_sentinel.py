import env_loader
import time
import json
import urllib.request
import urllib.parse
import base64
import os
import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# Configuration & Credentials loaded strictly from environment
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "+14155238886")
WHATSAPP_NUMBERS = [n.strip() for n in os.getenv("WHATSAPP_ALERT_NUMBERS", "").split(",") if n.strip()]
DEFAULT_ALERT_EMAIL = os.getenv("ALERT_RECIPIENT_EMAIL", os.getenv("GMAIL_USER", "advisory@cyclovision.in"))

CHAT_ID_FILE = os.path.join("v2", "data", "telegram_chat_id.txt")
LAST_ALERT_TIME = None
COOLDOWN_SECONDS = 60  # Responsive 1-minute cooldown window

import re

def save_persisted_chat_id(chat_id: str) -> bool:
    """Saves user's Telegram Chat ID with strict numeric validation."""
    clean_id = str(chat_id).strip()
    if not re.match(r'^-?\d{6,16}$', clean_id):
        print(f"  [ERROR] Invalid Telegram Chat ID rejected: {clean_id[:20]}")
        return False
    try:
        os.makedirs(os.path.dirname(CHAT_ID_FILE), exist_ok=True)
        with open(CHAT_ID_FILE, "w", encoding="utf-8") as f:
            f.write(clean_id)
        return True
    except Exception as e:
        print(f"Error saving chat_id: {e}")
        return False

def get_persisted_chat_id():
    # 0. Check environment variable
    env_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if env_id and re.match(r'^-?\d{6,16}$', env_id):
        return env_id

    # 1. Try reading from cache file
    if os.path.exists(CHAT_ID_FILE):
        try:
            with open(CHAT_ID_FILE, "r", encoding="utf-8") as f:
                cid = f.read().strip()
                if cid:
                    return cid
        except Exception:
            pass

    # 2. Try fetching from Telegram getUpdates and cache it
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or TOKEN
    if token:
        try:
            url = f"https://api.telegram.org/bot{token}/getUpdates"
            req = urllib.request.urlopen(url)
            data = json.loads(req.read().decode("utf-8"))
            if data.get("result"):
                cid = str(data["result"][-1]["message"]["chat"]["id"])
                os.makedirs(os.path.dirname(CHAT_ID_FILE), exist_ok=True)
                with open(CHAT_ID_FILE, "w", encoding="utf-8") as f:
                    f.write(cid)
                return cid
        except Exception:
            pass
    return None

def dispatch_telegram_alert(storm_name="CYCLONE-X", wind_kmph=165.0, stage="Very Severe Cyclonic Storm (VSCS)", landfall_time="Within 18 Hours"):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or TOKEN
    if not token or token.startswith("your_"):
        print("  [NOTICE] Telegram notice: TELEGRAM_BOT_TOKEN not configured in .env")
        return {"status": "NOT_CONFIGURED", "message": "Set TELEGRAM_BOT_TOKEN in .env"}

    chat_id = get_persisted_chat_id()
    if not chat_id:
        print("  [NOTICE] Telegram notice: No chat_id found yet. Please send /start or 'Hi' to your Telegram alert bot once!")
        return {"status": "WAITING_FOR_USER_START", "message": "Send /start to your bot or set TELEGRAM_CHAT_ID in .env"}

    msg = (
        f"🚨 *[CYCLOVISION V2 AUTONOMOUS RED ALERT]*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🌀 *Storm:* {storm_name}\n"
        f"⚠️ *Stage:* {stage}\n"
        f"💨 *Max Sustained Wind:* {wind_kmph} km/h\n"
        f"⏱️ *Estimated Landfall:* {landfall_time}\n"
        f"📍 *High-Risk Sector:* Visakhapatnam - Puri Coast\n"
        f"🌊 *Storm Surge:* 3.8m Inundation Threat\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🛡️ *Action Required:* Move immediately to designated cyclone shelters. Total deep-sea fishing ban.\n"
        f"📞 *Disaster Helpline:* 1070"
    )
    send_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}).encode("utf-8")
    try:
        req = urllib.request.Request(send_url, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [TELEGRAM] ALERT DELIVERED! (Chat ID: {chat_id})")
        return {"status": "DELIVERED", "chat_id": chat_id}
    except Exception as e:
        print("  [ERROR] Telegram error:", e)
        return {"status": "FAILED", "error": str(e)}

def dispatch_whatsapp_alerts(storm_name="CYCLONE-X", wind_kmph=165.0):
    sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip() or TWILIO_SID
    token = os.getenv("TWILIO_AUTH_TOKEN", "").strip() or TWILIO_TOKEN
    from_num = os.getenv("TWILIO_WHATSAPP_FROM", "+17372508034").strip()
    raw_nums = os.getenv("WHATSAPP_ALERT_NUMBERS", "").strip()
    numbers = [n.strip() for n in raw_nums.split(",") if n.strip()] or WHATSAPP_NUMBERS

    if not sid or not token or not numbers or sid.startswith("your_"):
        print("  [NOTICE] Twilio notice: Twilio credentials or WHATSAPP_ALERT_NUMBERS not configured in .env")
        return {"status": "NOT_CONFIGURED", "message": "Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and WHATSAPP_ALERT_NUMBERS in .env"}

    body_text = (
        f"🚨 *MoES NDMA CYCLONE RED ALERT*\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Severe Cyclone *{storm_name}* approaching with winds of *{wind_kmph} km/h*.\n"
        f"Sector: *Visakhapatnam - Puri Coast*.\n"
        f"Surge Hazard: *3.8m Inundation*.\n"
        f"Advisory: Evacuate lowlands immediately to concrete shelters.\n"
        f"State Helpline: *1070*"
    )

    clean_from = from_num if from_num.startswith("whatsapp:") else f"whatsapp:{from_num}"
    results = {}

    for num in numbers:
        clean_to = num if num.startswith("whatsapp:") else f"whatsapp:{num}"
        try:
            # 1. Try official Twilio Python Client
            try:
                from twilio.rest import Client
                client = Client(sid, token)
                msg = client.messages.create(from_=clean_from, to=clean_to, body=body_text)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [WHATSAPP] SUCCESS to {num}! (SID: {msg.sid[:12]}...)")
                results[num] = {"status": "SENT", "sid": msg.sid}
                continue
            except ImportError:
                pass

            # 2. Fallback to standard library urllib
            twilio_url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
            credentials = f"{sid}:{token}"
            base64_auth = base64.b64encode(credentials.encode("ascii")).decode("ascii")
            data = urllib.parse.urlencode({
                "From": clean_from,
                "To": clean_to,
                "Body": body_text
            }).encode("utf-8")
            
            req = urllib.request.Request(twilio_url, data=data)
            req.add_header("Authorization", f"Basic {base64_auth}")
            
            with urllib.request.urlopen(req) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                msg_sid = res_data.get("sid", "")[:12]
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [WHATSAPP] SUCCESS to {num}! (SID: {msg_sid}...)")
                results[num] = {"status": "SENT", "sid": msg_sid}
        except urllib.error.HTTPError as he:
            err_msg = he.read().decode("utf-8")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [NOTICE] Twilio response for {num}: {err_msg[:75]}...")
            results[num] = {"status": "NOTICE", "response": err_msg[:75]}
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] WhatsApp error for {num}: {e}")
            results[num] = {"status": "ERROR", "error": str(e)}
    return results

def dispatch_email_alert(storm_name="CYCLONE-X", wind_kmph=165.0, stage="Very Severe Cyclonic Storm (VSCS)", target_email=None):
    target_email = target_email or os.getenv("ALERT_RECIPIENT_EMAIL", "").strip() or os.getenv("GMAIL_USER", "").strip() or "advisory@cyclovision.in"
    subject = f"🚨 URGENT CYCLONE RED ALERT: {storm_name} ({stage})"
    body = (
        f"CYCLOVISION AI V2 — NATIONAL DECISION SUPPORT PLATFORM\n"
        f"===============================================================\n"
        f"Official Emergency Advisory issued by MoES / IMD Early Warning System\n\n"
        f"Storm Name: {storm_name}\n"
        f"Current Intensity Stage: {stage}\n"
        f"Maximum Sustained Surface Wind: {wind_kmph} km/h (Gusts to {round(wind_kmph*1.15, 1)} km/h)\n"
        f"Target Landfall Sector: Visakhapatnam - Puri Coast\n"
        f"Inundation Hazard: 3.5m - 4.2m Storm Surge Threat\n\n"
        f"Recommended Action:\n"
        f"- Move coastal population to multi-purpose cyclone shelters.\n"
        f"- Total suspension of fishing and marine operations.\n"
        f"- State Emergency Operations Centre (SEOC) Helpline: 1070\n\n"
        f"Data Provenance: ISRO MOSDAC INSAT-3D/3DS + NOAA IBTrACS v4 (DOI: 10.25921/82ty-9e16)\n"
        f"===============================================================\n"
    )
    
    email_path = os.path.join("v2", "data", "latest_email_alert.txt")
    os.makedirs(os.path.dirname(email_path), exist_ok=True)
    with open(email_path, "w", encoding="utf-8") as f:
        f.write(f"To: {target_email}\nSubject: {subject}\n\n{body}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [EMAIL] BULLETIN PREPARED & LOGGED to {email_path}!")

    smtp_user = os.getenv("GMAIL_USER")
    smtp_pass = os.getenv("GMAIL_APP_PASSWORD")
    if smtp_user and smtp_pass:
        try:
            msg = MIMEMultipart()
            msg["From"] = smtp_user
            msg["To"] = target_email
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as server:
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [EMAIL] DIRECTLY DISPATCHED via Gmail SMTP to {target_email}!")
            return {"status": "DISPATCHED_SMTP", "recipient": target_email}
        except Exception as e:
            print(f"  [NOTICE] SMTP notice: {e}")
    return {"status": "PREPARED_AND_LOGGED", "file": email_path, "recipient": target_email}

def play_audio_siren(storm_name="CYCLONE-X", wind_kmph=165.0):
    try:
        import winsound
        winsound.Beep(1400, 300)
        winsound.Beep(1800, 450)
        # Sanitize parameters strictly to eliminate command injection
        safe_name = re.sub(r'[^a-zA-Z0-9\s\-()]', '', str(storm_name)).strip() or "CYCLONE"
        safe_wind = round(float(wind_kmph), 1)
        spoken_text = f"Warning! CycloVision AI detected {safe_name} with wind speed of {safe_wind} kilometers per hour. Emergency red alert active."
        
        ps_cmd = [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "$txt = [Console]::In.ReadLine(); Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak($txt)"
        ]
        proc = subprocess.Popen(ps_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            proc.communicate(input=spoken_text + "\n", timeout=4)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [AUDIO] LAPTOP SPEAKER SIREN & VOICE BROADCAST PLAYED!")
        return {"status": "DISPATCHED", "sanitized_storm": safe_name, "spoken_text": spoken_text}
    except Exception as e:
        print(f"  [NOTICE] Audio siren notice: {e}")
        return {"status": "ERROR", "error": str(e)}

def run_sentinel_sweep(storm_name="CYCLONE-X (BOB-02)", wind_kmph=165.0, stage="Very Severe Cyclonic Storm (VSCS)", force=False):
    global LAST_ALERT_TIME
    now = datetime.now()
    if not force and LAST_ALERT_TIME and (now - LAST_ALERT_TIME).total_seconds() < COOLDOWN_SECONDS:
        print(f"[{now.strftime('%H:%M:%S')}] [COOLDOWN] Alert suppressed by anti-spam cooldown window.")
        return {"status": "COOLDOWN_ACTIVE"}

    print("=" * 70)
    print("CYCLOVISION AI - MULTI-CHANNEL DISASTER SENTINEL ACTIVE")
    print(f"Sweep Timestamp: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)

    # 1. Telegram
    t_res = dispatch_telegram_alert(storm_name, wind_kmph, stage, "Within 18 Hours")
    
    # 2. WhatsApp
    w_res = dispatch_whatsapp_alerts(storm_name, wind_kmph)
    
    # 3. Email
    e_res = dispatch_email_alert(storm_name, wind_kmph, stage)
    
    # 4. Audio Siren
    play_audio_siren(storm_name, wind_kmph)
    
    LAST_ALERT_TIME = now
    return {
        "status": "SENTINEL_SWEEP_COMPLETED",
        "telegram": t_res,
        "whatsapp": w_res,
        "email": e_res
    }

def dispatch_sentinel_alert(storm_name="CYCLONE-X (BOB-02)", wind_kmph=165.0, stage="Very Severe Cyclonic Storm (VSCS)", landfall_time="Within 18h"):
    """Backwards-compatible sentinel dispatcher for app.py and legacy callers."""
    return run_sentinel_sweep(storm_name=storm_name, wind_kmph=wind_kmph, stage=stage, force=True)

if __name__ == "__main__":
    run_sentinel_sweep(force=True)

