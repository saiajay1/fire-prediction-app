import time
from typing import Any, Dict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Fire Prediction (3 Models)", page_icon="🔥", layout="wide")

# ---- Endpoints ----
M1_URL = "https://cz6vmkmp6tnrkhojlpb3xsfw6i0icyqd.lambda-url.us-east-1.on.aws/"
M2_URL = "https://rnmsxp5s53.us-east-1.awsapprunner.com/predict_features"
M3_URL = "https://mfyemzf28h.us-east-1.awsapprunner.com/predict"

# ---------- Prefills: Model 1 (Saafe model) ----------
M1_NON_FIRE = {
  "frame": 5678,
  "timestamp": "2025-09-08T12:45:00Z",
  "features": {
    "t_mean": 24.0, "t_std": 0.5, "t_max": 28.0, "t_p95": 27.5,
    "t_hot_area_pct": 0.2, "t_hot_largest_blob_pct": 0.1,
    "t_grad_mean": 0.05, "t_grad_std": 0.02, "t_diff_mean": 0.03, "t_diff_std": 0.01,
    "flow_mag_mean": 0.1, "flow_mag_std": 0.01,
    "tproxy_val": 28.0, "tproxy_delta": 0.2, "tproxy_vel": 0.05,
    "CO": 0.2, "VOC": 0.5, "NO2": 0.01,
    "CO_diff": 0.02, "VOC_diff": 0.03, "NO2_diff": 0.0,
    "VOC_ma5": 0.4, "CO_ma5": 0.15, "NO2_ma5": 0.01,
    "VOC_z": 0.1, "CO_z": 0.1, "NO2_z": 0.0,
    "temp_rise_c_per_min": 0.2, "temp_slope_30s": 0.1,
    "gas_var_30s": 0.05, "delta_temp_30s": 0.2, "delta_gas_10s": 0.01,
    "spike_count_voc_2m": 0,
    "temp_co_corr_lag_0s": 0.10, "temp_co_corr_lag_15s": 0.08, "temp_co_corr_lag_60s": 0.05,
    "temp_voc_corr_lag_0s": 0.12, "temp_voc_corr_lag_15s": 0.10, "temp_voc_corr_lag_60s": 0.08,
    "temp_co_xcorr_max_abs": 0.15, "temp_voc_xcorr_max_abs": 0.18,
    "is_weekend": 0, "asleep_window": 0,
    "hrblk_0": 0, "hrblk_1": 0, "hrblk_2": 0, "hrblk_3": 0, "hrblk_4": 0, "hrblk_5": 0
  },
  "decision_threshold": 0.4
}
M1_FIRE = {
  "frame": 1234,
  "timestamp": "2025-09-08T12:34:56Z",
  "features": {
    "t_mean": 28.12, "t_std": 0.83, "t_max": 74.56, "t_p95": 71.92,
    "t_hot_area_pct": 8.20, "t_hot_largest_blob_pct": 5.47,
    "t_grad_mean": 0.42, "t_grad_std": 0.25, "t_diff_mean": 0.18, "t_diff_std": 0.09,
    "flow_mag_mean": 0.50, "flow_mag_std": 0.05,
    "tproxy_val": 74.56, "tproxy_delta": 1.32, "tproxy_vel": 0.87,
    "CO": 0.9, "VOC": 2.5, "NO2": 0.03,
    "CO_diff": 0.30, "VOC_diff": 0.40, "NO2_diff": -0.01,
    "VOC_ma5": 2.10, "CO_ma5": 0.75, "NO2_ma5": 0.02,
    "VOC_z": 2.2, "CO_z": 1.1, "NO2_z": -0.2,
    "temp_rise_c_per_min": 12.5, "temp_slope_30s": 3.2,
    "gas_var_30s": 0.45, "delta_temp_30s": 8.7, "delta_gas_10s": 0.6,
    "spike_count_voc_2m": 4,
    "temp_co_corr_lag_0s": 0.72, "temp_co_corr_lag_15s": 0.68, "temp_co_corr_lag_60s": 0.55,
    "temp_voc_corr_lag_0s": 0.81, "temp_voc_corr_lag_15s": 0.77, "temp_voc_corr_lag_60s": 0.60,
    "temp_co_xcorr_max_abs": 0.74, "temp_voc_xcorr_max_abs": 0.83,
    "is_weekend": 0, "asleep_window": 1,
    "hrblk_0": 0, "hrblk_1": 0, "hrblk_2": 0, "hrblk_3": 0, "hrblk_4": 1, "hrblk_5": 0
  },
  "decision_threshold": 0.4
}

# ---------- Prefills: Model 2 (18 Features research data) ----------
M2_NON_FIRE = {
  "data": {
    "features_dict": {
      "t_mean": 28.0, "t_std": 2.0, "t_max": 32.0, "t_p95": 31.0,
      "t_hot_area_pct": 0.2, "t_hot_largest_blob_pct": 0.0,
      "t_grad_mean": 0.5, "t_grad_std": 0.2,
      "t_diff_mean": 0.1, "t_diff_std": 0.05,
      "flow_mag_mean": 0.1, "flow_mag_std": 0.05,
      "gas_val": 400.0, "gas_delta": 5.0, "gas_vel": 0.5,
      "tproxy_val": 32.0, "tproxy_delta": 1.0, "tproxy_vel": 0.2
    }
  },
  "threshold": 0.5
}
M2_FIRE = {
  "data": {
    "features_dict": {
      "t_mean": 105.0, "t_std": 15.0, "t_max": 160.0, "t_p95": 150.0,
      "t_hot_area_pct": 40.0, "t_hot_largest_blob_pct": 30.0,
      "t_grad_mean": 12.0, "t_grad_std": 6.0,
      "t_diff_mean": 8.0, "t_diff_std": 4.0,
      "flow_mag_mean": 5.0, "flow_mag_std": 2.0,
      "gas_val": 2500.0, "gas_delta": 600.0, "gas_vel": 600.0,
      "tproxy_val": 160.0, "tproxy_delta": 20.0, "tproxy_vel": 20.0
    }
  },
  "threshold": 0.5
}

# ---------- Prefills: Model 3 (Kaggle Base model) ----------
M3_NON_FIRE = {
  "data": {
    "Temperature[C]": 23.5, "Humidity[%]": 42, "TVOC[ppb]": 3, "eCO2[ppm]": 420,
    "PM1.0": 1.2, "PM2.5": 2.3, "PM10": 3.4, "Pressure[hPa]": 1013.2,
    "Raw H2": 14500, "Raw Ethanol": 21000, "CNT": 0, "UTC": 0,
    "NC0.5": 0, "NC1.0": 0, "NC2.5": 0
  }
}
M3_FIRE = {
  "data": {
    "Temperature[C]": 45.7, "Humidity[%]": 15.3, "TVOC[ppb]": 850, "eCO2[ppm]": 2200,
    "PM1.0": 80.1, "PM2.5": 120.5, "PM10": 155.0, "Pressure[hPa]": 1002.1,
    "Raw H2": 30000, "Raw Ethanol": 42000, "CNT": 123456, "UTC": 1623859200,
    "NC0.5": 3500, "NC1.0": 2100, "NC2.5": 1500
  }
}

# ---------- Helpers ----------
def coerce_value(v: Any) -> Any:
    if isinstance(v, str):
        if v.strip() == "":
            return v
        try:
            if "." in v or "e" in v.lower():
                return float(v)
            return int(v)
        except Exception:
            return v
    return v

def df_from_features_dict(feats: Dict[str, Any]) -> pd.DataFrame:
    items = sorted(feats.items(), key=lambda kv: kv[0].lower())
    return pd.DataFrame([{"feature": k, "value": v} for k, v in items])

def features_dict_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    out = {}
    for _, row in df.iterrows():
        key = str(row.get("feature", "")).strip()
        if not key:
            continue
        out[key] = coerce_value(row.get("value"))
    return out

def post_json(url: str, payload: Dict[str, Any], timeout_s: float = 25.0) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()

def colored_box(text: str, bg: str, border: str = "#00000020", color: str = "#111"):
    st.markdown(
        f"""
        <div style="
            padding: 12px 14px; border-radius: 8px;
            background: {bg}; color: {color}; border: 1px solid {border};
            font-weight: 500;">
            {text}
        </div>
        """, unsafe_allow_html=True
    )

def html_summary_table(rows, key_to_color=None):
    """
    rows: List[Tuple[field, value]]
    key_to_color: dict like {"Label": ("red"|"green"|"none", condition_bool)} is ignored;
                  we color based on value semantics below.
    """
    def cell_style(field, val):
        # Model 1: Label
        if field.lower() == "label":
            if isinstance(val, str) and val.strip().lower() == "fire":
                return "background:#fde8e8;color:#7a1111;font-weight:600;"
            if isinstance(val, str) and val.strip().lower() == "not fire":
                return "background:#e6f4ea;color:#0b5b25;font-weight:600;"
        # Model 2: fire_detected
        if field == "fire_detected":
            if val is True:
                return "background:#fde8e8;color:#7a1111;font-weight:600;"
            if val is False:
                return "background:#e6f4ea;color:#0b5b25;font-weight:600;"
        # Model 3: fire_prediction
        if field == "fire_prediction":
            if val is True:
                return "background:#fde8e8;color:#7a1111;font-weight:600;"
            if val is False:
                return "background:#e6f4ea;color:#0b5b25;font-weight:600;"
        return ""
    html = ['<table style="width:100%;border-collapse:collapse;">']
    for field, val in rows:
        style = cell_style(field, val)
        html.append(
            f'<tr>'
            f'<td style="border:1px solid #ddd;padding:8px;width:35%;font-weight:600;background:#fafafa;">{field}</td>'
            f'<td style="border:1px solid #ddd;padding:8px;{style}">{val}</td>'
            f'</tr>'
        )
    html.append('</table>')
    st.markdown("".join(html), unsafe_allow_html=True)

# ========================= Email Notification System =========================
# Session state for email deduplication
if 'last_fire_alerts' not in st.session_state:
    st.session_state.last_fire_alerts = {}

def should_send_alert(model_name: str, prediction_data: Dict[str, Any]) -> bool:
    """Check if we should send an alert to avoid duplicates"""
    # Create a simple hash of the prediction data
    pred_hash = hash(str(sorted(prediction_data.items())))
    
    # Check if we've already sent an alert for this prediction
    last_alert = st.session_state.last_fire_alerts.get(model_name)
    if last_alert == pred_hash:
        return False
    
    # Update the last alert hash
    st.session_state.last_fire_alerts[model_name] = pred_hash
    return True

def send_fire_alert_email(model_name: str, prediction_data: Dict[str, Any], email_config: Dict[str, str]):
    """Send an email alert when fire is detected"""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_config['sender_email']
        msg['To'] = email_config['recipient_email']
        msg['Subject'] = f"🔥 FIRE ALERT - {model_name} Detected Fire"

        # Format prediction data for email
        prediction_str = "\n".join([f"{k}: {v}" for k, v in prediction_data.items()])

        # Create email body
        body = f"""
FIRE DETECTED by {model_name}!

Prediction Details:
{prediction_str}

Time: {time.strftime('%Y-%m-%d %H:%M:%S')}

Please take immediate action.
"""
        
        msg.attach(MIMEText(body, 'plain'))

        # Create SMTP session
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email_config['sender_email'], email_config['sender_password'])
        
        # Send email
        text = msg.as_string()
        server.sendmail(email_config['sender_email'], email_config['recipient_email'], text)
        server.quit()
        
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

# ========================= UI =========================
st.title("🔥 Fire Prediction — Live data")

# ========================= EMAIL CONFIGURATION =========================
st.sidebar.header("📧 Email Notifications")
enable_email = st.sidebar.checkbox("Enable Email Alerts", value=False)
sender_email = st.sidebar.text_input("Sender Email (Gmail)", placeholder="your_email@gmail.com")
sender_password = st.sidebar.text_input("App Password", type="password", placeholder="Gmail App Password")
recipient_email = st.sidebar.text_input("Recipient Email", value="ch.ajay1707@gmail.com")

# Test email configuration
if st.sidebar.button("Test Email Configuration"):
    if sender_email and sender_password and recipient_email:
        test_config = {
            'sender_email': sender_email,
            'sender_password': sender_password,
            'recipient_email': recipient_email
        }
        # Send test email
        test_data = {"status": "Test email from Fire Prediction App", "result": "Configuration successful"}
        email_sent = send_fire_alert_email("Test Notification", test_data, test_config)
        if email_sent:
            st.sidebar.success("✅ Test email sent successfully!")
        else:
            st.sidebar.error("❌ Failed to send test email. Check your configuration.")
    else:
        st.sidebar.warning("Please fill in all email fields first.")

st.sidebar.caption("Note: Use Gmail App Passwords for security. Enable 2FA and generate an app password.")
st.sidebar.markdown("""
    **Setup Instructions:**
    1. Enable 2-Factor Authentication on your Gmail account
    2. Generate an App Password in your Google Account settings
    3. Enter your Gmail address and App Password above
    4. Verify the recipient email is correct
    5. Click 'Test Email Configuration' to verify
    6. Enable email alerts
""")

# Store email config in session state
email_config = {
    'sender_email': sender_email,
    'sender_password': sender_password,
    'recipient_email': recipient_email
}

tabs = st.tabs(["Live data fire prediction", "18 Features research data", "Kaggle Base model"])

# ========================= Model 1 =========================
with tabs[0]:
    st.subheader("Fire prediction live data")
    left, right = st.columns([1, 1])
    with left:
        m1_prefill = st.selectbox("Prefill", ["None", "Non-Fire sample", "Fire sample"], index=1, key="m1_prefill")
    with right:
        m1_url = st.text_input("Endpoint", value=M1_URL, key="m1_endpoint")

    # Session defaults
    if "m1_frame" not in st.session_state:
        st.session_state.m1_frame = M1_NON_FIRE["frame"]
    if "m1_timestamp" not in st.session_state:
        st.session_state.m1_timestamp = M1_NON_FIRE["timestamp"]
    if "m1_threshold" not in st.session_state:
        st.session_state.m1_threshold = M1_NON_FIRE["decision_threshold"]
    if "m1_table" not in st.session_state:
        st.session_state.m1_table = df_from_features_dict(M1_NON_FIRE["features"])

    # Prefill
    src1 = M1_NON_FIRE if m1_prefill == "Non-Fire sample" else M1_FIRE if m1_prefill == "Fire sample" else None
    if src1 is not None:
        st.session_state.m1_frame = src1["frame"]
        st.session_state.m1_timestamp = src1["timestamp"]
        st.session_state.m1_threshold = src1.get("decision_threshold", 0.4)
        st.session_state.m1_table = df_from_features_dict(src1["features"])

    st.markdown("**Metadata**")
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c1:
        st.session_state.m1_frame = st.number_input("frame", value=int(st.session_state.m1_frame), step=1, key="m1_frame_input")
    with c2:
        st.session_state.m1_timestamp = st.text_input("timestamp (ISO 8601, Zulu)", value=st.session_state.m1_timestamp, key="m1_ts_input")
    with c3:
        st.session_state.m1_threshold = st.number_input("decision_threshold", value=float(st.session_state.m1_threshold), step=0.05, format="%.4f", key="m1_thr_input")

    st.markdown("**Features (editable table)**")
    st.caption("Edit feature names and values. Add/remove rows as needed.")
    st.session_state.m1_table = st.data_editor(
        st.session_state.m1_table.copy(),
        num_rows="dynamic", use_container_width=True, hide_index=True, key="m1_features_editor"
    )

    m1_run = st.button("Prediction (Saafe model)", type="primary", use_container_width=True, key="m1_run_btn")
    st.divider()

    if m1_run:
        try:
            payload1 = {
                "frame": int(st.session_state.m1_frame),
                "timestamp": st.session_state.m1_timestamp,
                "features": features_dict_from_df(st.session_state.m1_table),
                "decision_threshold": float(st.session_state.m1_threshold),
            }
            with st.spinner("Calling AWS Lambda..."):
                t0 = time.time()
                resp1 = post_json(m1_url, payload1, timeout_s=25.0)
                elapsed = time.time() - t0

            st.success(f"Response OK ({elapsed:.2f}s)")

            # Colored banner + summary table (Label cell colored)
            pred = resp1.get("prediction", {}) or {}
            lbl = str(pred.get("label", "")).lower()
            fire_detected = "fire" in lbl and "not" not in lbl
            
            if fire_detected:
                colored_box("🔥 Fire detected by Saafe model", bg="#fde8e8", border="#f5c2c2", color="#7a1111")
                # Send email alert if enabled and not a duplicate
                if enable_email and sender_email and sender_password and recipient_email:
                    if should_send_alert("Saafe Model", pred):
                        email_sent = send_fire_alert_email("Saafe Model", pred, email_config)
                        if email_sent:
                            st.success("📧 Fire alert email sent successfully!")
                        else:
                            st.error("📧 Failed to send fire alert email. Check your email configuration.")
                    else:
                        st.info("📧 Fire detected but email alert was already sent for this prediction.")
            else:
                colored_box("✅ Not Fire (Saafe model)", bg="#e6f4ea", border="#b7e1c1", color="#0b5b25")

            st.subheader("Prediction Summary")
            summary_rows = [
                ("Label", str(pred.get("label", "—"))),
                ("Fire probability", f"{pred.get('fire_probability'):.6f}" if isinstance(pred.get("fire_probability"), (int, float)) else str(pred.get("fire_probability"))),
                ("Frame", resp1.get("frame", payload1["frame"])),
                ("Timestamp", resp1.get("timestamp", payload1["timestamp"])),
            ]
            html_summary_table(summary_rows)

            # Explanation tables — Local first, then Global
            expl = resp1.get("explanation", {}) or {}
            lcontrib = expl.get("local_contributions", [])
            gtf = expl.get("global_top_features", [])

            if isinstance(lcontrib, list) and lcontrib:
                st.subheader("Local Contributions")
                st.dataframe(pd.DataFrame(lcontrib), use_container_width=True)

            if isinstance(gtf, list) and gtf:
                st.subheader("Global Top Features")
                st.dataframe(pd.DataFrame(gtf), use_container_width=True)

            notes = expl.get("notes")
            if notes:
                colored_box(f"📝 {notes}", bg="#fff8db", border="#f4e7a5", color="#6b5b00")

        except requests.HTTPError as e:
            st.error(f"HTTP error: {e}\n{getattr(e, 'response', None) and e.response.text}")
        except Exception as e:
            st.error(f"Error: {e}")

# ========================= Model 2 =========================
with tabs[1]:
    st.subheader("18 Features research data")
    left, right = st.columns([1, 1])
    with left:
        m2_prefill = st.selectbox("Prefill", ["None", "Non-Fire sample", "Fire sample"], index=1, key="m2_prefill")
    with right:
        m2_url = st.text_input("Endpoint", value=M2_URL, key="m2_endpoint")

    if "m2_threshold" not in st.session_state:
        st.session_state.m2_threshold = float(M2_NON_FIRE["threshold"])
    if "m2_table" not in st.session_state:
        st.session_state.m2_table = df_from_features_dict(M2_NON_FIRE["data"]["features_dict"])

    src2 = M2_NON_FIRE if m2_prefill == "Non-Fire sample" else M2_FIRE if m2_prefill == "Fire sample" else None
    if src2 is not None:
        st.session_state.m2_threshold = float(src2.get("threshold", 0.5))
        st.session_state.m2_table = df_from_features_dict(src2["data"]["features_dict"])

    st.markdown("**Threshold**")
    st.session_state.m2_threshold = st.number_input("threshold", value=float(st.session_state.m2_threshold), step=0.05, format="%.2f", key="m2_thr_input")

    st.markdown("**Features (editable table)**")
    st.caption("Exactly 18 core features expected by this API.")
    st.session_state.m2_table = st.data_editor(
        st.session_state.m2_table.copy(),
        num_rows="dynamic", use_container_width=True, hide_index=True, key="m2_features_editor"
    )

    m2_run = st.button("Prediction (18 Features)", type="primary", use_container_width=True, key="m2_run_btn")
    st.divider()

    if m2_run:
        try:
            payload2 = {
                "data": {"features_dict": features_dict_from_df(st.session_state.m2_table)},
                "threshold": float(st.session_state.m2_threshold),
            }
            with st.spinner("Calling 18-Features API..."):
                t0 = time.time()
                resp2 = post_json(m2_url, payload2, timeout_s=25.0)
                elapsed = time.time() - t0

            st.success(f"Response OK ({elapsed:.2f}s)")

            # Banner FIRST (as requested)
            fire_detected = bool(resp2.get("fire_detected", False))
            if fire_detected:
                colored_box("🔥 Fire detected (18 Features)", bg="#fde8e8", border="#f5c2c2", color="#7a1111")
                # Send email alert if enabled and not a duplicate
                if enable_email and sender_email and sender_password and recipient_email:
                    if should_send_alert("18 Features Model", resp2):
                        email_sent = send_fire_alert_email("18 Features Model", resp2, email_config)
                        if email_sent:
                            st.success("📧 Fire alert email sent successfully!")
                        else:
                            st.error("📧 Failed to send fire alert email. Check your email configuration.")
                    else:
                        st.info("📧 Fire detected but email alert was already sent for this prediction.")
            else:
                colored_box("✅ Not Fire (18 Features)", bg="#e6f4ea", border="#b7e1c1", color="#0b5b25")

            # Summary table with colored fire_detected cell
            st.subheader("Prediction Summary")
            rows = [
                ("fire_detected", bool(resp2.get("fire_detected", False))),
                ("score", f"{resp2.get('score'):.6f}" if isinstance(resp2.get("score"), (int, float)) else str(resp2.get("score"))),
                ("latency_ms", f"{resp2.get('latency_ms'):.3f}" if isinstance(resp2.get("latency_ms"), (int, float)) else str(resp2.get("latency_ms"))),
            ]
            html_summary_table(rows)

        except requests.HTTPError as e:
            st.error(f"HTTP error: {e}\n{getattr(e, 'response', None) and e.response.text}")
        except Exception as e:
            st.error(f"Error: {e}")

# ========================= Model 3 =========================
with tabs[2]:
    st.subheader("Kaggle Base model")
    left, right = st.columns([1, 1])
    with left:
        m3_prefill = st.selectbox("Prefill", ["None", "Non-Fire sample", "Fire sample"], index=1, key="m3_prefill")
    with right:
        m3_url = st.text_input("Endpoint", value=M3_URL, key="m3_endpoint")

    if "m3_table" not in st.session_state:
        st.session_state.m3_table = df_from_features_dict(M3_NON_FIRE["data"])

    # Prefill
    src3 = M3_NON_FIRE if m3_prefill == "Non-Fire sample" else M3_FIRE if m3_prefill == "Fire sample" else None
    if src3 is not None:
        st.session_state.m3_table = df_from_features_dict(src3["data"])

    st.markdown("**Features (editable table)**")
    st.caption("Kaggle Base model sensor set.")
    st.session_state.m3_table = st.data_editor(
        st.session_state.m3_table.copy(),
        num_rows="dynamic", use_container_width=True, hide_index=True, key="m3_features_editor"
    )

    m3_run = st.button("Prediction (Kaggle Base)", type="primary", use_container_width=True, key="m3_run_btn")
    st.divider()

    if m3_run:
        try:
            payload3 = {"data": features_dict_from_df(st.session_state.m3_table)}
            with st.spinner("Calling Kaggle Base API..."):
                t0 = time.time()
                resp3 = post_json(m3_url, payload3, timeout_s=25.0)
                elapsed = time.time() - t0

            st.success(f"Response OK ({elapsed:.2f}s)")

            # Banner FIRST (as requested)
            fire_detected = bool(resp3.get("fire_prediction", False))
            if fire_detected:
                colored_box("🔥 Fire detected (Kaggle Base model)", bg="#fde8e8", border="#f5c2c2", color="#7a1111")
                # Send email alert if enabled and not a duplicate
                if enable_email and sender_email and sender_password and recipient_email:
                    if should_send_alert("Kaggle Base Model", resp3):
                        email_sent = send_fire_alert_email("Kaggle Base Model", resp3, email_config)
                        if email_sent:
                            st.success("📧 Fire alert email sent successfully!")
                        else:
                            st.error("📧 Failed to send fire alert email. Check your email configuration.")
                    else:
                        st.info("📧 Fire detected but email alert was already sent for this prediction.")
            else:
                colored_box("✅ Not Fire (Kaggle Base model)", bg="#e6f4ea", border="#b7e1c1", color="#0b5b25")

            # Summary table with colored fire_prediction cell
            st.subheader("Prediction Summary")
            rows = [
                ("fire_prediction", bool(resp3.get("fire_prediction", False))),
                ("score", f"{resp3.get('score'):.12f}" if isinstance(resp3.get("score"), (int, float)) else str(resp3.get("score"))),
                ("latency_ms", f"{resp3.get('latency_ms'):.3f}" if isinstance(resp3.get("latency_ms"), (int, float)) else str(resp3.get("latency_ms"))),
            ]
            html_summary_table(rows)

        except requests.HTTPError as e:
            st.error(f"HTTP error: {e}\n{getattr(e, 'response', None) and e.response.text}")
        except Exception as e:
            st.error(f"Error: {e}")
