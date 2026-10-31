import json
import os
from pathlib import Path

import numpy as np
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
API = os.environ.get("API_URL", "http://127.0.0.1:8000")
REPORTS = ROOT / "reports"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

st.set_page_config(page_title="TrustScore AI — робоче місце кредитного аналітика", layout="wide")
st.title("TrustScore AI — підтримка рішення за кредитною заявкою")

try:
    health = requests.get(f"{API}/health", timeout=5).json()
except Exception as exc:
    st.error(f"Сервіс недоступний: {exc}. Запустіть `uvicorn src.app.main:app`.")
    st.stop()
if health.get("status") != "ok":
    st.error(f"Сервіс не готовий (503): {health}")
    st.stop()
st.caption(f"Модель: {health['model_version']} · поріг P(bad) ≥ {health['threshold']:.3f} → bad · OOD: {health['ood']}")

schema = requests.get(f"{API}/openapi.json", timeout=5).json()["components"]["schemas"]
tab_local, tab_global, tab_trust, tab_track = st.tabs(["Інференс і локальні пояснення", "Глобальні пояснення", "Довіра до пояснень", "Карта даних (трек 4C)"])


def _levels(name):
    ref = schema["PredictionRequest"]["properties"][name]
    for key in ("enum", "anyOf"):
        if key in ref:
            vals = ref["enum"] if key == "enum" else next((o["enum"] for o in ref["anyOf"] if "enum" in o), [])
            return vals
    return []


with tab_local:
    st.subheader("Заявка (без захищених ознак)")
    with st.form("application"):
        c1, c2, c3 = st.columns(3)
        duration = c1.number_input("duration, міс.", 1, 120, 24)
        credit_amount = c1.number_input("credit_amount, DM", 100, 50000, 3000, step=100)
        age = c1.number_input("age", 18, 100, 35)
        installment_rate = c2.number_input("installment_rate (1–4)", 1, 4, 3)
        residence_since = c2.number_input("residence_since (1–4)", 1, 4, 2)
        existing_credits = c2.number_input("existing_credits", 1, 10, 1)
        cat_values = {}
        cols = [c1, c2, c3]
        for i, name in enumerate(["checking_account", "credit_history", "purpose", "savings_account", "other_debtors", "property",
                                  "other_installment_plans", "housing", "telephone", "employment_since", "job"]):
            cat_values[name] = cols[(i + 1) % 3].selectbox(name, _levels(name))
        submitted = st.form_submit_button("Оцінити")
    if submitted:
        payload = {"duration": int(duration), "credit_amount": float(credit_amount), "age": int(age), "installment_rate": int(installment_rate),
                   "residence_since": int(residence_since), "existing_credits": int(existing_credits), **cat_values}
        r = requests.post(f"{API}/v1/explain", json={"request": payload}, timeout=60)
        if r.status_code != 200:
            st.error(f"{r.status_code}: {r.json()}")
        else:
            body = r.json(); pred = body["prediction"]; meta = pred["meta"]
            st.session_state["last_request_id"] = pred["request_id"]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("P(bad)", f"{pred['score']:.3f}")
            k2.metric("Рекомендація", "ВІДМОВА / ручний розгляд" if pred["label"] == 1 else "СХВАЛИТИ")
            k3.metric("Невизначеність (ентропія)", f"{pred['uncertainty']:.3f}")
            k4.metric("Сегмент", meta["segment"]["name"][:40] if meta.get("segment") else "— (OOD)")
            if meta["fallback_applied"]:
                st.warning("Нетиповий запит: відповідь дало Non-ML правило ЛР №1 (checking <0 DM або duration > 24 і savings <100 DM). "
                           "Моделі тут не довіряють; SHAP і LIME нижче описують модель лише довідково — її прогноз користувачу не віддано.")
            st.caption(f"request_id = {pred['request_id']} · log-густина {meta['ood_log_density']:.2f}")
            left, right = st.columns(2)
            with left:
                st.markdown("**SHAP waterfall** (внески в P(bad), одиниці ймовірності — не відсотки й не причини)")
                import matplotlib.pyplot as plt
                import shap
                s = body["shap"]
                exp = shap.Explanation(values=np.array(s["values"]), base_values=s["base_value"],
                                       data=np.array([v if isinstance(v, (int, float)) else np.nan for v in s["data"]]), feature_names=s["feature_names"])
                shap.plots.waterfall(exp, show=False, max_display=10)
                st.pyplot(plt.gcf()); plt.close("all")
            with right:
                st.markdown("**LIME** (локальна лінійна апроксимація)")
                st.table([{"ознака": k, "вага": round(v, 4)} for k, v in body["lime"][:10]])
                ag = body["agreement"]
                st.metric("Збіг топ-5 SHAP/LIME", f"{ag['top5_overlap']:.0%}", help=f"Спірмен за |внесками|: {ag['spearman']}")
            with st.expander("Надіслати фактичний результат (фідбек)"):
                y_true = st.radio("Фактично", [0, 1], format_func=lambda v: "повернув (good)" if v == 0 else "дефолт (bad)", horizontal=True)
                if st.button("Надіслати /v1/feedback"):
                    fr = requests.post(f"{API}/v1/feedback", json={"request_id": pred["request_id"], "y_true": int(y_true)}, timeout=10)
                    st.write(fr.json())

with tab_global:
    xs = REPORTS / "xai" / "xai_summary.json"
    if xs.exists():
        summ = json.loads(xs.read_text(encoding="utf-8"))
        st.caption(f"run_id {summ['run_id']} · версія {summ['model_version']}")
        st.image(str(REPORTS / "xai" / "shap_summary.png"), caption="SHAP summary (beeswarm) на test")
        st.image(str(REPORTS / "xai" / "pdp_ice_or_ale.png"), caption="PDP + ICE або ALE для двох найважливіших ознак")
        st.json(summ["pdp"])
        st.markdown("Вибір PDP чи ALE: ALE — якщо |ρ Спірмена| з будь-якою іншою ознакою > 0.7 (PDP тоді будує нереалістичні комбінації).")
    else:
        st.info("Запустіть `python -m src.xai.offline_report`.")

with tab_trust:
    xs = REPORTS / "xai" / "xai_summary.json"
    if xs.exists():
        summ = json.loads(xs.read_text(encoding="utf-8"))
        c1, c2 = st.columns(2)
        c1.metric("Fidelity сурогатного дерева (R², test)", f"{summ['surrogate']['fidelity_r2_test']:.3f}")
        c2.metric("Стабільність SHAP (Спірмен топ-5, 30×20 збурень)", f"{summ['stability']['mean_spearman_top5']:.3f} ± {summ['stability']['sd']:.3f}")
        st.markdown("Стабільність і збіг SHAP з LIME — **діагностика надійності, а не доказ правильності** пояснення.")
        st.code((REPORTS / "xai" / "surrogate_tree.txt").read_text(encoding="utf-8")[:3000])

with tab_track:
    sj = REPORTS / "segments.json"
    if sj.exists():
        seg = json.loads(sj.read_text(encoding="utf-8"))
        import matplotlib.pyplot as plt
        pts = np.load(REPORTS / "xai" / "train_map.npy")
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for k in np.unique(pts[:, 2]).astype(int):
            mm = pts[:, 2] == k
            name = next((s["name"] for s in seg["segments"] if s["id"] == k), str(k))
            ax.scatter(pts[mm, 0], pts[mm, 1], s=8, color=PALETTE[k % 8], alpha=0.7, edgecolors="none", label=name[:40])
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(frameon=False, fontsize=7)
        if "last_request_id" in st.session_state:
            st.caption("Поточний запит позначити на карті можна лише за координатами проєкції; сервіс їх не повертає у відповіді — "
                       "показано log-густину й сегмент у вкладці інференсу.")
        st.pyplot(fig); plt.close(fig)
        rows = [{"сегмент": s["id"], "назва": s["name"], "n test": s["n_test"], "частка bad": s["bad_share"], "ROC-AUC": s.get("roc_auc"),
                 "F1": s.get("f1"), "CI Δ AUC": s.get("auc_diff_ci"), "fallback": s.get("fallback_share"), "обмежена довіра": s.get("limited_trust")} for s in seg["segments"]]
        st.dataframe(rows)
        st.markdown("Топ-ознаки за середнім |SHAP| у сегменті:")
        st.json({s["name"][:40]: s["mean_abs_shap"] for s in seg["segments"]})
    else:
        st.info("Запустіть `python -m src.xai.offline_report`.")
