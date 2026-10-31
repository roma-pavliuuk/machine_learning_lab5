import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def slide(pdf, title, lines, fig=None, foot=""):
    if fig is None:
        fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor("#fcfcfb")
    fig.text(0.05, 0.92, title, fontsize=24, weight="bold", color="#0b0b0b", va="top")
    y = 0.82
    for line in lines:
        size = 15 if not line.startswith("  ") else 13
        fig.text(0.05, y, line, fontsize=size - (1 if fig.axes else 0), color="#0b0b0b" if not line.startswith("  ") else "#52514e", va="top", wrap=True)
        y -= 0.065 if not line.startswith("  ") else 0.05
    if foot:
        fig.text(0.05, 0.03, foot, fontsize=9, color="#8a8984")
    pdf.savefig(fig); plt.close(fig)


def main() -> None:
    meta, parity = load(ROOT / "models/model_meta.json"), load(ROOT / "models/parity_report.json")
    ood, roi, fair = load(R / "ood_evaluation.json"), load(R / "roi.json"), load(R / "fairness_test.json")
    xai, seg, lat = load(R / "xai/xai_summary.json"), load(R / "segments.json"), load(R / "latency.json")
    foot = f"run_id {meta['run_id']} · {meta.get('release')} · test n=300 · валюта DM"
    with PdfPages(R / "defense.pdf") as pdf:
        slide(pdf, "1. Контекст — TrustScore AI для кредитного аналітика", [
            "Користувач: кредитний аналітик банку. Рішення: схвалити заявку, відхилити або передати на ручний розгляд.",
            "Вхід: анкетні й фінансові дані позичальника — без захищених ознак (стать, сімейний стан, іноземець).",
            "Вихід: P(дефолту), рекомендація, невизначеність, фактори для обґрунтування; для нетипових заявок — відповідь правила.",
            "Дані: Statlog German Credit, 1 000 заявок, 30 % дефолтів; тренування 700 / тест 300 (ЛР №1–3, DVC).",
            "Мета з ML Canvas (ЛР №1): менше втрат від неповернених кредитів без дискримінації за статтю.",
            "Ціна помилок (ЛР №3): FN — видали неплатникові, 1 500 DM; FP — відмовили надійному, 150 DM.",
        ], foot=foot)
        rule = ood["rule_all_test"]; model = ood["model_all_test"]; system = ood["system_model_plus_rule"]
        slide(pdf, "2. Конфлікт — межі Non-ML правила", [
            f"Правило ЛР №1 («рахунок < 0 DM або строк > 24 міс. при заощадженнях < 100 DM»): {rule['cost_per_applicant']:.0f} DM втрат на заявку на тесті.",
            f"Воно ловить лише {roi['confusion']['rule']['TP']} з 90 дефолтів ({roi['confusion']['rule']['FN']} пропущено × 1 500 DM) — за 10:1 ціною помилок пропуски домінують.",
            "ЛР №2–3 показали: розширення ознак чужими даними (скарги CFPB, платежі Home Credit) не допомагає,",
            "  а витік даних дає хибні +0.10 AUC (p = 0.005). Потрібна герметична модель на анкеті з контролем витоку й OOD.",
            f"Модель ЛР №3 (RF, class_weight 1:10, поріг {meta['threshold']:.2f}): {model['cost_per_applicant']:.1f} DM/заявку на тесті, ROC-AUC {model['roc_auc']:.3f}.",
            "Але: вона схвалює лише ~16 % заявок — бізнес-оптимум за цієї матриці втрат дуже обережний.",
        ], foot=foot)
        slide(pdf, "3. Вирішення — архітектура PoC", [
            "Модель → ONNX (RandomForest + відбір у графі); препроцесор ЛР №1 (імпутер, one-hot, відношення) — поза графом, skops.",
            "MLflow Registry: v1 (модель) → v2 (+ OOD-детектор, сегменти), аліас @champion; відкат = перепризначення аліасу.",
            "FastAPI: контракт Pydantic v2 (межі з домену, strict, захищені ознаки → 422); інференс у threadpool; SQLite-журнал і /v1/feedback.",
            f"OOD-fallback: GMM-густина у просторі моделі ({ood['detectors']['projection']['n_components']} компонент PCA, {ood['detectors']['projection']['explained_variance']:.0%} дисперсії); нижче 1-го перцентиля → відповідь правила.",
            f"Трек 4C «Карта даних»: {len(seg['segments'])} сегментів заявників із метриками й топ-ознаками; зони обмеженої довіри в Model Card.",
            "XAI: SHAP (TreeExplainer, interventional) + LIME на сервері; дашборд Streamlit лише через HTTP.",
        ], foot=foot)
        fig = plt.figure(figsize=(13.33, 7.5))
        ax = fig.add_axes([0.16, 0.12, 0.80, 0.24])
        names = ["правило ЛР №1", "модель", "система\n(модель + правило)"]
        vals = [rule["cost_per_applicant"], model["cost_per_applicant"], system["cost_per_applicant"]]
        ax.barh(names, vals, color=[GRAY, BLUE, ORANGE], height=0.6); ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v + 2, i, f"{v:.0f}", va="center")
        ax.set_xlabel("втрати, DM на заявку (test)"); ax.set_title("Втрати на тесті", loc="left", fontsize=11); ax.grid(axis="y", visible=False)
        f_sys = fair["system"]
        slide(pdf, "4. Перевірка — інтервали, паритет, тести, справедливість", [
            f"Вкладена CV ЛР №3 (5 фолдів): ROC-AUC 0.745 (0.69–0.80), втрати 99.5 DM (92–107).",
            f"Тест 300: ROC-AUC {model['roc_auc']:.3f}, F1 {model['f1']:.2f}; паритет ONNX max|Δp| = {parity['max_abs_diff']:.1e}, {parity['n_label_changed']} змінених міток.",
            f"Автотести: паритет + 8 сценаріїв API + 2 треку (pytest); затримка /v1/predict p95 = {lat['endpoints']['/v1/predict']['p95_ms']:.0f} мс.",
            f"OOD: fallback на тесті {ood['fallback_share_test']:.1%}, штучних OOD виявлено {ood['detectors'][ood['selected_detector']]['synthetic_ood_detected_share']:.0%}.",
            f"Справедливість (стать, система): DPR {f_sys['DPR']:.2f} [{f_sys['DPR_ci'][0]:.2f}; {f_sys['DPR_ci'][1]:.2f}], EOR {f_sys['EOR']:.2f} — правило 4/5 не підтверджено,",
            "  інтервал містить 1: мало схвалень (13 % жінок, 18 % чоловіків). Конфлікт бізнес-оптимуму і 4/5 зафіксовано.",
            f"Пояснення: fidelity сурогату R² = {xai['surrogate']['fidelity_r2_test']:.2f}; стабільність SHAP (Спірмен топ-5) {xai['stability']['mean_spearman_top5']:.2f} — діагностика, не доказ.",
        ], fig=fig, foot=foot)
        fig = plt.figure(figsize=(13.33, 7.5))
        ax = fig.add_axes([0.16, 0.12, 0.80, 0.24])
        sc = roi["scenarios"]; keys = ["песимістичний", "центральний", "оптимістичний"]
        eff = [sc[k]["effect_per_applicant"] for k in keys]
        ax.barh(keys, eff, color=[GRAY, BLUE, AQUA], height=0.6); ax.invert_yaxis()
        ci = sc["центральний"]["effect_ci95"]
        ax.errorbar([eff[1]], [1], xerr=[[eff[1] - ci[0] / 300], [ci[1] / 300 - eff[1]]], fmt="none", ecolor="#0b0b0b", capsize=6)
        for i, v in enumerate(eff):
            ax.text((ci[1] / 300 if i == 1 else max(v, 0)) + 2, i, f"{v:+.0f}", va="center")
        ax.axvline(0, color="#0b0b0b", lw=0.8); ax.set_xlabel("ефект, DM на заявку vs правило ЛР №1 (95 % CI — центральний)"); ax.grid(axis="y", visible=False)
        ax.set_title("Чесний ROI: три сценарії вартостей", loc="left", fontsize=11)
        slide(pdf, "5. Вплив — чесний ROI, OOD, трековий модуль, регуляторика", [
            f"Ефект = втрати(правило) − втрати(система), той самий тест: центральний +{eff[1]:.0f} DM/заявку (95 % CI {ci[0]/300:+.0f}…{ci[1]/300:+.0f}),",
            f"  песимістичний +{eff[0]:.1f} (C01×1.5, C10×0.6), оптимістичний +{eff[2]:.0f} (C01×0.7, C10×1.3). Припущення — з матриці ЛР №3.",
            f"Ціна: схвалення падає з 64 % (правило) до ~16 % — бінарне рішення за 10:1 неадекватне; наступний крок — ціноутворення за ризиком.",
            f"OOD: {ood['n_fallback']} з 300 нетипових заявок отримали відповідь правила; на них якість детектора не оцінено — міток OOD немає.",
            "Трек 4C: аналітик бачить, чи типовий позичальник і наскільки точна модель для його сегмента; зони обмеженої довіри — у Model Card.",
            "EU AI Act: кредитний скоринг — Annex III (високий ризик), обов'язки з 2 грудня 2027 р.; остаточне рішення — за людиною.",
        ], fig=fig, foot=foot)
    print("reports/defense.pdf — 5 слайдів")


if __name__ == "__main__":
    main()
