"""
Script para un sistema de alerta temprana de recesiónutilizando modelos Logit con variables de FRED.
- Construye targets adelantados a 6 y 12 meses a partir de USREC.
- Realiza feature engineering (YoY, MA3).
- Aplica validación temporal expandible con buffer para evitar fuga.
- Evalúa AUC, KS y define umbrales de alerta por FPR.
- Entrena modelos Logit parsimoniosos (Reducidos) y muestra coeficientes y VIF, por horizontes.
Autor: César Roberto Morataya García
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

API_KEY = os.getenv("FRED_API_KEY", "")
HORIZONS_MONTHS = [12, 6]

"""
Se utiliza USREC como target (1 = Recesión, 0 = No recesión). El NBER define una recesión como una disminución significativa 
de la actividad económica que se extiende por toda la economía y que dura más de unos pocos meses. Existen recesiones técnicas
definidas como dos trimestres consecutivos de crecimiento negativo del PIB, y que NBER puede o no reconocer oficialmente como recesión
oficial, ya que consideran múltiples factores.

- Ya que NBER tarda en confirmar recesiones, se busca predecir recesiones con anticipación.
"""
SERIES_MAP = {
    "USREC": "Target_Recession",
    "T10Y3M": "Spread_10Y_3M",
    "DCOILWTICO": "Oil_Price",
    "RSAFS": "Retail_Sales",
    "UMCSENT": "Consumer_Sentiment",
}

#Debido a la poca cantidad de datos de recesión, es necesario usar pocos features para evitar overfitting, para ambos modelos fueron 2 variables, sin contar el intercepto.
FINAL_FEATURES = [
    "Spread_10Y_3M",
    "Sales_YoY",
    "Oil_Price",
    "Oil_YoY",
    "Consumer_Sentiment_YoY"
]

MODEL_FEATURES_BASE = [
    "Spread_10Y_3M",
    "Oil_YoY"
]
MODEL_FEATURES_BASE6 = [
    "Consumer_Sentiment_YoY",
    "Sales_YoY",
]
MODEL_FEATURES_BY_HORIZON = {
    12: MODEL_FEATURES_BASE,
    6: MODEL_FEATURES_BASE6,
}

def require_api_key(api_key: str) -> bool:
    """
    Verifica que la API key esté definida y avisa si falta.
    """
    if not api_key:
        print("FRED_API_KEY no está configurada. Define la variable de entorno antes de ejecutar.")
        return False
    return True


def download_and_engineer_base_data(fred_connection: Fred) -> pd.DataFrame:
    """
    Descarga las series de FRED, consolida frecuencia mensual y calcula transformaciones base.
    """
    print("--- 1) Descargando y consolidando datos ---")
    df = pd.DataFrame()
    for series_id, name in SERIES_MAP.items():
        try:
            data = fred_connection.get_series(series_id)
            temp_df = pd.DataFrame(data, columns=[name])
            df = df.join(temp_df, how="outer") if not df.empty else temp_df
        except Exception as exc:
            print(f"Error al descargar {series_id}: {exc}")

    df = df.resample("M").last().ffill().dropna(how="all")
    df = df[df.index >= "1950-01-01"]
    print(f"Filas base: {len(df)}")

    print("\n--- 2) Feature engineering ---")
    df["Sales_YoY"] = df["Retail_Sales"].pct_change(12) * 100
    df["Oil_YoY"] = df["Oil_Price"].pct_change(12) * 100
    df["Consumer_Sentiment_MA3"] = df["Consumer_Sentiment"].rolling(window=3).mean()
    df["Consumer_Sentiment_YoY"] = df["Consumer_Sentiment_MA3"].pct_change(12) * 100

    return df


def prepare_dataset_for_horizon(df: pd.DataFrame, horizon_months: int) -> tuple[pd.DataFrame, str]:
    """
    Construye dataset final para un horizonte específico desplazando el target y filtrando filas completas.
    """
    target_col = f"Target_Recession_{horizon_months}M"
    df_copy = df.copy()
    df_copy[target_col] = df_copy["Target_Recession"].shift(-horizon_months)
    df_final = df_copy[FINAL_FEATURES + [target_col]].dropna()
    print(f"\n--- Dataset preparado para horizonte {horizon_months}M ---")
    print(f"DataFrame final: {len(df_final)} filas, {df_final.shape[1]} columnas")
    print(f"Recesiones (1s): {int(df_final[target_col].sum())} | Prevalencia: {df_final[target_col].mean():.2%}")
    return df_final, target_col


def ks_from_roc(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray):
    """
    Calcula el valor KS máximo y el umbral asociado a partir de la curva ROC.
    """
    lift = tpr - fpr
    ks_idx = np.argmax(lift)
    return float(lift[ks_idx]), float(thresholds[ks_idx]), int(ks_idx)


def evaluate_split(name: str, y_true: np.ndarray, probs: np.ndarray):
    """
    Evalúa probabilidades con AUC, KS, umbral de KS y matriz de confusión en ese umbral.
    """
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    ks_value, ks_thr, ks_idx = ks_from_roc(fpr, tpr, thresholds)
    preds = (probs >= ks_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    print(f"\n[{name}] AUC={auc:.3f} | KS={ks_value:.3f} | Thr(KS)={ks_thr:.3f}")
    print(f"[{name}] Confusion KS -> TP:{tp} FP:{fp} TN:{tn} FN:{fn}")
    return {"auc": auc, "ks": ks_value, "ks_thr": ks_thr, "fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def plot_roc_curves(curves: list[tuple[str, dict]], horizon_label: str = ""):
    """
    Grafica curvas ROC para una lista de resultados y etiqueta con el horizonte si se indica.
    """
    plt.figure(figsize=(6, 5))
    for label, eval_res in curves:
        plt.plot(eval_res["fpr"], eval_res["tpr"], label=f"{label} AUC {eval_res['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    title_suffix = f" a {horizon_label}" if horizon_label else ""
    plt.title(f"ROC Recesion{title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)
    plt.close()


def plot_upper_corr(df: pd.DataFrame):
    """
    Muestra la matriz de correlación superior del DataFrame con anotaciones numéricas.
    """
    corr = df.corr()
    mask = np.tril(np.ones_like(corr, dtype=bool))
    corr_masked = corr.mask(mask)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_masked, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if not mask[i, j] and not pd.isna(corr_masked.iat[i, j]):
                ax.text(j, i, f"{corr_masked.iat[i, j]:.2f}", ha="center", va="center", color="black", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.title("Matriz de Correlación")
    plt.tight_layout()
    plt.show(block=True)
    plt.close()


def rolling_temporal_validation(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    horizon_months: int,
    val_size_months: int = 48,
    step_months: int = 6,
    min_train_months: int = 120,
):
    """
    Ejecuta validación temporal expandible dejando un buffer igual al horizonte para evitar fuga
    y avanzando las ventanas según los tamaños y pasos configurados.
    """
    results = []
    n_rows = len(df)
    train_end = min_train_months

    while True:
        val_start = train_end + horizon_months
        val_end = val_start + val_size_months
        if val_end > n_rows:
            break

        train_df = df.iloc[:train_end]
        val_df = df.iloc[val_start:val_end]

        X_train = train_df[features]
        y_train = train_df[target_col]
        X_val = val_df[features]
        y_val = val_df[target_col]

        if y_val.nunique() < 2:
            results.append(
                {
                    "train_end": df.index[train_end - 1],
                    "val_start": df.index[val_start],
                    "val_end": df.index[val_end - 1],
                    "train_auc": np.nan,
                    "train_ks": np.nan,
                    "val_auc": np.nan,
                    "val_ks": np.nan,
                    "penalized": False,
                    "skipped_single_class": True,
                }
            )
            train_end += step_months
            continue

        X_train_c = sm.add_constant(X_train)
        X_val_c = sm.add_constant(X_val, has_constant="add")

        glm = sm.GLM(y_train, X_train_c, family=sm.families.Binomial())
        penalized = False
        try:
            glm_res = glm.fit()
        except np.linalg.LinAlgError:
            penalized = True
            glm_res = glm.fit_regularized(alpha=0.1, L1_wt=0.0)

        train_probs = glm_res.predict(X_train_c)
        val_probs = glm_res.predict(X_val_c)
        train_eval = evaluate_split("Train (cv)", y_train.to_numpy(), train_probs)
        val_eval = evaluate_split("Val (cv)", y_val.to_numpy(), val_probs)

        results.append(
            {
                "train_end": df.index[train_end - 1],
                "val_start": df.index[val_start],
                "val_end": df.index[val_end - 1],
                "train_auc": train_eval["auc"],
                "train_ks": train_eval["ks"],
                "val_auc": val_eval["auc"],
                "val_ks": val_eval["ks"],
                "penalized": penalized,
                "skipped_single_class": False,
                "val_probs": val_probs,
                "val_true": y_val.to_numpy(),
                "val_index": val_df.index.to_numpy(),
            }
        )

        train_end += step_months

    return results


def compute_threshold_by_fpr(y_true: np.ndarray, probs: np.ndarray, max_fpr: float = 0.15):
    """
    Calcula el umbral cuya tasa de falsos positivos no supere max_fpr, usando el más conservador si es necesario.
    """
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    mask = fpr <= max_fpr
    if not mask.any():
        idx = np.argmin(fpr)
    else:
        idx = np.where(mask)[0][-1]
    return float(thresholds[idx]), float(fpr[idx]), float(tpr[idx])


def summarize_risk_bands(y_true: np.ndarray, probs: np.ndarray, tau_alerta: float, tau_vigilancia: float):
    """
    Asigna bandas de riesgo a las probabilidades y devuelve conteos y matriz de confusión de alertas.
    """
    bands = pd.Series("bajo", index=range(len(probs)))
    bands[(probs >= tau_vigilancia) & (probs < tau_alerta)] = "vigilancia"
    bands[probs >= tau_alerta] = "alta"
    counts = bands.value_counts()

    alerts = (probs >= tau_alerta).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, alerts, labels=[0, 1]).ravel()
    return {
        "band_counts": counts.to_dict(),
        "alert_confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }

def run_model_for_horizon(df_final: pd.DataFrame, target_col: str, horizon_months: int, model_features: list[str]):
    """
    Ejecuta el pipeline completo para un horizonte: correlación, validación walk-forward,
    calibración de umbrales y ajuste final del modelo reducido.
    """
    print(f"\n================ Horizonte {horizon_months} meses ================")
    print(f"Features usados (modelo reducido): {model_features}")
    print("Mostrando matriz de correlacion (todas las variables + target)...")
    plot_upper_corr(df_final[FINAL_FEATURES + [target_col]])

    wf_min_train_months = 144
    wf_val_months = 36
    wf_step_months = 6
    val_eval_agg = None
    cv_results = rolling_temporal_validation(
        df_final,
        model_features,
        target_col,
        horizon_months=horizon_months,
        val_size_months=wf_val_months,
        step_months=wf_step_months,
        min_train_months=wf_min_train_months,
    )
    if cv_results:
        avg_train_auc = np.nanmean([r["train_auc"] for r in cv_results])
        avg_val_auc = np.nanmean([r["val_auc"] for r in cv_results])
        avg_train_ks = np.nanmean([r["train_ks"] for r in cv_results])
        avg_val_ks = np.nanmean([r["val_ks"] for r in cv_results])
        val_probs_list = [r["val_probs"] for r in cv_results if r.get("val_probs") is not None and not np.isnan(r["val_auc"])]
        val_true_list = [r["val_true"] for r in cv_results if r.get("val_true") is not None and not np.isnan(r["val_auc"])]
        val_index_list = [r["val_index"] for r in cv_results if r.get("val_index") is not None and not np.isnan(r["val_auc"])]
        val_probs_concat = np.concatenate(val_probs_list) if val_probs_list else None
        val_true_concat = np.concatenate(val_true_list) if val_true_list else None
        val_index_concat = np.concatenate(val_index_list) if val_index_list else None
        print(
            f"Walk-forward promedio ({horizon_months}M) -> AUC Train/Val: {avg_train_auc:.3f}/{avg_val_auc:.3f} | "
            f"KS Train/Val: {avg_train_ks:.3f}/{avg_val_ks:.3f}"
        )
        if val_probs_concat is not None and val_true_concat is not None and val_index_concat is not None:
            oos_df = pd.DataFrame(
                {"prob": val_probs_concat, "target": val_true_concat},
                index=pd.DatetimeIndex(val_index_concat),
            ).sort_index()
            oos_df = oos_df[~oos_df.index.duplicated(keep="last")]
            val_probs_oos = oos_df["prob"].to_numpy()
            val_true_oos = oos_df["target"].to_numpy()
            val_index_oos = oos_df.index.to_numpy()

            val_eval_agg = evaluate_split(f"Val OOS agg {horizon_months}M", val_true_oos, val_probs_oos)
            max_fpr_target = 0.05 if horizon_months == 12 else 0.15
            tau_alerta, fpr_alerta, tpr_alerta = compute_threshold_by_fpr(val_true_oos, val_probs_oos, max_fpr=max_fpr_target)
            non_event_probs = val_probs_oos[val_true_oos == 0]
            tau_vigilancia = float(np.percentile(non_event_probs, 70)) if len(non_event_probs) else 0.2
            bands_summary = summarize_risk_bands(val_true_oos, val_probs_oos, tau_alerta, tau_vigilancia)
            alerts_mask = val_probs_oos >= tau_alerta
            alert_dates = pd.DatetimeIndex(val_index_oos)[alerts_mask]
            alert_targets = val_true_oos[alerts_mask]
            alert_probs = val_probs_oos[alerts_mask]
            print(
                f"\nUmbrales out-of-sample ({horizon_months}M) -> tau_alerta={tau_alerta:.3f} (FPR~{fpr_alerta:.3f}, TPR~{tpr_alerta:.3f}) | "
                f"tau_vigilancia={tau_vigilancia:.3f} (percentil 70 no-recesion)"
            )
            print(f"Bandas (val oos): {bands_summary['band_counts']}")
            ac = bands_summary["alert_confusion"]
            print(f"Confusion alta alerta (val oos): TP:{ac['tp']} FP:{ac['fp']} TN:{ac['tn']} FN:{ac['fn']}")
            if len(alert_dates):
                print("\nFechas con alta alerta (val OOS):")
                for d, p, t in zip(alert_dates, alert_probs, alert_targets):
                    print(f"input={d.date()} | prob={p:.3f} | target={int(t)} (objetivo t+horizonte)")
            else:
                print("\nNo hubo altas alertas en validación OOS con este umbral.")
        else:
            print("No hay predicciones de validación con ambas clases; no se puede calibrar umbrales OOS.")
    else:
        print(f"No se pudieron generar folds suficientes para CV temporal (horizonte {horizon_months}M).")

    print(f"\n--- Modelo logit reducido entrenado en todo el histA3rico ({horizon_months}M) ---")
    X_full = df_final[model_features]
    y_full = df_final[target_col]
    X_full_c = sm.add_constant(X_full)
    glm_full = sm.GLM(y_full, X_full_c, family=sm.families.Binomial())
    penalized_full = False
    try:
        glm_full_res = glm_full.fit()
    except np.linalg.LinAlgError:
        penalized_full = True
        glm_full_res = glm_full.fit_regularized(alpha=0.1, L1_wt=0.0)

    full_eval = evaluate_split(f"Full {horizon_months}M", y_full.to_numpy(), glm_full_res.predict(X_full_c))
    roc_curves = [("Full", full_eval)]
    if val_eval_agg is not None:
        roc_curves.insert(0, ("Val OOS", val_eval_agg))
    plot_roc_curves(roc_curves, horizon_label=f"{horizon_months}M")

    if penalized_full:
        coef_full_df = pd.DataFrame({"coef": glm_full_res.params})
        coef_full_df["pvalue"] = np.nan
        print(f"\nCoeficientes logit reducido (penalizado, pvalues no disponibles) {horizon_months}M:")
        print(coef_full_df)
    else:
        coef_full_df = pd.DataFrame({"coef": glm_full_res.params, "pvalue": glm_full_res.pvalues})
        print(f"\nCoeficientes logit reducido (con constante, full sample) {horizon_months}M:")
        print(coef_full_df)

    vif_full_df = pd.DataFrame(
        {
            "feature": model_features,
            "vif": [variance_inflation_factor(X_full.values, i) for i in range(X_full.shape[1])],
        }
    )
    print(f"\nVIF modelo reducido (full sample) {horizon_months}M:")
    print(vif_full_df)


if __name__ == "__main__":
    if not require_api_key(API_KEY):
        sys.exit(1)
    fred = Fred(api_key=API_KEY)
    base_df = download_and_engineer_base_data(fred)

    for horizon_months in HORIZONS_MONTHS:
        df_final, target_col = prepare_dataset_for_horizon(base_df, horizon_months)
        model_features = MODEL_FEATURES_BY_HORIZON.get(horizon_months, MODEL_FEATURES_BASE)
        run_model_for_horizon(df_final, target_col, horizon_months, model_features)

