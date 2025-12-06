# Modelo Logit de alerta temprana para recesiones en Estados Unidos
## Resumen ejecutivo
El objetivo de este proyecto es crear un sistema de alerta temprana para recesiones en Estados Unidos en dos temporalidades, doce meses y seis meses. La variable dependiente es binaria ($Y=1$ si hay recesión, $Y=0$ si no la hay), definida según la demarcación oficial de recesiones del NBER (National Bureau of Economic Research). Los modelos busca predecir con datos de hoy, si va a haber recesión en 6 y 12 meses
## Modelo logit a 12 meses
Para el modelo de doce meses se utilizaron como variables explicativas el precio del petróleo y la *Spread 10Y-3M*, la primera en cambios interanuales.

#### Explicación de las variables
**Precio del petróleo:** Precio spot del petróleo crudo West Texas Intermediate, medido en dólares por barril.

**Spread 10Y - 3M:** Es una diferencia entre dos tasas de interés de la curva de rendimiento de los bonos del Tesoro de EE. UU

Al utilizar las variables en cambios interanuales, conseguimos estacionariedad y evitamos regresiones espurias.

### Interpretación económica de las variables
Estas variables tienen alta relación con el estado de la economía en el largo plazo y pueden alertarnos de posibles recesiones en un futuro.

En el caso de la variable del precio del petróleo, su interpretación económica como una variable que puede explicar una posible recesión es la siguiente: al aumentar su precio y generar un shock, produce tanto un aumento de costos como una reducción de la oferta, generando presiones inflacionarias. Para contrarrestar estas presiones, la Fed utiliza políticas monetarias restrictivas, lo que puede enfriar la economía y llevar a una recesión. Casos como este se han visto en recesiones como la de 1973, 1979 y 1990, entre otras.

En cuanto a por qué el *Spread 10Y–3M* es un indicador adelantado de recesión, es conocido por su robustez empírica. Cuando el rendimiento del bono a 3 meses, fuertemente influenciado por la política monetaria, supera al de los bonos de largo plazo, esto refleja que el mercado espera un deterioro en el crecimiento y la inflación a futuro. El rendimiento de corto plazo sube cuando la Fed incrementa la federal funds rate, mientras que el rendimiento a 10 años puede caer si aumentan las expectativas de bajo crecimiento, lo que explica la inversión de la curva y el spread negativo. De modo que el Spread 10Y–3M capta simultáneamente las condiciones monetarias contractivas y las expectativas pesimistas del mercado. Además, cuando existen presiones inflacionarias, la Fed responde endureciendo la política monetaria, elevando las tasas de corto plazo y generando condiciones financieras más estrictas, lo cual puede contribuir a desacelerar la actividad económica y aumentar la probabilidad de una recesión.

En conclusión, tanto el precio del petróleo crudo como el *Spread 10Y - 3M* tienen relación y pueden llegar a pronosticar una recesión desde diferentes aristas. Estas variables fueron seleccionadas al tener muy buena correlación con las recesiones y no presentar multicolinealidad que afecte al modelo.

## Modelo logit a 6 meses
En un modelo donde el horizonte es más corto, las variables que pueden explicar una recesión son distintas. Para el caso de un horizonte de 6 meses, las que mejor comportamiento tuvieron fueron el sentimiento del consumidor y las ventas de *retail*.

En ambas variables, igual representadas por cambios interanuales, se consigue estacionariedad y se evitan regresiones espurias.

#### Explicación de las variables
**Sentimiento del consumidor:** Índice realizado por la Universidad de Michigan, que resume la situación financiera actual, sus expectativas de ingreso y empleo, y si consideran buen momento para comprar bienes duraderos, entre otros.

**Ventas de *retail*:** Indicador del gasto de los hogares en comercios minoristas.

### Interpretación económica de las variables
El sentimiento del consumidor es una variable sumamente valiosa para saber el estado actual de los hogares y sus perspectivas sobre el futuro. Es importante porque los hogares tienden a cambiar su patrón de consumo y ahorro en base a sus perspectivas. Cuando tienen incertidumbre sobre el futuro, tienden a reducir su consumo y aumentar su ahorro, lo que acentúa y acelera la recesión.

El indicador del gasto de los hogares en comercios minoristas refleja ese comportamiento ya efectivo en el patrón de consumo. Esta variable afecta significativamente a los ciclos, ya que el consumo privado representa cerca del 70% del PIB de Estados Unidos; una reducción del consumo tiene un impacto directo en el PIB del país.

Aunque pareciera que ambas variables están muy correlacionadas y que podrían traer problemas de multicolinealidad, los valores VIF se encuentran en 1, y ambas variables son significativas, por lo que se descarta que generen problemas de multicolinealidad al modelo.

## Resultados de los modelos
| **Logit 12 meses** | **Logit 6 meses** |
|------------------|------------------|
| AUC (Full) = 0.0.837 | AUC (Full) = 0.855 |
| AUC (Val OOS) = 0.530 | AUC (Val OOS) = 0.794 |

El uso de **walk-forward** evita “ver el futuro” y hace la evaluación más realista que un simple train/test aleatorio.
- El **OOS de AUC** junta todas las predicciones de validación en una sola serie continua y mide AUC/KS ahí.
- Eso suele ser más riguroso porque expone cómo se ordenan eventos y no‑eventos entre ventanas distintas.
- Aunque el VAL OSS parece bajo, sucede porque el modelo detecta falsos positivos en el periodo de 2022, donde hubo una recesión técnica, la cual no fue marcada como recesión por la NBER.
<p align="center">
  <img src="img/ROC Modelo 12 meses.png" width="400">
  <img src="img/ROC Modelo 6 meses.png" width="400">
</p>

## Conclusión
El sistema se construye bajo un enfoque multihorizonte, donde cada modelo captura dimensiones distintas del ciclo económico.

El modelo de 12 meses, aunque su capacidad predictiva fuera de muestra es limitada (AUC≈0.53), sirve como indicador temprano: refleja tensiones macroeconómicas de largo plazo y funciona como una señal inicial de monitoreo.

El modelo de 6 meses, con un desempeño OOS robusto (AUC≈0.79), aporta la señal operativa principal y permite distinguir episodios de riesgo con mayor precisión.

En conjunto, ambos modelos no se interpretan como “predicciones independientes”, sino como módulos complementarios dentro de un mismo marco analítico, donde diferentes variables y horizontes aportan información desde perspectivas distintas

## Caso de Estudio
#### Recesión técnica 2022
Aunque la NBER no clasificó este periodo como recesión oficial, principalmente por la fortaleza del mercado laboral, la economía estadounidense experimentó una contracción, con dos trimestres consecutivos de caída del PIB real.

En este episodio, el modelo con horizonte de 12 meses mostró un incremento de las probabilidades de recesión desde 2021, enviando una alerta alta de recesión. Su desempeño fuera de muestra es moderado (AUC≈0.53), por lo que se interpreta como una señal temprana pero de baja potencia estadística.

Posteriormente, el modelo con horizonte de 6 meses, que presenta un desempeño out-of-sample sólido (AUC≈0.79), emitió una señal de vigilancia más clara.

De esta forma, el sistema multihorizonte permite analizar el riesgo de recesión en distintos plazos: el modelo de 12 meses funciona como indicador exploratorio de largo plazo, mientras que el de 6 meses aporta la señal operativa principal.
