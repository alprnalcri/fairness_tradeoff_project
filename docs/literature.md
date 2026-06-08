---
title: "Literature Review"
---

# Literature Review

This section summarizes key studies related to algorithmic fairness, fairness-aware machine learning methods, and the fairness–accuracy trade-off.

---

## Algorithmic Fairness

Algorithmic fairness has become a critical research area in machine learning due to concerns about discrimination against protected groups in automated decision-making systems.

Fairness definitions often vary depending on the application context, and achieving all fairness criteria simultaneously is generally not possible in real-world systems.

---

## Fairness Definitions and Theoretical Foundations

Hardt et al. (2016) introduced **Equality of Opportunity**, which requires that individuals who truly qualify for a positive outcome should have equal chances of being correctly predicted, regardless of protected attributes.

Chouldechova (2017) demonstrated that different fairness definitions (e.g., calibration, equalized odds, demographic parity) may be mathematically incompatible, highlighting inherent trade-offs in fairness-aware modeling.

---

## Pre-processing Approaches

Kamiran and Calders (2012) proposed the **Reweighing method**, a preprocessing technique that assigns different weights to training instances in order to reduce bias before model training.

This approach aims to balance representation across protected groups without modifying the underlying learning algorithm.

---

## In-processing Approaches

Agarwal et al. (2018) introduced **Exponentiated Gradient Reduction (ExpGrad)**, an in-processing method that incorporates fairness constraints directly into the optimization process of the model.

Unlike preprocessing techniques, this method enforces fairness during training, often achieving stronger bias mitigation at the cost of computational complexity and potential performance loss.

---

## Fairness–Accuracy Trade-off

Recent surveys (Caton and Haas, 2024) emphasize that fairness interventions must always be evaluated alongside predictive performance.

Most fairness-aware methods introduce a **trade-off between accuracy and fairness**, where improvements in fairness can lead to reductions in predictive performance.

This trade-off is central to the motivation of this study and is empirically analyzed using multiple models and mitigation strategies.