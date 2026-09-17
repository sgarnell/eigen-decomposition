# zscore_context.md  
*A complete description of how pre‑ and post‑synaptic z‑scores are defined, computed, and unified into a single functional weight for the Z‑matrix used in spectral analysis and dynamical modeling.*

---

## **1. What a z‑score is and why we use it**

A **z‑score** is a normalized measure of how strongly a neuron participates in a functional relationship relative to the statistical distribution of all such relationships in the dataset.

For any raw value \(X\) (e.g., correlation, co‑activation strength, temporal alignment), the z‑score is:



\[
Z = \frac{X - \mu}{\sigma}
\]



Where:

- \(X\) = the raw measurement  
- \(\mu\) = mean of all measurements in the relevant distribution  
- \(\sigma\) = standard deviation of that distribution  

This normalization ensures:

- values are comparable across neurons  
- extreme relationships stand out  
- weak/noisy relationships collapse toward zero  
- the matrix is scale‑free and suitable for spectral decomposition

---

## **2. Pre‑ and post‑synaptic z‑scores**

For each directed pair of neurons \(i \rightarrow j\), we compute **two** z‑scores:

### **Pre z‑score**


\[
Z_{\text{pre}}(i \rightarrow j)
\]



This measures how strongly neuron **i** (the presynaptic/source neuron) participates in the relationship. It reflects:

- how predictive i’s activity is  
- how aligned i is with the functional pattern  
- how strongly i contributes to the connection

### **Post z‑score**


\[
Z_{\text{post}}(i \rightarrow j)
\]



This measures how strongly neuron **j** (the postsynaptic/target neuron) responds to or reflects the relationship. It captures:

- how sensitive j is to the incoming signal  
- how much j participates in the functional pattern  
- how strongly j receives or expresses the connection

These two values encode **directional functional roles**.  
They are not interchangeable and should not be collapsed naively.

---

## **3. Why we need a unified weight**

Spectral decomposition requires a **single number** per pair of neurons.  
This number must encode:

- **directionality** (gain vs attenuation)  
- **strength** (overall magnitude)  
- **boundedness** (no explosions from tiny pre values)  
- **biological meaning** (role‑specific functional influence)

Simple combinations (mean, product, norm) fail because they erase the **gain/attenuation** concept.

Your original intuition — using a ratio like post/pre — was correct in spirit but unstable in practice.

We therefore use a **bounded, symmetric gain metric** that preserves the ratio logic while remaining numerically stable.

---

## **4. The bounded, symmetric gain metric**

### **Step 1 — Ratio with stabilizer**


\[
R = \frac{\text{post}}{\text{pre} + \epsilon}
\]



- \(R > 1\) → gain  
- \(R < 1\) → attenuation  
- \(\epsilon\) prevents division by zero (default: \(0.1\))

---

### **Step 2 — Log transform**


\[
\log R
\]



This converts multiplicative differences into additive ones and makes gain/attenuation symmetric:

- \(\log(2)\) and \(\log(1/2)\) have equal magnitude, opposite sign.

---

### **Step 3 — Bounded gain metric**


\[
g = \tanh(\alpha \cdot \log R)
\]



- \(g \in (-1, 1)\)  
- \(g > 0\) → gain  
- \(g < 0\) → attenuation  
- \(\alpha\) controls sensitivity (default: \(1\))

This is the **directional functional term**.

---

### **Step 4 — Strength term**


\[
s = \frac{|\text{pre}| + |\text{post}|}{2}
\]



This captures the **magnitude** of the underlying relationship:

- small if both z‑scores are small  
- large if both are large  
- moderate if one is large and the other small

---

### **Step 5 — Final unified weight**


\[
w = s \cdot g
\]



This is the value placed in the **Z‑matrix**.

It encodes:

- **directionality** (gain vs attenuation)  
- **strength** (overall magnitude)  
- **boundedness** (no explosions)  
- **biological meaning** (role‑specific functional influence)

This is the official method for converting pre/post z‑scores into the functional weights used throughout the spectral pipeline.

---

## **5. Examples**

### **Strong gain**
pre = 0.2, post = 0.9 → **w ≈ +0.44**

### **Neutral**
pre = 0.6, post = 0.7 → **w = 0**

### **Strong attenuation**
pre = 0.9, post = 0.2 → **w ≈ –0.51**

### **Weak but high gain**
pre = 0.1, post = 0.5 → **w ≈ +0.22**

### **Very weak**
pre = 0.05, post = 0.08 → **w ≈ –0.036**

---

## **6. Why this metric is used in the Z‑matrix**

This unified weight:

- preserves **functional role asymmetry**  
- captures **amplification vs suppression**  
- avoids instability from small pre values  
- produces **bounded**, interpretable values  
- works cleanly with eigen‑decomposition and SVD  
- respects the biological reality of FB circuits  
- supports downstream dynamical modeling and motif discovery

This metric is the foundation of the functional connectivity matrix used throughout the project.

---
