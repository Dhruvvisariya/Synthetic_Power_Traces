# Synthetic Power Trace Generation for Power Analysis Attacks on AES

This project presents a lightweight and practical framework to generate synthetic power traces for AES encryption using real-trace data and interpolation models. It enables accessible side-channel analysis (SCA) without requiring expensive lab-grade hardware. The work builds upon my earlier research on Differential Power Analysis (DPA) and expands it by focusing on realistic trace modeling using Hamming-based leakage metrics.

This project was completed as part of my BTech final semester under the supervision of Prof. Rajneesh Kumar Srivastava at the Department of Electronics and Communication Engineering, University of Allahabad (Jan–May 2025).

This work is a direct continuation of my earlier project:  
🔗 **[Cryptanalysis Using Differential Power Analysis on AES](https://github.com/Dhruvvisariya/DPA-AES-1)**

In that project, I built a basic DPA framework using simulated traces modeled via Hamming Distance (HD) and performed the Difference of Means (DoM) attack to extract the first byte of an AES key. However, synthetic signals lacked realism due to fixed segment designs and uniformity across inputs.

This follow-up project focuses on realistic synthetic power trace generation guided by Points of Interest (POIs) and HD/HW-based leakage modeling. Instead of simulating from scratch, this approach uses the real, unmasked DPAcontest v2 dataset to extract repeating patterns and build accurate synthetic versions using interpolation and mathematical leakage modeling.

The pipeline involves analyzing real AES traces to find key operation timings, detecting POIs using signal derivatives, reconstructing those segments using cubic spline and Chebyshev polynomial interpolation, mapping amplitudes to HD/HW values, and statistically validating the synthetic traces using Pearson correlation and Cohen’s d effect size.

The project is implemented in Python 3.8+ and uses libraries such as `numpy`, `scipy`, `matplotlib`, `seaborn`, and `pandas`.

**How it works:**

* **Real Trace Segmentation:** The DPAcontest v2 AES-128 dataset is used to extract repetitive AES round patterns via visual inspection and statistical analysis.
* **POI Detection:** First and second derivatives of real traces are used to identify inflection points, edge changes, and leakage-prone time samples. These are selected as Points of Interest (POIs).
* **Interpolation & Synthesis:** Power trace segments are interpolated using cubic splines for smooth fitting and Chebyshev polynomials for edge accuracy. These are then reused to generate synthetic traces parameterized by HD and HW models.
* **Mapping Leakage Models:** Using knowledge of plaintext-key XOR operations, leakage values are mapped to HD or HW values and linked to the synthetic POIs to form complete traces.
* **Statistical Validation:** Intra-group similarity (same HD or HW) is measured using Pearson correlation coefficient (mean r ≈ 0.995), and inter-group separability (different HD/HW) is measured using Cohen’s d effect size (|d| > 1.4 across many POIs).

**Results:**

The synthetic traces closely match the statistical behavior of real traces. Pearson correlation (intra-group) showed >0.95 consistency across all major HD/HW classes. Cohen’s d (inter-group) comparisons revealed clear separability, even in noisy trace regions (|d| > 10 in some HW groups). These results strongly validate the approach as useful for studying trace-based leakage without hardware and for testing countermeasures or training machine learning models.

1)FIGURE : Output of Initial Simulation Approach-

![asperbook_trace1](https://github.com/user-attachments/assets/96596640-4a53-4fa0-9fed-a11a8ef42d2f)

2)FIGURE : AES round segregation based on repetitive pattern-

![trace1](https://github.com/user-attachments/assets/b49135f8-fb49-491e-a06a-a78b80e63b11)

3)FIGURE: Segregation of power signature of operations in AES round-

![trace2](https://github.com/user-attachments/assets/46f3313f-4f23-4d17-821b-91c6f53a9a9e)

4)FIGURE:POI DETECTION IN DEDICATED REGION -

![trace3](https://github.com/user-attachments/assets/0ff05651-042e-4420-925b-3b1bb0ef8cac)

5)FIGURE:a comparison plot of (a) a real trace segment, (b) POI locations marked on it, and (c) its interpolated version using Cubic Spline and Chebyshev methods.

![trace4](https://github.com/user-attachments/assets/aa482412-aeb6-47ec-a94e-98e4b2fdf735)

6) Intra-Group Similarity: Pearson Correlation Coefficient-
   
![trace5](https://github.com/user-attachments/assets/96722b78-1659-48d1-8485-b6fd68976114)

7)Inter-Group Differences: Cohen’s d Effect Size

![trace6](https://github.com/user-attachments/assets/ef8bc4f0-91ce-478b-a5cd-88e12ec91a63)


**Applications and Future Work:**

This method reduces dependency on real capture setups and can be extended to:

* Train and test SCA countermeasures like masking, hiding, and noise injection
* Generate synthetic datasets for machine learning-based key recovery
* Model other ciphers such as DES, RSA, or masked implementations
* Add delay/clock artifacts for more realism
* Automate POI selection using mutual information or feature selection techniques

**References:**

* Mangard, S., Oswald, E., & Popp, T. *Power Analysis Attacks: Revealing the Secrets of Smart Cards*, Springer, 2007
* DPAcontest v2 Dataset – [https://www.dpacontest.org/v2/](https://www.dpacontest.org/v2/)
* Harris, C. R., et al. *Array programming with NumPy*, Nature, 2020
* McKinney, W. *Data Structures for Statistical Computing in Python*, PyData, 2010
* Hunter, J. D. *Matplotlib: A 2D Graphics Environment*, 2007
* Seaborn: *Statistical Data Visualization* (JOSS, 2021)

---

**Author:**
**Dhruv Visariya**

BTech in Electronics & Communication Engineering

University of Allahabad

GitHub: [@Dhruvvisariya](https://github.com/Dhruvvisariya)

Email: [dhruvvisariya@gmail.com](mailto:dhruvvisariya@gmail.com)

📌 *This project is intended for academic research, learning, and ethical use only.*
