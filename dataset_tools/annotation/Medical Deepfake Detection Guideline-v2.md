# Medical Deepfake Detection Guideline

## General Principles (Universal Criteria)

This section applies to all medical imaging modalities. The core philosophy is that AI generation often fails to replicate the **"Biological Interconnectivity"** and **"Pathological Chronology"** of real diseases.

### Biological Plausibility & Secondary Effects (The "Ripple Effect")
*   **Mass Effect Absence:** Real lesions (especially tumors/masses) are physical objects that displace surrounding tissue.
    *   Reject if a space-occupying lesion exists without corresponding compression, displacement, or deformation of adjacent structures (e.g., midline shift, sulcal effacement).
*   **Lack of Host Reaction:** The body reacts to pathology (edema, inflammation, scarring).
    *   Reject if an aggressive lesion (e.g., malignancy, acute infarction) appears "isolated" with a sharp boundary and no surrounding edema (perilesional signal change) or infiltration.
*   **Chronological Inconsistency:** Diseases follow a timeline.
    *   Reject if late-stage features (e.g., neovascularization in DR) appear without precursor signs (e.g., ischemia/microaneurysms), or if "B" exists without "A".

### Image Physics & Texture Consistency
*   **The "Sticker" Artifact:** AI-generated lesions often look like stickers pasted onto a background.
    *   Inspect the lesion-background interface. Reject if the boundary is unnaturally sharp or lacks the gradual transition zone seen in biological tissues.
*   **Noise Distribution Analysis:** Medical images have inherent noise (grain) from the acquisition process.
    *   Reject if the noise pattern (granularity) within the lesion or a specific region is significantly smoother, more uniform, or different in texture compared to the surrounding unaffected tissue (indicates inpainting/smoothing).
*   **Inpainting Artifacts (Removal):**
    *   In areas where a lesion might have been removed, look for "smudging," blurring, or repetitive texture patterns (cloning artifacts) that disrupt the natural stochastic texture of biological tissue.


---

## Modality-Specific Criteria

### Brain MRI (Magnetic Resonance Imaging)

**Anatomical & Structural Logic**
*   **Gyral/Sulcal Morphology:**
    *   Are the sulci adjacent to a mass narrowed or effaced?
    *   Is the ventricular system symmetrical? (Unless displaced by pathology).
    *   Is the Grey/White matter differentiation preserved and anatomically correct in the generated region?
*   **Midline Structure:**
    *   Does a large unilateral mass cause a contralateral midline shift?

**Signal Intensity & Pathological Features**
*   **Edema Signal:**
    *   Is peritumoral edema present? Does it follow the correct signal intensity for fluid (e.g., Hyperintense on T2/FLAIR, Hypointense on T1)? AI often generates edema as "black" or wrong intensity.
*   **Lesion Specificity:**
    *   *Meningioma:* Must exhibit dural attachment ("Dural Tail Sign").
    *   *Glioma:* Must show infiltrative margins, potential necrosis (central hypointensity), and cross-midline growth if advanced.
    *   *Infarction:* Must follow vascular territories. Signal evolution must match the stage (e.g., Acute: DWI High / ADC Low).
*   **Multi-Sequence Consistency:**
    *   Does the lesion appearance logically translate across sequences (T1, T2, FLAIR, DWI)? (e.g., fluid is bright on T2, dark on T1).

---

### Fundus Photography (Retinal Imaging)

**Vascular Network Logic**
*   **Vessel Morphology:**
    *   Do vessels taper gradually from the optic disc to the periphery?
    *   Are there unnatural discontinuities, abrupt endings, or "floating" vessel segments?
    *   Is the Artery/Vein (A/V) ratio and crossing pattern (A/V nipping) physiologically consistent?
*   **Optic Disc & Macula:**
    *   Is the Macula located approximately 2 Disc Diameters (PD) temporal to the Optic Disc?
    *   Is the Optic Disc morphology (cup/disc ratio) natural?

**Lesion Characteristics & Distribution**
*   **Diabetic Retinopathy (DR):**
    *   Microaneurysms should be distinct, red dots <125µm.
    *   Hard exudates (waxy, sharp edges) vs. Soft exudates (cotton-wool spots, fuzzy edges).
    *   Distribution logic: DR lesions usually spare the extreme periphery initially.
*   **AMD (Age-related Macular Degeneration):**
    *   Drusen (yellow deposits) must be concentrated in the Macular region.
*   **Hemorrhage Depth:**
    *   Flame-shaped (superficial/nerve fiber layer) vs. Dot/Blot (deep/inner nuclear layer). AI often confuses the shape with the implied depth.
*   **Henle’s Layer Effect:**
    *   Do macular exudates form a "Star" pattern (macular star) due to Henle fiber arrangement? (AI often misses this specific anatomical constraint).

**Global Features**
*   **Illumination Gradient:**
    *   Is the posterior pole brighter than the periphery? (Natural vignetting).
*   **Texture & Artifacts:**
    *   Look for "smudged" vessel edges or residual debris where exudates might have been imperfectly removed.

---

### Chest X-Ray (CXR)

**3D Projection & Superposition Logic**
*   **Anatomical Overlap:**
    *   Do lung markings (vascular/bronchial bundles) correctly overlap with ribs and the heart?
    *   **Female Breast Shadows:** Are the lower borders sharp, semi-circular, and continuous with the axilla? Does cardiomegaly generation incorrectly distort or erase the breast shadow?
*   **Bone Structure:**
    *   Rib count and continuity. Are the clavicles "S" shaped? Are scapulae projected outside the lung fields (in a standard PA view)?

**Density & Texture Gradient**
*   **Density Ladder:**
    *   Adherence to: Air (Black) < Fat < Water/Soft Tissue < Bone < Metal (White).
*   **Lung Markings (Vascular Markings):**
    *   Are markings visible behind the heart? Do they taper peripherally? Are lower zone markings more prominent than upper zone (gravity effect in erect films)?
*   **Lesion Texture:**
    *   Homogeneous (Consolidation/Atelectasis) vs. Patchy (Pneumonia).
    *   Nodule margins (Spiculated/Lobulated vs. Smooth).

**Secondary Signs (The "Hidden" Checks)**
*   **Atelectasis (Collapse):**
    *   Must see volume loss signs: Elevated diaphragm, Tracheal/Mediastinal shift towards the lesion, Crowding of ribs.
*   **Cardiomegaly:**
    *   If the heart is enlarged (Boot-shaped), are there signs of pulmonary congestion (thickened markings/Kerley B lines)?
*   **Pleural Interaction:**
    *   Do peripheral lesions cause "Pleural Tagging" or retraction?