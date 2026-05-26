# Code-Reading Question Catalogue

Questions for practicing data-science code reading.

Questions categories are roughly ordered by CRISP-DM phases though you'll find some questions to be cross-cutting. Each question has sub-types indicated by `main.subtype` tags followed by a list of example questions.

## Contents

Phase 1 · Business Understanding: [Business Understanding](#business-understanding) · [Explainability](#explainability)

Phase 2 · Data Understanding: [EDA](#eda)

Phase 3 · Data Preparation: [Transformations](#transformations) · [Pipeline](#pipeline)

Phase 4 · Modeling: [Modeling](#modeling) · [Training](#training)

Phase 5 · Evaluation: [Training](#training) · [Evaluation](#evaluation) · [Pipeline](#pipeline) · [Explainability](#explainability)

Phase 6 · Deployment: [Deployment](#deployment)

Cross-cutting: [Scientific Process](#scientific-process) · [Code Quality](#code-quality) · [Extension](#extension)

---

## Question Categories

### Business Understanding
> **Skill:** Link loss function to business goal; identify missing context for GO/STOP decisions
>
> **Bloom's demand level:** apply - evaluate
>
> **Strong answers:** name cost asymmetry or alternative metric; state what's missing before deciding

`business.objective-alignment`
- Is this loss function aligned with the business cost of getting it wrong?
- What would change if the goal shifted from prediction to anomaly detection?
- Which error type is costlier to which stakeholder?
- Where would we have to ask stakeholders - where might they even disagree?

`business.decision-readiness`
- What information is missing before you can say this model is good enough?
- Train score X, test score Y: is this production-ready?
- At what performance level would you decide this model adds no value?

Note: Translating between business and data-science metrics is shared ground. See also [Evaluation](#evaluation).

---

### EDA
> **Skill:** Link stastical and plot patterns to modeling assumptions; detect confounds and representativeness issues
> 
> **Bloom's demand level:** analyze - evaluate
> 
> **Strong answers:** name violated assumptions or confounds; propose a corrective step

`eda.plot-assumption`
- Describe the shape of this relationship and name the modeling assumption it violates.
- Would a transformation fix this? How would you check?
- Which feature not shown here might improve the "situation"?

`eda.confounding`
- If you included only this variable in a regression, would its coefficient sign be misleading?
- Could there be an unused/unmeasured variable driving the pattern you see?
- What would happen to the coefficient if the grouping variable were omitted (Simpson's paradox)?

`eda.sampling-bias`
- What population does this dataset actually represent?
- Which groups might be over- or under-represented?
- How would results change if the sample were drawn differently?

---

### Transformations
> **Skill:** Trace data through steps; understand what each transformation does and assumes
> 
> **Bloom's demand level:** understand - analyze
> 
> **Strong answers:** give exact shapes or types; name the assumption a transformation makes; predict what breaks when it is violated

`transform.trace-pred`
- What is the shape of the array after this step, and why?
- Trace the data through each transformation and give the shape at each point.
- Does this step silently depend on a prior one in a non-obvious way?
- What value would you expect this variable to hold after this line?
- Without running the code, which outcome would you expect?

`transform.feature-engineering`
- Why was this transformation chosen over a alternative?
- What does this encoding silently assume about category order or frequency?
- What would break if a new category appeared at prediction time?

---

### Pipeline
> **Skill:** Detect silent errors and missing/wrongly-ordered steps
> 
> **Bloom's demand level:** analyze - evaluate
> 
> **Strong answers:** name the exact step that is wrong; predict potential bias on the score

`pipeline.data-leakage`
- Where is this statistic computed, and what data should it have access to at that point?
- Would the reported score look better or worse than the true generalisation score?
- Are the validation and test sets used in the right roles?

`pipeline.validation`
- Why this split strategy rather than a simple train/test split?
- What does this scheme assume about the data that a random split does not?
- How would the reported score change if you used leave-one-out instead?
- For an actual pipeline patterns (sequence of steps): Is it sound?

`pipeline.errors`
- Find the error and explain when and why it would cause problems.
- Would the score look better, worse, or the same compared to correct code?
- Is this bug detectable from the output alone?
- What information is discarded by this line, and could it cause a silent failure?

---

### Modeling
> **Skill:** Map code to algorithm and math; explain what a parameter controls
> 
> **Bloom's demand level:** understand - analyze
> 
> **Strong answers:** name the algorithm; identify where key computation occurs; rate the choice of algorithm w.r.t. outcomes

`modeling.algorithm`
- What algorithm is implemented here?
- Where exactly is that quantity computed in the code?
- What quantity does this model minimize here? Where is it computed (in script or inside a library call)?
- How many parameters will the model learn?

`modeling.param-effect`
- What does this argument control? What are the consequences of changing it to X?
- What would the model become if this parameter were set to its extreme value?

---

### Training
> **Skill:** Read training curves; diagnose convergence and sensitivity to hyperparameters
> 
> **Bloom's demand level:** analyze - evaluate
> 
> **Strong answers:** identify the regime (underfit / converged / diverging); predict the effect of changes to parameter settings

`training.convergence`
- What does this reference line represent?
- If the curve never reaches convergence, what does that tell you? What would you change?
- Identify the point where underfitting transitions to overfitting.
- What does the gap between the two lines represent?

`training.train-dynamics`
- Predict what the loss curve would look like if you doubled this parameter.
- How would you distinguish "too high" from "not converged" in the plot?
- In which scenario would adding more data help more?

---

### Evaluation
> **Skill:** Diagnose model fit from metrics; ensure fair evaluation; compare against baselines
> 
> **Bloom's demand level:** analyze - evaluate
> 
> **Strong answers:** correctly read the gap or score; propose a concrete fix or next step; flag evaluation errors or dishonesties

`evaluation.overfit`
- Train and test scores are both similar — does that mean the model is not overfitting?
- What else in this script might explain a small train-test gap?
- If both models have equal test accuracy, does that mean they learned the same thing?

`evaluation.baseline`
- What would a majority-class predictor score on this metric?
- Is this model meaningfully better than predicting the mean?

`evaluation.metrics`
- Under what conditions would you prefer recall over precision here?
- If the class distribution changed, which metric would be most affected?
- When do accuracy and F1 tell different stories?
- What would a stakeholder focused on X want instead?

→ see also: [Business Understanding](#business-understanding)

---

### Explainability
> **Skill:** Map learned weights or importance scores to domain meaning
> 
> **Bloom's demand level:** analyze - evaluate
> 
> **Strong answers:** connect direction and magnitude to real-world effects; flag implausible rankings

→ see also: [Business Understanding](#business-understanding)

`explain.importance`
- Do the top-ranked features make domain sense?
- If two correlated features are both present, how does that affect their scores?
- What would change if you removed the top feature entirely?

`explain.coeff-meaning`
- What does a positive coefficient on this feature mean in plain language?
- If you scaled this feature by 10, what would change in the coefficient?
- Can you compare coefficient magnitudes across features directly?

---

### Deployment
> **Skill:** Identify unstated assumptions and anticipate performance degradation after deployment
> 
> **Bloom's demand level:** evaluate
> 
> **Strong answers:** name a specific assumption and a realistic scenario where it breaks

`deployment.input-assump`
- What does this step silently assume about the input at prediction time?
- What would happen if a single feature were missing for a new observation?
- What would happen if a category unseen during training appeared in the live data?

`deployment.prod-fit`
- Name two assumptions this model makes that might not hold in a live system.
- For each, describe a realistic scenario where it breaks.

`deployment.dist-shift`
- Which features are most likely to behave differently in a new deployment context?
- How would you detect that the model's performance had degraded after deployment?

---

### Scientific Process
> **Skill:** Evaluate whether the investigation follows sound scientific and engineering discipline
> 
> **Bloom's demand level:** evaluate
> 
> **Strong answers:** rate best practices, name violations and describedistortions they introduce

`process.simplicity`
- What is the simplest model that could answer this question?
- Was a baseline tried before this approach? What would it look like?
- Is the added complexity here justified? With what evidence?

`process.controlled-change`
- Between these two runs, how many things changed? What does that mean for interpreting the difference?
- How would you isolate whether this component actually contributes?
- If you removed this step, what would you expect? How would you confirm it?

`process.reproducibility`
- Which outputs would change across runs if the seed were removed?
- Which outputs are deterministic regardless of the seed?
- How would you make this pipeline fully reproducible?

---

### Code Quality
> **Skill:** Identify structural issues that cause silent failures, reduce maintainability, or reflect AI code-generation patterns
> 
> **Bloom's demand level:** analyze
> 
> **Strong answers:** name the exact line or pattern; explain what breaks and under what condition

`quality.magic-number`
- What does this constant represent, and should it be a named parameter?
- What would break if this value changed without updating downstream code?
- Is this value data-dependent or a design choice?

`quality.abstraction`
- Does this function do more than one thing? Where is the natural split point?
- Is the level of abstraction consistent throughout, or does it mix high-level calls with low-level mechanics?
- Do the names reveal what the code intends, or what it does?
- Which parts would need to change if you swapped out this component?

`quality.idioms`
- Is this written the way a practitioner would use this library?
- What would this look like in idiomatic pandas / numpy / sklearn?
- Is the code working with the library's design, or around it?

`quality.dead-code`
- Can any of these lines be removed without changing behavior?
- Is this guard or condition reachable given the upstream context?
- What is this defensive check actually protecting against — can it trigger?


---

### Extension
> **Skill:** Synthesize the script's findings and identify what comes next
> 
> **Bloom's demand level:** evaluate - create
> 
> **Strong answers:** point to a specific result or gap in the script; propose a concrete, realistic next step

`extension.follow-up`
- What modeling approach would you try after seeing these results?
- What experiment would confirm or challenge the main finding?

`extension.gaps`
- What question does this script leave unanswered that a stakeholder would ask?
- Which finding here would you not trust without additional validation?
- What would you need before handing this off to a deployment team?

`extension.alternatives`
- What different approach would you use to answer the same question, and why might it be preferable?
- Why might a practitioner make a different modeling choice here?