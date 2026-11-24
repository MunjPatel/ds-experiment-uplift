# Interview narrative — Experimentation & Uplift

**30-second version**  
“I don’t stop at a t-test. I show SRM, A/A health, and why peeking lies — then I train a T-learner uplift model on Criteo incrementality data and prove that targeting top uplift beats treating high-propensity converters.”

## Structure

1. Business: budgeted treatments (retention, cross-sell, collections)  
2. Diagnostics: Kohavi/Airbnb pitfalls  
3. Uplift ≠ propensity  
4. Qini / top-decile lift  
5. Limitations: anonymized features; assumed treatment cost  

## Likely follow-ups

| Question | Direction |
|----------|-----------|
| Why T-learner? | Simple, interviewable; X-learner next if residual bias. |
| Criteo treatment imbalance? | Design rate ~85%; SRM vs empirical assignment, not vs 50/50. |
| Finance angle? | Same math for retention offers and collections outreach. |
