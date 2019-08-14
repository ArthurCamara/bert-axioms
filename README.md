# DeepIR-Interpretability
What can we extract from deep models to use back in more traditional IR settings?

## Argument chain
1. We have seen that with sufficient training data, deep retrieval models outpeform retrieval models that are not ML-based (BM25, language modeling). Let's call the latter *traditional models* (ignoring L2R for now).
2. In many use cases we do not have access to a lot of training data, rendering deep retrieval models useless.
3. We also know that there are retrieval heuristics that traditional models are based on (heuristics that experts came up with). We also have so-called axioms that good retrieval models should fulfill.
  - Axioms have been employed in the past to "fix" traditional retrieval models, which overall led to better retrieval effectiveness. Axioms are limited though - they depend on ideas/insights of experts. They help us to identify shortcomings of deep retrieval models (which axioms are they performing poorly on?), but they do not help us to learn what kind of novel heuristics those deep models have learnt that our traditional models lack.
4. We need to find a technique that allows us to extract retrieval heuristics from trained deep models.
5. We need to inject those novel heuristics into traditional models and investigate whether the retrieval effectiveness improves.

Overall, we want better traditional models that do not rely on training data by exploiting insights from deep models.

Our research contribution is the move from trained deep model to retrieval heuristic(s).

## Related work
- Axioms (how did people come up with them, how have they been employed to fix retrieval models)
- Deep retrieval models (types, achievements)
- Analysing deep models (approaches, insights)
