## HTML as a Non-Regular / Context-Free Language — Cell Summary

**Current state:** The non-regularity sub-claim is proven; the "strictly harder than CFL" sub-claim for HTML5 constructs remains formally unproven; and the applicability of the Chomsky framework to the WHATWG spec itself is contested.

**Key findings**

- **Non-regularity is established by proof, not just evidence.** Four independent Level-2 sources (pumping lemma, Chomsky hierarchy, ODU CS390, CMU 15-411) confirm that unbounded nested tag structures are isomorphic to {aⁿbⁿ} and cannot be recognized by any finite automaton. *(Sources 1–3, 20–21, 31–32)*
- **The "strictly harder than CFL" claim for `<script>`/`<template>` is architecturally plausible but formally unproven.** WHATWG's multi-mode procedural tokenizer implies context-sensitive behaviour, but no published source provides a Chomsky-reduction proof placing HTML5 strictly above Type 2. *(Dependency Auditor, Skeptic: UNADDRESSED_COUNTERARGUMENT)*
- **The WHATWG spec undermines the Chomsky framing altogether.** The normative HTML5 parser is a total transducer — it maps every byte sequence to a DOM without ever rejecting input, meaning its "language" is trivially Σ*, to which the Chomsky hierarchy does not apply. *(CMU 15-411 notes; Fitch & Friederici; Critique Phase skeptic)*
- **Finite browser nesting caps create a scoping problem.** Chromium/Firefox enforce a finite maximum nesting depth K, making the actually-implemented sublanguage technically regular (a very large DFA exists). The pumping-lemma argument holds only for the infinite-depth idealization, which the claim never explicitly invokes. *(HN discussion; Critique Phase Source 3)*
- **The "classical regex vs. PCRE" conflation is unresolved.** Modern PCRE engines with recursive subpatterns (`(?R)`) exceed Type-3 and can match nested HTML, making the colloquial form of the claim false even if the formal-language form is true. *(Neil Madden 2019; John D. Cook 2013)*
- **Practical significance is high regardless.** The non-regularity result is the formal basis for why regex-based HTML sanitizers fail and produce exploitable XSS — the claim's directional correctness for security-engineering purposes is not disputed. *(Pragmatist: DIRECTIONALLY_CORRECT; impact_assessor: CRITICAL_PATH)*

**Open questions**

- Is there a published formal proof (pumping lemma for CFLs, or Turing-reduction) that HTML5 `<script>`/`<template>` semantics place the language strictly above Type 2?
- Should the claim be re-scoped to the *abstract idealized grammar* rather than the WHATWG normative spec, to make the Chomsky classification well-defined?
- Does the claim need to explicitly distinguish classical regular expressions (Type 3) from PCRE-style engines to avoid the terminological conflation flagged across both critique phases?

**Confidence:** Medium — the non-regularity core is high-confidence (proof-level), but the compound claim as written contains one formally unproven sub-claim and one framing ambiguity that the adversarial agents have not seen resolved.