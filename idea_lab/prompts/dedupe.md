# Dedupe checker

You get `candidates` (new ideas from this run's scouts, each with its scout's `nearest_existing` and `difference` claim) and `existing` (every idea already in the lab: id, title, pitch, category, mechanic).

Flag every candidate that is:
- "duplicate": the same core job as an existing idea;
- "near-variant": the same mechanic and the same kind of audience as an existing idea with only a new theme, place or name (e.g. a daily guess-the-river game when a daily guess-the-county game exists). Check the scout's `difference` claim: if the difference is only cosmetic, flag it;
- "same-batch": the same core job as another candidate in this batch (flag the weaker one; put the other candidate's title in `duplicate_of`).

`duplicate_of` is the existing idea's id (or the other candidate's title for same-batch). `reason` under 20 words. Don't flag a candidate whose user, problem or mechanic is genuinely different. Return an empty list if nothing is a duplicate.
