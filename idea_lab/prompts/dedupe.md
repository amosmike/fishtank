# Dedupe checker

You get `candidates`: new ideas from this run's scouts, each with its scout's `nearest_existing` and `difference` claim and, when search is on, `nearest_in_lab`: the existing ideas most similar to it (id, title, pitch, similarity score), found by hybrid keyword + vector search. `all_titles` maps every existing idea's id to its title, so you can still catch a duplicate the search missed. (Without search you get `existing` instead: every idea's id, title, pitch, category and mechanic.)

Flag every candidate that is:
- "duplicate": the same core job as an existing idea;
- "near-variant": the same mechanic and the same kind of audience as an existing idea with only a new theme, place or name (e.g. a daily guess-the-river game when a daily guess-the-county game exists). Check the scout's `difference` claim: if the difference is only cosmetic, flag it;
- "same-batch": the same core job as another candidate in this batch (flag the weaker one; put the other candidate's title in `duplicate_of`).

`duplicate_of` is the existing idea's id (or the other candidate's title for same-batch). `reason` under 20 words. Don't flag a candidate whose user, problem or mechanic is genuinely different. Return an empty list if nothing is a duplicate.
