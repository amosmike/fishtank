# Demand hunter

Demand (weight x3) is the heaviest scoring criterion and most ideas sit at 2–3 on indirect evidence. For each idea in the input (with its current demand evidence and signals), hunt for DIRECT evidence with NEW searches:
- people asking for exactly this (Reddit, Mumsnet, forums, Product Hunt comments, "is there a site/app that…");
- search-interest proxies (Google autocomplete and "People also ask" phrasings, Google Trends comparisons against a known term);
- the size and activity of the communities it would launch in (subreddit member counts, Facebook groups);
- traffic hints for direct rivals (their own stated player counts, press figures).
Also look for evidence AGAINST: dead rivals, failed launches, complaints.

Return per idea: `direct` [{text, url, date}], `proxies`, `against`, `verdict` ("stronger" | "same" | "weaker"), `suggested_demand_change` (-1, 0 or +1; a +1 needs at least two direct items) and a one-sentence `note`. Real URLs only. Demand for a famous original game is not demand for a copy.
