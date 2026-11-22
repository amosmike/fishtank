# Logistics checker

Bigger builds fail on logistics, not ideas: no licensed data, a free tier that breaks at real traffic, a name already taken, a daily game that runs out of content after a month. For each idea in the input, judge the logistics.

The pipeline has ALREADY run code checks for you, in each idea's `code_checks`:
- `domains`: RDAP lookups for the working title (.com via Verisign, .uk/.co.uk via Nominet): "registered", "not found (likely free)" or "unknown". Use these for `domain_check`; don't re-search domains.
- `urls`: whether each cited URL resolves (HTTP status).
- `licence_pages`: the text of the data-source and licence pages, fetched now. Read them for the licence terms.

Use your few web searches only for what the code couldn't do: existing games/products with the same name (`name_check`: search the title and 1–2 alternatives), current price pages and free-tier limits, ad-network or portal policy.

For each idea:
- `data`: the exact source, its download or API URL, licence and attribution terms, update frequency, and how much hand curation it needs. For a daily game or reader: can it fill 365 days, and how many hours a week to keep it going?
- `stack` and `free_tier_limit`: the lightest stack that works and the limit that breaks first at 1,000 and 10,000 visits a day (Cloudflare Workers/Pages/D1/R2 or Supabase).
- `cost_monthly`: at 1,000 and 10,000 visits a day from current price pages (LLM or paid API calls included), against a rough ad income of 1–3 GBP per 1,000 visits (say it's a rough guess). `price_url` is a real pricing page.
- `effort_days`: realistic build days for one person working evenings and weekends with Claude Code; `riskiest_part`; `upkeep_hours_week`.
- `ad_policy` and `compliance`: AdSense/portal acceptance, cookie consent for ads, anything aimed at children (UK ICO Children's Code, AdSense child-directed rules), accounts or uploads (UK GDPR, moderation), scraping and API terms, image/audio/map/trademark rights.
- `verdict`: "go", "fixable" (list `fixes`) or "blocker" (list `blockers`). Set `blocker_is_legal` true only for a legal problem, and `blocker_has_no_workaround` true only when nothing works around it (no usable data licence, costs ads can't cover even at 10,000 visits a day). `note`: one sentence.
