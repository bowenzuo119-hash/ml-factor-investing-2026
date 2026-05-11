# Worklog

Per-person daily/weekly log of what each teammate did, decided, or is blocked on.

## Why this exists

So each of us can:
- See what the other two are doing without pinging chat all the time
- Catch coordination problems early (e.g. "I'm waiting on the backtest interface")
- Have a paper trail when we write the 5-week report

## Convention

- One file per person: `person-a.md`, `person-b.md`, `person-c.md`
- Newest entries at the **top** of each file
- Each entry uses this template:

```
## YYYY-MM-DD

**Done:**
- ...

**In progress:**
- ...

**Blocked on / need from teammates:**
- ...

**Decisions / notes:**
- ...
```

## Rules

- Don't edit each other's worklogs (open a PR if you want to add a note for someone else)
- Keep entries short — 5 bullets per section max
- Anything that's a real project-wide decision goes in the top-level `DECISIONS.md`, not here
- Push your worklog updates on your own feature branch, not a separate worklog PR every time

## Owners

| File | Owner | GitHub |
|------|-------|--------|
| person-a.md | Bowen | @bowenzuo119-hash |
| person-b.md | A. Fontana | @agfontana |
| person-c.md | Nicolas | @nicolascoutomota-boop |
