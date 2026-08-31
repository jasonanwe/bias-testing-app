# AI Bias Testing Tool

A single-file, browser-based tool for running a disparate impact analysis on an AI system and producing a report you can save as a PDF.

It walks through seven steps: system details, group data, metrics, statistical significance, context, classification, and report. Everything runs locally in the browser.

## Two builds

`index.html` is the standalone build: one file, no dependencies, no build step. Download it and open it in any modern browser, host it on GitHub Pages, or run it from `file://` on a machine with no network connection.

`BiasTestingTool.jsx` is the same tool as a React component, for dropping into an existing app. It exports a default component with no required props and expects React 18 or later and Tailwind CSS for styling.

Both produce identical results. The standalone build is the one to use if you just want to run a test.

## What it does

You enter two groups defined by a protected class along with their favourable and unfavourable outcome counts. The tool then calculates the selection rate for each group and the disparate impact ratio, evaluated against the four-fifths rule from the Uniform Guidelines on Employee Selection Procedures (29 CFR 1607.4).

It runs three tests of statistical significance and uses the strongest applicable result: Fisher's Exact Test, which is valid at any sample size, the Chi-Square Test of Independence, which requires expected cell counts of five or more, and the Two-Proportion Z-Test, which requires at least 30 observations per group. Tests that are not valid for your data are marked as such rather than reported.

Four context questions then adjust the finding: the type of decision, who it affects, how many decisions are made per year, and how reversible the outcome is. A decision that directly determines outcomes for vulnerable populations at scale carries more weight than an informational output affecting internal staff.

## How the classification works

The quantitative finding comes from the disparate impact ratio:

| Ratio | Finding |
|---|---|
| 0.90 and above | None |
| 0.80 to 0.89 | Low |
| 0.65 to 0.79 | Moderate |
| Below 0.65 | Critical |

That finding is then raised by up to two levels based on the highest context score, and adjusted by statistical significance. A result that is not significant (p greater than 0.10) reduces the finding by one level, on the basis that the disparity could reasonably be due to chance. A highly significant result (p at or below 0.01) raises it by one.

The report includes a recommended response and a re-review interval with a target date, scaled to the finding.

## Plain-language summary

Alongside the numbers, the tool writes a short narrative explaining what the result means: what was tested, how large the gap is in everyday terms, whether the statistics support treating it as real, which context factors moved the severity, and what happens next. It is generated from the figures by a deterministic function, so the same inputs always produce the same wording and nothing is sent to a model or an API. That matters for a governance record, where a summary that changes between runs would be difficult to defend.

## Privacy

Every calculation happens in your browser, including the plain-language summary. The tool makes no network requests, stores nothing, and has no analytics, no cookies, and no external dependencies. Fonts are drawn from the system stack rather than a font CDN, so nothing is fetched at load time. Data you enter exists only in the page and disappears when you close the tab.

This matters if you are testing systems that touch sensitive data, because the counts never leave the machine.

## Saving the report

The report step opens your browser's print dialog. Choose "Save as PDF" as the destination and keep background graphics enabled so the classification colour is preserved in the saved file.

## Limitations

This tool is a screening instrument. The four-fifths rule is a threshold for further investigation, not a definitive legal standard, and a passing ratio does not establish that a system is free of bias.

It compares two groups at a time on a single binary outcome. Testing several protected classes, or an outcome that is not a simple favourable or unfavourable split, means running the analysis more than once and interpreting the results together.

Statistical tests have limited power at small sample sizes. When a group has fewer than 30 observations, treat the result as directional and gather more data before drawing conclusions.

Nothing here is legal advice. Disparate impact analysis sits in a contested area of law that varies by jurisdiction and by the domain the system operates in. Involve counsel before acting on a finding.

## Methodology and references

- NIST AI Risk Management Framework 1.0 (AI 100-1), Measure function
- NIST SP 1270, Towards a Standard for Identifying and Managing Bias in Artificial Intelligence
- EEOC Uniform Guidelines on Employee Selection Procedures, 29 CFR 1607.4
- EEOC Technical Assistance on AI and Title VII

## Contributing

Issues and pull requests are welcome. The standalone tool is one HTML file with no build tooling, so a change there is a direct edit to `index.html`. Changes to the calculation or the summary wording should be made in both `index.html` and `BiasTestingTool.jsx` so the two builds stay in step.

## License

See `LICENSE`.
