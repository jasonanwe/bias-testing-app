# Sample data

Three files for trying the import and for testing changes to the parser.

**None of this is real data.** Every row is synthetic, generated for this repository. No file contains information about any actual person, case, program, or agency.

## The files

**`bias-testing-data-template.csv`**

The format, in eight rows. Three columns: an identifier the tool ignores, a protected class, and an outcome. Start here if you want to see the shape of a file the tool accepts, or use it as a starting point for your own export.

**`sample-2-realistic.csv`**

640 rows across nine columns, with five columns that can each serve as the protected class: race and ethnicity, sex, age band, primary language, and county type. The outcome column carries four values, so you have to decide which count as favorable rather than having two obvious choices.

This is the file to use for exercising most of the tool. Running it on race and ethnicity produces several comparisons at once, which brings in the multiplicity adjustment and flags two groups that fall under 30 observations. Running it on primary language produces a larger disparity than race does, which is a useful demonstration that a system can look acceptable on one dimension and not on another.

**`sample-3-messy.csv`**

Ten rows built to break a parser. It contains quoted fields with embedded commas, an escaped quote inside a quoted field, a byte order mark of the kind Excel writes when saving as CSV UTF-8, blank cells in both the class and outcome columns, leading and trailing spaces, and inconsistent casing.

Importing it should read 10 rows, count 8, and report 2 skipped for a blank class or outcome value. If a change to the parser breaks any of that, this file will show it.

## Generating your own

The tool needs one row per decision, a header row, one column identifying the protected class, and one holding the outcome. Any other columns are ignored. See the Importing outcome data section of the main README for the full requirements.
