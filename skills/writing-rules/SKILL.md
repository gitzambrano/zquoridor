---
name: writing-rules
description: "Use when writing or reviewing any prose in this project: comments, docstrings, commit messages, helper text, tooltips, dialog and error messages, README files, markdown docs, or any user-facing documentation. Adapts ASD-STE100 (Simplified Technical English) into a concise rules table and a review checklist. Triggers: writing documentation, writing a docstring, writing GUI or CLI help text, reviewing text for clarity, technical writing style."
---

# Writing rules

This skill adapts ASD-STE100 (Simplified Technical English), the aerospace
controlled-language standard, for any technical project. It governs every
piece of prose in a repository: source comments, docstrings, commit
messages, CLI help and error text, UI strings, README files, and markdown
documentation.

ASD-STE100 was built so a maintenance technician can never misread an
instruction. The same discipline keeps a project's documentation short,
unambiguous, and easy to translate or parse, by a human reader or by an
agent.

Text is either **procedural** (steps, CLI usage, UI instructions, a
tutorial) or **descriptive** (an explanation, a docstring, a module
overview, prose in docs). The General rules apply to both. The Procedural
and Descriptive rules apply only to their own kind of text.

## General rules

| # | Do | Don't |
|---|----|----|
| G1 | Write in formal, technical English. "The importer did not finish within the time limit." | Write in a casual or conversational tone. "Looks like the importer just ran out of time." |
| G2 | Give each sentence one topic or one instruction. "Set the batch size. Then run the job." | Chain unrelated ideas into one sentence. "Set the batch size and run the job, which then writes the report." |
| G3 | Use active voice. "The loader validates the manifest." | Use passive voice when the actor is known. "The manifest is validated by the loader." |
| G4 | Use only simple verb forms: infinitive, imperative, simple present, simple past, simple future, or past participle as an adjective. "The config file failed to load." | Build a compound tense with an auxiliary verb. "The config file has failed to load." |
| G5 | Use a past participle as an adjective to show a state. "Examine the corrupted index." | Use it to build a passive construction. "The index was corrupted by a crash." |
| G6 | Write the full sentence: its verb, its articles, and any connector it needs. Don't use contractions. "If a custom schema is used, validate it first." | Write telegraphic text that drops the verb, the articles, or an adverbial connector to save space. "Custom schema: validate first." |
| G7 | Keep a multi-word noun to 3 words or fewer. "Timeout for the metadata cache." | Stack four or more nouns together. "Metadata cache request timeout handler." |
| G8 | Use a single plain verb. "Remove the stale entry." | Build a phrasal verb from two words. "Take out the stale entry." |
| G9 | Use an "-ing" word only as a technical noun or as a modifier in one. "See the Logging appendix." | Use "-ing" as a verb form. "While debugging the build, check the log." |
| G10 | Use the same term for the same thing every time. Always "config file," never switch to "settings file" mid-document. | Rotate synonyms for the same concept. "Config file" in one line, "settings file" in the next. |
| G11 | Use a plain verb to name an action. "Check the schema." | Turn the action into a noun. "Do a check of the schema." |
| G12 | Use an article before a noun. "Open the results tab." | Drop the article for a telegraphic style. "Open results tab." |
| G13 | Use plain, well-known words. "The batch run failed." | Use slang or jargon outside the project's own vocabulary. "The batch run bombed." |
| G14 | Split related facts into separate sentences. "The test did not pass. Increase the timeout." | Join them with a semicolon. "The test did not pass; increase the timeout." |
| G15 | Use a plain sentence break instead of a dash. "The flag changes the default port. This affects local development." | Use a dash to join two ideas. "The flag changes the default port — this affects local development." |
| G16 | Add a connector when one sentence follows logically from the one before it: *however*, *therefore*, *thus*, *then*, *nevertheless*. "The cache is empty. Therefore, the loader reads from disk." | Leave the logical link implicit. "The cache is empty. The loader reads from disk." |
| G17 | Use "that" after verbs like "make sure," "confirm," or "show." "Make sure that the schema is loaded." | Drop "that" and risk a misread clause. "Make sure the schema is loaded." |
| G18 | Replace a pronoun with the noun it refers to, if more than one noun could fit. "If the certificates are expired, renew the certificates." | Leave an ambiguous pronoun. "If the certificates are expired, renew them." |
| G19 | State what "this" refers to when more than one reading is possible. "If the endpoint is rate-limited, this limit applies per key." | Leave "this" to point at an unclear antecedent. "If the endpoint is rate-limited, this applies per key." |
| G20 | Spell out "for example," "that is," "and so on." | Use a Latin abbreviation. "e.g.," "i.e.," "etc." |
| G21 | Use a numbered or bulleted list for a sequence or a set of conditions with 3 or more items. | Bury a 3-step sequence inside one paragraph of prose. |
| G22 | Keep sentence length in check: about 20 words for an instruction, about 25 words for a description. | Write a long sentence with several clauses. |
| G23 | When an approved word does not fit, restructure the sentence around a word that does. "Make sure that you can see the free disk space." | Force a word-for-word swap that reads as nonsense. "Make sure that the free disk space is visible." (when "visible" is not an approved word here) |
| G24 | Re-read every sentence that uses "with." Confirm the reader cannot mistake it for association, means, or an instrument. "Encrypt the backup with the project key." | Leave "with" open to more than one reading. "Update the table with the new rows." (unclear whether "with" means "using" or "together with") |
| G25 | In prose, write a sentence break instead of `--`, and write "approximately" instead of `~`. "Compression is lossless. The ratio is approximately 3 to 1." | Use `--` or `~` as a substitute for punctuation or a word in prose. "a lossless--compression codec", "~3x ratio" |
| G26 | Write "and" or "or" between two words. "the client and server logs", "a GET or POST request" | Join two words with a slash and leave the reader to guess which you mean. "Client/server logs", "a GET/POST request" |
| G27 | Keep a parenthesis short, and put a fact the reader needs in its own sentence. | Hide a needed fact inside a long parenthesis. |
| G28 | Write a range in words. "5 to 30 retries" | Write a range with a hyphen. "5-30 retries" |
| G29 | Write a cross-reference in full. "Section 5" | Abbreviate a cross-reference. "Sec.5" |
| G30 | State what the code or the specification defines or computes. "the checksum of the payload", "The protocol does not specify this case." | Give code or a specification human intent. "the checksum the payload sees", "The spec says nothing about this case." |
| G31 | Use American spelling. "behavior", "normalize", "labeled", "center" | Use British spelling. "behaviour", "normalise", "labelled", "centre" |
| G32 | State only what is factual and measurable, and keep a hedge exactly as strong as the source. "The migration finished in 12 seconds." "The run may fail." | Use a superlative, a marketing word, an exaggeration or a non-technical vague word (robust, powerful, seamless, huge, the best). Stack hedges until the sentence claims nothing ("it may potentially help to improve"), or promote a hedge to a fact. |

**`G25`, `G26` and `G28` never apply to code.** A command-line flag keeps
its dashes (`--project`, `--set`, `--max-iter`). A hyphenated compound
adjective keeps its single hyphen (`a content-addressed store`, `a
fixed-size buffer`). A file path, an operator, a numeric literal and a
range inside a code sample stay exactly as the code writes them. These
rules govern prose only.

All documentation in a project, including docstrings, helper text,
tooltips, and UI strings, must follow this table.

## Procedural rules (steps, CLI usage, UI instructions)

| # | Do | Don't |
|---|----|----|
| P1 | Write one instruction per sentence, unless two actions happen at the same time. "Open the Settings tab. Select the profile." | Merge two sequential steps into one sentence. "Open the Settings tab and select the profile." |
| P2 | Write instructions in the imperative form. "Set the concurrency level." | Describe the instruction instead of giving it. "The concurrency level can be set." |
| P3 | State a condition first, then the command, separated by a comma. "If the lock file exists, abort the import." | Bury the condition after the command. "Abort the import if the lock file exists." |
| P4 | Use a note only to give information. "Note: the dry-run flag also checks permissions." | Put an instruction or a requirement inside a note. "Note: run the dry-run before the import." |
| P5 | Name the concrete risk in a warning or an error message. "This deletes the database and its backups." | State an abstract risk. "This action is not recommended." |

## Descriptive rules (explanations, docstrings, module overviews, prose)

| # | Do | Don't |
|---|----|----|
| D1 | Give information gradually, one subject per sentence. "The indexer builds an inverted index. It processes documents in batches." | Front-load several facts into one dense sentence. "The indexer, which processes documents in batches, builds an inverted index using a background worker pool." |
| D2 | Open a paragraph with a topic sentence that states its subject. | Start a paragraph mid-detail, with the topic implied. |
| D3 | Keep one topic per paragraph, and keep each paragraph to 6 sentences or fewer. | Mix two topics, or let a paragraph run past 6 sentences. |

## How to review a text

1. Read the text once, for meaning only. Do not edit yet.
2. Decide whether it is procedural or descriptive, and pull in that
   section's rules (`P1`-`P5` or `D1`-`D3`) along with the General rules.
3. Go through every applicable rule, one at a time, from `G1` to the last
   rule in each table. For each one, check whether the text complies. This
   step is mandatory: do not skip straight to a general impression.
4. `G1`: confirm the tone is formal and technical, not casual or
   conversational.
5. `G2` and `P1`: split any sentence that holds more than one topic or
   instruction, unless the actions happen at the same time.
6. `G14`: split any sentence joined by a semicolon.
7. `G15`: replace any dash with a plain sentence break.
8. `G10`: look for synonym rotation, the same thing named two different
   ways. Pick one name and use it everywhere.
9. `G11`: look for a nominalization ("perform a check of"). Replace it
   with the verb ("check").
10. `G32`: delete every superlative, marketing word and exaggeration, and
    state the claim the text is hedging around. Do not change how strong
    the original claim was.
11. `G7` and `G8`: replace any remaining noun cluster over 3 words, or any
    phrasal verb, with a plain, specific one.
12. `G16`: add a connector where one sentence's meaning depends on the
    sentence before it.
13. `G23` and `G24`: confirm any reworded sentence keeps its original
    meaning, and that every "with" reads without ambiguity.
14. `G25`, `G26` and `G28`: in prose, replace `--`, `~`, a joining slash
    and a hyphenated range. First confirm the text is prose. Never touch a
    command-line flag, a compound adjective, a path, or a code sample.
15. `G20`, `G29`, `G30` and `G31`: replace a Latin abbreviation, an
    abbreviated cross-reference, a phrase that gives code human intent,
    and any British spelling.
16. `D3` (descriptive text only): confirm each paragraph has one topic and
    6 sentences or fewer.
17. Reread the whole text start to finish. Confirm every rule from step 3
    is satisfied, and that no fact or hedge was lost in the edit.

## Scope

This skill governs prose and user-facing strings. It does not govern code
identifiers, which follow the project's existing naming conventions. It
complements, and does not override, other project conventions.
