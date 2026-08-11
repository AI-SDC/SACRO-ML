# AI Contributions Policy

## Overview

We welcome AI-assisted contributions. AI tools can be genuinely useful for writing, editing, refactoring, and exploring ideas. However, AI is a tool, not a contributor. Every submission must be owned, understood, and vouched for by a human.

The rules below exist to respect everyone's time, preserve code quality, and ensure authentic engagement in our community. They apply to all contributors equally, regardless of role.

## All AI usage must be disclosed

If any part of your contribution - code or documentation - was meaningfully shaped by an AI tool, you **must** state this in the opening comment of your pull request. Include the specific tool and version, and a brief description of how it was used.

Version matters. "ChatGPT" tells us very little. "ChatGPT o3" or "Claude-Sonnet-4-5" tells us something meaningful about the capabilities and tendencies of the model involved. If you do not know the version, check before submitting. If you still cannot determine it then provide as much information as possible.

Routine use of AI for spell-checking or light grammar correction does not require disclosure.

Example AI usage disclosure:

```text
This PR used ChatGPT o3 to help refactor a parser function and generate an initial set of unit tests. All code and tests were reviewed, simplified, and rewritten where necessary.
```

## You must fully understand everything you submit

If you cannot explain what your changes do, why they are the correct approach, and how they interact with the rest of the codebase *without the aid of AI tools* your contribution is not ready for review. When you open a PR, you **must** provide a *short* summary that explains what it does and why the changes were made - these **must not** be written by AI.

This is the most important rule in this policy. We will ask questions during review. "The AI suggested it" is not an answer.

## AI-generated code must be indistinguishable from high-quality human work

Submitting a raw or lightly edited AI output is not acceptable. Before opening a PR, you are expected to thoroughly read, test, and clean up any AI-generated code. In practice this means:

- **Remove unnecessary code.** AI models routinely generate guards for conditions that cannot realistically occur because the model is overly-cautious. If you cannot point to a real scenario where a guard fires, remove it.
- **Remove redundant or superfluous tests.** AI-generated tests frequently cover unreachable states, re-test behaviour already covered by other tests, or test the language itself rather than your code. Every test you submit should cover a real, meaningful case. If a test would pass regardless of whether your code is correct, it has no place here.
- **Strip all AI artefacts.** This includes issue numbers embedded in docstrings or comments, references to specific line numbers, summaries of what a function does written in a way that mirrors the prompt, `TODO` comments left by the model, and any other content that reads like the model narrating its own output. None of this belongs in submitted code and its presence signals the output has not been read.
- **Remove unnecessary comments.** AI models over-comment. Comments that restate what the code obviously does add noise and should be deleted. Comments should explain *why*, not *what*.
- **Eliminate inconsistent or incorrect naming and style.** AI output frequently mixes naming conventions, uses overly generic identifiers, or invents abstractions that do not exist elsewhere in the codebase. Rename things to match the project's conventions and remove abstractions that are not pulling their weight.
- **Do not pad scope.** AI models have a tendency to add things that were not asked for such as extra utility functions, additional configuration options, broader error hierarchies, convenience overloads. If it was not part of the intended change, remove it. Scope creep in AI-generated PRs is common and wastes review time. **Keep PRs small and focused.**
- **Conform to project style.** Beyond naming, this means matching the project's patterns for error handling, logging, module structure, and code organisation. AI output is trained on the entire Internet, not this repository.
- **All tests must pass.** This obviously applies to both human and AI-assisted contributions.

If a PR shows obvious signs of unreviewed AI output, it may be closed without detailed feedback. Cleaning up someone else's AI-generated junk is not a reasonable thing to ask of a reviewer.

## Issues and discussions must be written by you

You **must not** use AI to write issue reports, pull request descriptions, or discussion comments.

This is intentional. Writing a PR description or issue in your own words is one of the clearest signals that you actually understand what you are submitting. A concise, accurate, human-written description saves everyone time and demonstrates genuine engagement with the problem.

AI-generated descriptions are often verbose, imprecise, and critically, may not accurately reflect what the code actually does. They also tend to pad with noise: lists of files modified, marketing-style statements of purported benefits (e.g., "improves robustness"), and bullet-point summaries of trivial changes that restate what the code obviously does. A reviewer who reads a description like that learns nothing useful and may be actively misled. PR descriptions **must not** contain any of this. Inclusion of such text may be cited as evidence of AI generation and failure to read and understand this policy.

If you cannot write a clear description of your change in your own words, that is a sign the contribution is not ready.

Posting AI-generated content via automated bots or agents is strictly forbidden. Accounts repeatedly doing this may be banned and reported to GitHub as spam.

## Documentation

The same rules that apply to code apply to documentation. AI tools can hallucinate or invent details, as well as produce confident-sounding text that is subtly wrong. Review everything carefully and make sure you can stand behind it.

## AI-generated media

AI-generated images, illustrations, audio, or other media assets are not accepted without explicit prior approval from the project maintainers due to copyright reasons.

## Enforcement

Contributions that appear to be low-effort AI output, including anything that reads like it was generated without genuine engagement with the codebase, will be closed without detailed feedback.

AI **must not** be used as the sole basis for approving or rejecting a contribution. A reviewer is always accountable for their decision, and that decision must reflect their own understanding of the change.

When someone submits unreviewed AI output, they are not really contributing - they are offloading the actual work of understanding, validating, and cleaning up onto the person reviewing it. That is not a fair exchange of effort, and it is not acceptable regardless of who is doing it.

Repeated low-quality submissions may result in a contributor being blocked.

## Why this policy exists

This is not an anti-AI policy. It is an anti-slop policy.

Reviewing a contribution takes real time and attention from another person. When a submission has not been genuinely understood or cleaned up by its author, the reviewer ends up doing all the actual work. That is true whether the author is a first-time contributor or a long-standing one. This policy applies to everyone.

If you are using AI to learn - great. Use it to understand the codebase, experiment locally, and build your skills. When you submit, the work should be yours: you read it, you tested it, you understand it, you can defend it, and you assert that you are the copyright owner (or if it includes parts of other open source software, the license is included).

---

This policy draws on the [Ghostty AI Usage Policy](https://github.com/ghostty-org/ghostty/blob/main/AI_POLICY.md), the [htop AI-Assisted Contributions Policy](https://github.com/htop-dev/htop/blob/main/docs/ai-contributions-policy.md), and the [Fedora AI Contribution Policy](https://docs.fedoraproject.org/en-US/council/policy/ai-contribution-policy/).
