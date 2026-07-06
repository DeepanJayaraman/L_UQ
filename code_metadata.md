# Code metadata

Fill in the cells marked `TODO` once the actions in the "Action needed"
column are complete, then transfer this table into the SoftwareX
manuscript template (it goes directly under the title/author block).

| Nr | Code metadata description | Please fill in this column |
|----|---|---|
| C1 | Current code version | `v1.0.0` *(TODO: tag this version — see below)* |
| C2 | Permanent link to code/repository used for this code version | `TODO`: e.g. `https://doi.org/10.5281/zenodo.XXXXXXX` (Zenodo archive of the `v1.0.0` tag). **Do not submit a plain `github.com/DeepanJayaraman/UQ` link** — SoftwareX requires a permanent, versioned archive because a live repo can change after acceptance. |
| C3 | Permanent link to Reproducible Capsule | Not applicable (no Code Ocean / containerized capsule prepared) |
| C4 | Legal Code License | MIT (the previously bundled third-party `lhsgeneral.m` has been removed; the repository is now 100% MIT) |
| C5 | Code versioning system used | git |
| C6 | Software code languages, tools, and services used | MATLAB |
| C7 | Compilation requirements, operating environments & dependencies | MATLAB (R2018b or later) with the Statistics and Machine Learning Toolbox; no other dependencies; platform-independent (Windows/macOS/Linux wherever MATLAB runs) |
| C8 | If available, link to developer documentation/manual | `https://github.com/DeepanJayaraman/UQ/blob/main/README.md` (function reference table + quick-start usage) |
| C9 | Support email for questions | deepanjayram@gmail.com |

## Action needed before submission: cut a versioned, archived release

SoftwareX's editorial checks require C2 to be a **permanent** link, not a
branch/HEAD link that can change. The standard way to satisfy this:

1. On GitHub, create a release tag for the current state of the repo,
   e.g. `v1.0.0` (`git tag -a v1.0.0 -m "SoftwareX submission" && git push origin v1.0.0`,
   or use the "Releases" tab in the GitHub UI).
2. Link your GitHub account to [Zenodo](https://zenodo.org) (Zenodo →
   Settings → GitHub → toggle on the `UQ` repository).
3. Publish the GitHub release; Zenodo automatically archives it and
   mints a DOI (e.g. `10.5281/zenodo.XXXXXXX`).
4. Put that DOI link in C1/C2 above and in the manuscript's title-page
   footnote / data-availability statement.

This is the one step in this checklist that requires your GitHub/Zenodo
login, so I've left it as a TODO rather than doing it for you.
