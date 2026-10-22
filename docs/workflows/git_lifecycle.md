# Git Branch Lifecycle

Use one protected integration branch, short-lived task branches, and an
explicit retirement assessment at handoff. Branch refs, worktree directories,
scientific artifacts, and task/run records have separate lifetimes.

## Authority and scope

[AGENTS.md](../../AGENTS.md) defines the mandatory contribution and validation
constraints. This document defines the HW_analysis lifecycle; the shared
`git-worktree-cleanup` skill owns auditing, archival, deletion, verification,
and recovery mechanics. Use `venus-hpc` for remote execution and deployment,
and `command-center` for associated tracked work. Do not duplicate their
procedures or introduce a separate branch-status registry.

These are contribution rules, not installed Git hooks or hosted branch
protections. Adopting them does not itself authorize commits, integration,
publication, deployment, job submission, transfers, or deletion. Execute only
the operations covered by the current request. When a request includes
retirement, complete the authorized steps without repeatedly asking for the
same permission; local and remote deletion remain separately scoped. An audit
or merge-only request does not authorize cleanup.

## Protected integration and retained history

- `master` is the sole long-lived integration branch. Do not develop directly
  on it, delete it, or rewrite its published history.
- Use refreshed `origin/master` as the shared integration baseline. Verify the
  actual branch, SHA, remote, and worktree rather than inferring them from a
  directory name. If the remote cannot be verified, state that limitation;
  do not claim a cached ref is current or retire work on that assumption.
- Keep one stable, clean local `master` checkout. Inspect and reconcile any
  local-only integration commits deliberately; never discard them to match
  the remote.
- Preserve archive tags and other refs that retain otherwise unreachable
  commits needed by recorded runs or artifacts. Their removal is a separate
  reviewed operation, not routine branch cleanup.
- Never rewrite published history or commits referenced by recorded runs.
  In particular, do not routinely rebase or squash such task branches.

## Create or resume a task branch

1. Inspect existing branches, registered worktrees, and relevant task context.
   Reuse the branch for the same unfinished outcome instead of making another
   branch for an agent session, region, render attempt, or scheduler retry.
2. Start new work from refreshed `origin/master`. Starting from another task
   branch requires an explicitly recorded dependency and integration order.
   Do not unintentionally include unfinished work from a different task.
3. Use descriptive names such as `feat/presentation-heating`,
   `fix/window-reduction`, or `docs/branch-lifecycle`. Existing branches do not
   need cosmetic renaming. New follow-up work after integration starts from
   current `origin/master`, not from a completed branch.
4. For tracked work, record the branch, base SHA, checkout path, intended
   outcome, dependencies, and acceptance criteria in the existing task. Use
   the command-center CLI through its skill. Ordinary synchronous work can
   record this context in its handoff without creating an artificial run.
5. Before implementation, follow the documentation-first change process in
   [the documentation index](../README.md). Define compatibility and the
   local and, where applicable, Venus validation required for integration.

Resuming a paused branch includes reviewing its divergence from current
`origin/master`, unpublished commits, and recorded dependencies. Preserve that
history while updating it for the resumed task.

## Allocate worktrees deliberately

Keep additional local worktrees only while their paths serve active work.
Normally use one development worktree per active task, not one per region or
run attempt. Reuse a suitable existing checkout only after confirming its
branch, file state, and current users; never commandeer another task's path.

A paused branch usually needs no checkout. Preserve all unpublished commits
and necessary tracked, untracked, and ignored files before removing an idle
worktree. Git cleanliness alone does not establish that its data is disposable.

The existing main checkout at `/home/mhpereir/work/projects/HW_analysis` mixes
code and generated results. Preserve it and its `results/` tree; do not move,
clean, or reorganize them as a side effect of adopting this lifecycle. A
separate, explicitly authorized migration is required to change that layout.
New disposable source checkouts must not become the only home of durable
research outputs. Preserve existing retrieved artifacts and recorded Venus
output namespaces independently of any checkout being retired.

## Validate and integrate the exact candidate

Integration requires coherent commits, a reviewed diff against current
`origin/master`, updated documentation and tests, passing applicable formatting
and lint checks, and the full local test suite in `dev_env`. Do not weaken
scientific contracts to make a candidate pass.

For changes affecting ingestion, product construction, scientific defaults, or
production plotting, also validate a representative short Venus PBS smoke run
of the exact candidate after authorized commit, push, and deployment. Check
the affected product contracts and provenance; inspect affected figures at
original resolution when visual behavior changes. A successful scheduler exit
alone is insufficient. Documentation-only changes need no Venus run when they
do not alter executable or scientific behavior; verify their links and
consistency locally and run the full local suite as required by `AGENTS.md`.

Use this integration sequence:

1. Refresh the published baseline. If `master` advanced beyond the task's
   ancestry, merge `origin/master` into the task branch locally. Resolve any
   conflicts locally and validate the resulting candidate, including the
   applicable Venus checks. Preserve existing published and run-pinned SHAs.
2. Record the exact candidate SHA and its validation evidence. Uncommitted
   source changes or evidence for an older candidate do not satisfy this gate.
3. Recheck the task tip, the published baseline, and the clean integration
   checkout. If they changed, reassess and repeat the necessary validation.
4. With integration authorization, fast-forward `master` to the validated SHA
   using `git merge --ff-only`. Do not resolve a new merge on `master` after
   testing a different candidate. Stop on divergence rather than forcing it.
5. With publication authorization, push without rewriting history and verify
   that the published integration ref resolves to the intended SHA. A
   concurrent update or rejected push requires a refreshed review, not force.
6. Run the cleanup skill's audit and provide the retirement assessment. Carry
   out removal only within the request's authorized scope.

This preserves task-commit ancestry and allows `master` to identify the exact
validated candidate. It does not require a squash merge or a new integration
commit after candidate validation.

## Deploy and accept scientific results separately

A merge establishes code integration, not acceptance of all downstream
campaigns. Representative pre-integration validation and full campaign
acceptance are distinct gates; record what each actually covers.

- Follow the local-to-Venus pipeline: develop locally, validate, commit and
  push with authorization, deploy the exact clean commit, then run through
  OpenPBS. Never edit or reconcile source directly on Venus.
- Prefer immutable commit-pinned checkouts for queued or overlapping runs.
  Record the full SHA and require `PROJECT_ROOT` and `EXPECTED_COMMIT` checks.
  Do not advance or remove a checkout while queued/running jobs depend on it.
- Keep durable data, figures, logs, checksums, and immutable provenance outside
  disposable development checkouts. Preserve separate no-overwrite output
  namespaces for new attempts; source retirement must not delete artifacts.
- Record each durable attempt separately through `command-center`. Keep run
  state, scientific acceptance, task completion, and Git retirement distinct.
  Preserve task/run history rather than deleting it during cleanup.
- A merged development branch and its local worktree may be retired while a
  campaign continues if neither the branch name nor that checkout is needed,
  the exact run commits remain durably reachable, and the required deployment
  checkout and artifacts remain available. Check dependencies separately on
  each host; local retirement does not authorize remote checkout removal.

## Pause, archive, or retire

Use the shared cleanup skill for the actual audit and execution. Never decide
from branch age, an old task label, a clean status, or apparent inactivity
alone. These dispositions are not new command-center status values.

| Disposition | Branch ref | Linked worktree |
| --- | --- | --- |
| Active | Retain | Retain while its path is needed |
| Paused | Retain all unfinished and unpublished history | Remove only when idle and necessary files are preserved |
| Fully integrated, no remaining dependency | Delete within authorized local/remote scope after ancestry checks | Remove after file and usage checks |
| Abandoned, worth preserving | Verify an annotated archive tag or backup before deletion | Remove after preservation and usage checks |
| Protected or uncertain | Retain until reviewed and resolved | Retain while its path may still be needed |

For fully integrated work, prove each retiring tip is an ancestor of the
verified published integration commit. Check the live remote tip separately
before remote deletion. A successful `git branch -d` is not this proof: its
guard uses the branch's configured upstream, or HEAD when no upstream exists.

Ordinary merged branches need no archive tag. Preserve valuable unmerged work
with a verified annotated `archive/YYYY-MM-DD/topic` tag or appropriate backup
before deleting its branch. A local tag is not an off-machine backup; publish
and verify it only when remote archival is authorized. Preserve divergent
local and remote histories and unsaved files separately where needed. Never
merge an unwanted experiment merely to make its branch deletable.

Check editors, agents, scripts, deployments, queued/running jobs, and artifact
dependencies before removing a path or ref. Do not bypass a dirty or locked
worktree with force, remove directories recursively, or prune objects as
routine cleanup. Follow the skill's recovery and post-removal verification
procedure, keeping research artifacts and unrelated work intact.

## Required handoff

At each integration, pause, and final handoff, report:

- the task/outcome, branch, final SHA, and relevant checkout paths;
- the verified integration ref or preservation tag/backup, including whether
  the work is still unpublished or uncommitted;
- validation completed and still outstanding, with evidence locations;
- artifact and deployment paths that must survive;
- branch and checkout retirement eligibility, assessed separately;
- any retained dependency, unresolved decision, and concrete next action; and
- what was actually removed and how preserved work can be recovered.

When cleanup is not authorized, report candidates without deleting them.
When a dependency blocks retirement, retain only what is still needed and
revisit it when that dependency ends. Do not automatically complete scientific
tasks merely because a branch was merged or its checkout was removed.
