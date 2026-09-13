#!/usr/bin/env bash
# Build GitHub Release notes from the conventional commits since the previous
# tag, writing them to the path given as $1.
#
# Shared by both release paths -- ci.yml's auto-publish job and release.yml's
# dispatch job -- so the two can never drift into describing the same commits
# differently.
#
# Neither path drafts, so this output ships to users unreviewed: the commit
# subjects it collects are the release notes, not a starting point for them.
#
# Runs on both ubuntu-latest and macos-latest, and macOS ships bash 3.2, so
# nothing here may use bash 4 syntax (no arrays, no `${x,,}`).
#
# Required env:
#   VERSION   the new version, without the `v` prefix (e.g. 0.9.1)
#   REPO_URL  https://github.com/<owner>/<repo>
set -euo pipefail

out="$1"
: >"$out"

# Must be read BEFORE the new tag is created, or the range collapses to nothing.
prev="$(git describe --tags --abbrev=0 2>/dev/null || true)"

commit_subjects() {
  if [ -n "$prev" ]; then
    git log --no-merges --format='%s' "$prev..HEAD"
  else
    # First release: no previous tag, so the range is the whole history. Spelled
    # as two calls rather than an interpolated range, because the empty-range
    # form needs an unquoted expansion and bash 3.2 has no safe empty array.
    git log --no-merges --format='%s'
  fi
}

# grep exits 1 when a type is absent from the range, which `set -o pipefail`
# would otherwise turn into a failed run -- hence the `|| true`.
#
# The scope is `[^)]+`, not `.+`. POSIX ERE matching is leftmost-longest, so
# `.+` let the scope group swallow a `!` and everything up to a later `): `
# inside the subject: `feat(ui)!: drop X (and Y): text only` was filed under
# Breaking changes AND under Features as `- text only`, and a non-breaking
# `feat(ui): foo (bar): baz` rendered as `- baz`. An unscoped `feat!:` was
# never affected, since `\(` has to follow the type directly.
section() {
  body="$(commit_subjects | grep -E "^$1(\([^)]+\))?: " | sed -E "s/^$1(\([^)]+\))?: /- /" || true)"
  if [ -n "$body" ]; then
    printf '### %s\n\n%s\n\n' "$2" "$body" >>"$out"
  fi
}

# A `!` before the colon marks a breaking change (conventional commits), which
# on 0.x is what removing a feature is. Those subjects go first, under their
# own heading, and only there: `section`'s pattern has no `!`, so a breaking
# commit is never also filed under its type -- and before this existed a
# `feat!:` subject matched nothing and silently vanished from the notes. Any
# type qualifies, including the four omitted below: `build!: require Python
# 3.14` is user-facing however it is typed, and the `!` is the signal. Only
# the subject form is read; a `BREAKING CHANGE:` footer lives in the body,
# which these notes never see. Needs its `|| true` for the same reason
# `section` does.
breaking() {
  body="$(commit_subjects | grep -E '^[a-z]+(\([^)]+\))?!: ' | sed -E 's/^[a-z]+(\([^)]+\))?!: /- /' || true)"
  if [ -n "$body" ]; then
    printf '### %s\n\n%s\n\n' "Breaking changes" "$body" >>"$out"
  fi
}

# chore/test/ci/build are deliberately omitted unless marked `!` -- the compare
# link below covers them, and they are noise in user-facing notes.
breaking
section feat "Features"
section fix "Fixes"
section perf "Performance"
section refactor "Refactoring"
section docs "Documentation"

if [ -n "$prev" ]; then
  printf '**Full changelog:** %s/compare/%s...v%s\n' \
    "$REPO_URL" "$prev" "$VERSION" >>"$out"
fi
