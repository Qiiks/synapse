# Repo-local preflight sourced by scripts/train-push.sh before a train is
# pushed: the same workflow-shape assertions CI runs as its first step, run
# here first because a drifted workflow would fail in seconds on CI anyway and
# failing locally is free.
bash scripts/check-train-preconditions.sh || refuse "train preconditions failed — see scripts/check-train-preconditions.sh"

# CI compiles 7 of the 22 workspace members. The Metal engine and the ANE and
# MLX workers need a Mac, and the macos runner left with the M1 box, so this
# preflight is the ONLY automated gate they have — synapse-engine-owned is the
# primary production engine on this platform and would otherwise reach master
# having been compiled nowhere but an author's terminal. Warm cost is about ten
# seconds; a cold build is slower and still cheaper than shipping it unbuilt.
# Delete this block the day a Mac runner rejoins CI, not before.
if [ "$(uname -s)" = "Darwin" ]; then
  mac_crates="-p synapse-engine-owned -p synapse-worker-ane -p synapse-worker-mlx"
  # The Metal toolchain lives in full Xcode; Command Line Tools alone cannot
  # compile the shaders, and the failure is a confusing linker error rather
  # than a missing-tool message.
  if [ -d /Applications/Xcode.app ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
  fi
  # These two commands rewrite Cargo.lock as a side effect, because the sibling
  # path dependencies (../subconscious, ../commons) are other agents' working
  # checkouts and routinely sit ahead of siblings.lock. That rewrite dirties the
  # tree, and the NEXT train then refuses on a change this script made itself.
  #
  # --locked does not solve it: it refuses to run at all whenever a sibling has
  # moved, which here is most of the time, and those checkouts are not ours to
  # roll back. CI is unaffected either way because it checks the siblings out at
  # the pinned commits. So verify against whatever is on disk, then put the lock
  # back exactly as it was — and leave a deliberate lock edit alone.
  lock_was_clean=no
  git diff --quiet -- Cargo.lock 2>/dev/null && lock_was_clean=yes
  restore_lock() {
    if [ "$lock_was_clean" = yes ]; then
      git checkout -- Cargo.lock 2>/dev/null || true
    fi
  }
  # shellcheck disable=SC2086
  cargo clippy $mac_crates --all-targets -- -D warnings \
    || { restore_lock; refuse "macOS-only crates failed clippy (CI cannot run these; see scripts/train-push.local.sh)"; }
  cargo test -p synapse-engine-owned --lib \
    || { restore_lock; refuse "synapse-engine-owned lib tests failed (CI cannot run these)"; }
  restore_lock
fi
