# Release New Version

Prepare and commit a new crate release. The version number will be provided as a parameter (e.g., `/release 0.5.0`).

## Steps

### 1. Gather commits since last release

Find the last version commit by searching git history for commits matching a version pattern (e.g., "0.4.0", "Release 0.4.0"). Then list all commits since that point:

```bash
git log --oneline <last-version-commit>..HEAD
```

### 2. Update CHANGELOG.md

Add a new version section at the top (below the header), following the existing format:

```markdown
## [X.X.X] - YYYY-MM-DD

### Added
- New features...

### Changed
- Changes to existing functionality...

### Fixed
- Bug fixes...

### Removed
- Removed features...
```

Only include sections that have entries. Write clear, user-facing descriptions based on the commits. Group related changes together.

### 3. Update Cargo.toml

Change the `version` field to the new version:

```toml
version = "X.X.X"
```

### 4. Update the lockfile

Run cargo to regenerate `Cargo.lock` with the new version:

```bash
cargo check
```

### 5. Format and lint

Format code and verify it passes CI checks:

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
```

If clippy reports warnings, fix them before proceeding.

### 6. Commit the release

Create a commit with just the version number as the message:

```bash
git add -A
git commit -m "X.X.X"
```

### 7. Confirm ready to publish

After the commit succeeds, remind the user they can now run:

```bash
cargo publish
```

## Notes

- The version parameter is required (e.g., `/release 0.5.0`)
- Use today's date for the changelog entry
- Follow semantic versioning: MAJOR.MINOR.PATCH
- Breaking changes should be clearly marked with **BREAKING:**
