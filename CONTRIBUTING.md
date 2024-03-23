# Contributing Guidelines

## Golden Rules

### DON'T MERGE, ALWAYS SQUASH AND REBASE
### CREATE WORK BRANCHES FROM DEVELOP

## Commits

- **Capitalized**, short but **descriptive** names: `Add foo function`;
- Do **small commits** (don't make a single commit after editing thousand lines of code).

## Issues

- **Descriptive** name: `Update README with installation instructions`;
- Add issues to the **kanban**.

## Pull Requests

- Name it **as the related issue**;
- Don't forget to **select the `develop` branch when merging a work branch**;
- **Squash** from work branch to `develop`;
- **Rebase** from `develop` to `main`.

## Workflow

1. Create a feature/bugfix/doc branch from an issue. **The source branch must be `develop`**;
2. Once the work done and the PR approved, merge work branch into `develop`;
3. Once develop is **stable** with enough changes, merge it into `main`.

## Format

**Format your code**! You can enable auto-formatting on save.
