# Zone Workspace Dependencies

Use this skill when a workspace needs tools, services, environment variables, ports, mounts, or setup commands added to its Zone container.

Call this skill as `zone-workspace-dependencies`.

Zone configuration lives in `zone.toml`. Update that file instead of editing generated files under `.zone/`.

## Workflow

1. Inspect the project to identify the dependency and how it should be available inside the container.
2. Edit `zone.toml` using the smallest relevant section.
3. Run `zone validate` after changes.
4. Tell the user when a rebuild or restart is needed, usually `zone restart --rebuild` for changes that affect the image.

## Common Changes

- System packages: add apt package names to `[packages].apt`.
- Language tooling: prefer project-native manifests first; add Python packages to `[packages].pip` or Node packages to `[packages].npm` only when the container image needs them.
- Ports: add mappings to `[workspace].ports`, for example `"3000:3000"`.
- Mounts: add explicit paths to `[workspace].extra_mounts` and keep read-only mounts read-only when write access is not required.
- Environment: add safe host variables to `[auth].forward_env`.
- Setup commands: add deterministic build-time commands to `[hooks].pre_build`.
- Network access: update `[network].allow` or `[network].deny` only for endpoints the dependency actually needs.

## Guardrails

- Do not edit files in `.zone/`; Zone regenerates them.
- Do not weaken sandboxing broadly when a narrow config change works.
- Do not mount secrets or host directories unless the task explicitly requires it.
- Keep comments in `zone.toml` useful for future agents and remove stale commented examples when they become misleading.
