// V7-H04: spec schema versioning + migrate-on-load.
//
// `CURRENT_SCHEMA_VERSION` is the integer that fresh specs are stamped
// with. Every breaking change to the persisted spec shape (renaming a
// brick kind, removing a param, changing a default) must:
//   1. Bump CURRENT_SCHEMA_VERSION.
//   2. Add a pure migrate_v{n-1}_to_v{n} function below.
//   3. Append it to MIGRATIONS (keyed by source version).
//
// migrate(spec) walks from the spec's stored version up to current and
// returns the migrated object. Specs from a future version (i.e. saved
// by a newer build) throw — the caller is expected to surface this as
// a UI error rather than silently corrupting the spec.

export const CURRENT_SCHEMA_VERSION = 1;

export type SchemaVersion = number;

export interface VersionedSpec {
  schema_version?: SchemaVersion;
  [key: string]: unknown;
}

export class FutureSchemaError extends Error {
  constructor(version: SchemaVersion) {
    super(
      `spec schema_version=${version} is newer than this build's ` +
      `CURRENT_SCHEMA_VERSION=${CURRENT_SCHEMA_VERSION}; upgrade the ` +
      `app or open the spec in a matching build.`,
    );
    this.name = "FutureSchemaError";
  }
}

/** v0 → v1: stamp schema_version=1 onto pre-versioning specs. */
function migrate_v0_to_v1(spec: VersionedSpec): VersionedSpec {
  return { ...spec, schema_version: 1 };
}

const MIGRATIONS: Record<number,
  (s: VersionedSpec) => VersionedSpec> = {
  0: migrate_v0_to_v1,
};

export function migrate(spec: VersionedSpec): VersionedSpec {
  let current: VersionedSpec = { ...spec };
  let v = typeof current.schema_version === "number"
    ? current.schema_version
    : 0;
  if (v > CURRENT_SCHEMA_VERSION) {
    throw new FutureSchemaError(v);
  }
  while (v < CURRENT_SCHEMA_VERSION) {
    const step = MIGRATIONS[v];
    if (!step) {
      throw new Error(
        `V7-H04: no migration registered for schema_version=${v}; ` +
        `add migrate_v${v}_to_v${v + 1} to MIGRATIONS`,
      );
    }
    current = step(current);
    v = typeof current.schema_version === "number"
      ? current.schema_version
      : v + 1;
  }
  return current;
}
