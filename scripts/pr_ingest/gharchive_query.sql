-- Tier-2 PR ingest, component (2): GH Archive BigQuery extraction.
--
-- Source: the public `githubarchive.month.20*` sharded tables. Each row is one
-- raw GitHub events-API event; `payload` is a JSON STRING, so all PR/comment
-- fields are pulled with JSON_EXTRACT_SCALAR(payload, '$....').
--
-- We extract the four event types that carry PR text + discussion + reviews +
-- linked issues, restricted to the corpus repos, and write a small materialized
-- table that the local export step then downloads.
--
-- RULE #1 / cost discipline:
--   * NEVER `SELECT *`. We only read the columns we use: type, repo.name,
--     created_at, and `payload` (only in the final projection).
--   * Partition-prune with _TABLE_SUFFIX BETWEEN @suffix_start AND @suffix_end
--     so only the needed monthly shards are scanned (GH Archive months are
--     'YYMM' suffixes, e.g. '1501' = Jan 2015 .. current month).
--   * ALWAYS run with --dry_run first to see bytes-scanned / cost, then run for
--     real. See gharchive_run.sh.
--
-- Parameters (passed via bq --parameter):
--   @suffix_start STRING  -- e.g. '1501'  (Jan 2015; GH Archive monthly start)
--   @suffix_end   STRING  -- e.g. '2606'  (current month, YYMM)
--   @repo_names   ARRAY<STRING>  -- ['opencv/opencv','llvm/llvm-project',...]
--
-- The destination table is set on the bq command line (see gharchive_run.sh):
--   bq query --destination_table <proj>:<ds>.pr_discussion_raw --replace ...

SELECT
  -- Repo + event identity
  repo.name AS repo_name,
  type AS event_type,
  created_at,

  -- PR identity / join keys. IssueCommentEvent carries the number under
  -- $.issue.number (and is a PR iff $.issue.pull_request is non-null);
  -- the PR/review events carry it under $.pull_request.number.
  COALESCE(
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.number'),
    JSON_EXTRACT_SCALAR(payload, '$.issue.number')
  ) AS pr_number,
  JSON_EXTRACT_SCALAR(payload, '$.pull_request.merge_commit_sha') AS merge_commit_sha,
  JSON_EXTRACT_SCALAR(payload, '$.action') AS action,
  JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') AS pr_merged,

  -- PR title/body (only present on PullRequestEvent / review events).
  JSON_EXTRACT_SCALAR(payload, '$.pull_request.title') AS pr_title,
  JSON_EXTRACT_SCALAR(payload, '$.pull_request.body') AS pr_body,

  -- Discriminator: non-null only when the issue IS a PR.
  JSON_EXTRACT_SCALAR(payload, '$.issue.pull_request.url') AS issue_is_pr_url,
  JSON_EXTRACT_SCALAR(payload, '$.issue.title') AS issue_title,
  JSON_EXTRACT_SCALAR(payload, '$.issue.body') AS issue_body,

  -- Comment / review bodies (the discussion thread).
  JSON_EXTRACT_SCALAR(payload, '$.comment.body') AS comment_body,
  JSON_EXTRACT_SCALAR(payload, '$.comment.user.login') AS comment_user,
  JSON_EXTRACT_SCALAR(payload, '$.comment.path') AS comment_path,
  JSON_EXTRACT_SCALAR(payload, '$.review.body') AS review_body,
  JSON_EXTRACT_SCALAR(payload, '$.review.state') AS review_state,
  JSON_EXTRACT_SCALAR(payload, '$.review.user.login') AS review_user

FROM
  `githubarchive.month.20*`
WHERE
  _TABLE_SUFFIX BETWEEN @suffix_start AND @suffix_end
  AND type IN (
    'PullRequestEvent',
    'IssueCommentEvent',
    'PullRequestReviewCommentEvent',
    'PullRequestReviewEvent'
  )
  AND repo.name IN UNNEST(@repo_names)
  -- For IssueCommentEvent, keep only PR comments (issue.pull_request present).
  AND (
    type != 'IssueCommentEvent'
    OR JSON_EXTRACT_SCALAR(payload, '$.issue.pull_request.url') IS NOT NULL
  )
ORDER BY
  repo_name, pr_number, created_at
